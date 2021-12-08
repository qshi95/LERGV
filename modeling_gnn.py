from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)


import torch
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GATConv
import copy
from torch.autograd import Variable

def get_target_index(x, y):
    x = x.sort()[0]
    x_new = x.repeat(y.size(0)).view(y.size(0), x.size(0))
    y_new = y.repeat(x.size(0)).view(x.size(0), y.size(0)).t()
    return torch.nonzero(x_new==y_new)[:,0]

def aug(t):
    return torch.cat((torch.Tensor([0]).long().to(t.device), t), 0)

def remove_nodes_func(parent, remove):
    for i in range(5):
        parent_remove = aug(remove).index_select(0, parent)
        new_parent = parent_remove * (aug(parent).index_select(0, parent)) + (1 - parent_remove) * parent
        if not (parent - new_parent).sum():
            break
        parent = new_parent
    
    return parent

def remove_nodes_and_edges(edge_index, edge_index_ent, remove_nodes):
    index = get_target_index(remove_nodes, edge_index[0])
    edge_mask = torch.ones(edge_index.size(1)).to(edge_index.device).index_fill_(0, index, 0)
    edge_mask_reverse = torch.zeros(int(edge_index[0][-1])).to(edge_index.device).index_fill_(0, remove_nodes.long()-1, 1)
    temp = remove_nodes_func(edge_index[1].long(), edge_mask_reverse.byte())
    edge_index_new = torch.cat([edge_index[0].long().unsqueeze(0), temp.unsqueeze(0)], dim=0) * edge_mask.unsqueeze(0).long()

    return edge_index_new, edge_index_ent, torch.cat([torch.Tensor([1]).to(edge_mask_reverse.device), (1 - edge_mask_reverse)], dim=0).bool()

class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)
        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn

class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class MLP(nn.Module):
    """
    Multi-layer perceptron
    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class GATMultiLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1, heads=4, num_layers=1):
        super().__init__()
        self.heads = heads

        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList([GATConv(input_size * 2 + output_size, output_size, heads=heads, dropout=dropout) for _ in range(num_layers)])

        self.n_node_type = 2
        self.emb_node_type = nn.Linear(self.n_node_type, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout
        self.scorer = nn.Linear(input_size * 2, 1)

    def multiple_layer_gnn(self, x, edge_index, edge_index_ent, statement_rep, node_mask, type_emb):
        for _ in range(self.num_layers):
            
            node_score = self.scorer(torch.cat([x, statement_rep], dim=1))
            node_score = node_score.squeeze(1) * node_mask
            nonzero_index = torch.nonzero(node_score).squeeze(1)
            nonzero_scores = torch.index_select(node_score, 0, nonzero_index)
            if int(len(nonzero_scores) * 0.3) != 0:
                to_remove_index = torch.topk(nonzero_scores, int(len(nonzero_scores) * 0.3+1), largest=False, sorted=True)[1]
                map_index = torch.index_select(nonzero_index, 0, to_remove_index)
                edge_index, edge_index_ent, node_mask_upd = remove_nodes_and_edges(edge_index, edge_index_ent, map_index)
                node_mask = node_mask * node_mask_upd

            x = torch.cat([x, statement_rep, type_emb], dim=1)
            x = self.gnn_layers[_](x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, self.dropout_rate)
        return x, node_mask
    
    def make_one_hot(self, type_id, n_node_type):
        type_id = type_id.unsqueeze(1)
        one_hot = torch.FloatTensor(type_id.size(0), n_node_type).zero_().to(type_id.device)
        target = one_hot.scatter_(1, type_id.data, 1)
        target = Variable(target)
        return target

    def forward(self, x, edge_index, edge_index_ent, statement_rep, node_mask, type_id):
        
        batch_size, max_len = node_mask.shape
        node_mask = node_mask.view(-1)
        type_id = type_id.view(-1)
        one_hot = self.make_one_hot(type_id, self.n_node_type)
        type_emb = self.activation(self.emb_node_type(one_hot))
        x, node_mask = self.multiple_layer_gnn(x, edge_index, edge_index_ent, statement_rep, node_mask, type_emb)

        return x, node_mask.view(batch_size, max_len)