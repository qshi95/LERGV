import re

class Node():
    def __init__(self, bert_index, type=['statement', 'table'], column_index=None, index=None, value=None, word=None):
        self.value = value
        self.index = index
        self.type = type
        self.column_index = column_index
        self.bert_index = bert_index
        self.word = word

class Tree(object):
    def __init__(self, idx, word, bert_index=None, map_node_idx=None):
        self.idx = idx
        self.word = word
        self.child = []
        self.bert_index = bert_index
        self.map_node_idx = map_node_idx
        self.parent = None
        self.map_parent_idx = None
        
    def add_child(self, child):
        if child and not isinstance(child, Tree):
            raise ValueError('TreeNode only add another TreeNode obj as child')
        child.parent = self

        self.child.append(child)
    
def build_tree(program, l, r):
    global idx
    global node_list
    node_list.append(Tree(idx=idx, word=program[l]))
    root_idx, idx = idx, idx+1
    l += 1
    
    if program[l] == '{':
        l += 1
        while l <= r:
            sub_root_idx, l = build_tree(program, l, r)
            node = [node for node in node_list if node.idx==sub_root_idx]
            assert len(node) == 1
            node = node[0]
            node_list[root_idx].add_child(node)
            if l <= r and program[l] == ';':
                l += 1
                continue
            elif l <= r and program[l] == '}':
                l += 1
                return root_idx, l
            elif l > r:
                return root_idx, l
    elif program[l] == ';':
    	return root_idx, l+1
    elif program[l] == '}':
        return root_idx, l

def is_split(token):
    return token == ';' or token == '{' or token == '}'

def parse_program(program):

    program = program.split()
    program = [item.strip() for item in program]
    global idx
    idx = 0
    tmp = []
    for i in range(len(program)):
        if i > 1 and not is_split(program[i]) and not is_split(program[i-1]):
            tmp[-1] = tmp[-1] + ' ' + program[i]
        else:
            tmp.append(program[i])
    global node_list
    node_list = []
    build_tree(tmp, 0, len(tmp)-1)
    return node_list

def batched_parse_program(program_list):
    parsed_list = []
    temp = 0
    for program in program_list:
        parsed_program = parse_program(program)
        for node in parsed_program:
            node.idx = node.idx + temp
        temp += len(parsed_program)
        parsed_list.extend(parsed_program)
    
    return parsed_list

def mask_program(program):
    program = program.split()
    func_list = list(APIs.keys()) + ['within', 'not_within', 'uniq', 'hop', 'all_rows']
    for i in range(len(program)-1):
        if program[i] not in ['{', ';', '}', 'all_rows'] and program[i+1] in ['{', ';', '}', 'all_rows']:
            if program[i+1] == '}' or program[i+1] == ';':
                program[i] = '[ENT]'
    
    for i in range(len(program)):
        if program[i] not in ['{', ';', '}', 'all_rows'] and program[i] not in func_list:
            program[i] = '[ENT]'
    
    return ' '.join(program)

if __name__ == '__main__':
    program1 = "less { count { filter_eq { all_rows ; storm name ; frank }  }  ; count { filter_eq { all_rows ; storm name ; estelle }  }  }"
    program2 = "eq { 5 ; count { filter_eq { all_rows ; competition ; south american championship }  }  } "
    program3 = "eq { 40 ; hop { filter_eq { all_rows ; player ; terry west }  ; pick }  }"
    # result = batched_parse_program([program1, program2, program3])

    x = parse_program(program1)
    print(x[1].parent)
    print(x[0])
    print(x[2])
    if x[1].parent == x[0]:
        print('True')
    for item in x:
        print(item.word)