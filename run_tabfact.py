# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from typing import List

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    # AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from datasets import Features, Sequence, Value, ClassLabel, Array2D
from modeling_auto import AutoModelForSequenceClassification
from node import parse_program, batched_parse_program
from itertools import combinations, permutations
from APIs import APIs


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "tab_fact":("table_text", "statement")
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
    default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
    simple_test_file: Optional[str] = field(
    default=None, metadata={"help": "A csv or a json file containing the simple_test data."}
    )
    complex_test_file: Optional[str] = field(
    default=None, metadata={"help": "A csv or a json file containing the complex_test data."}
    )
    small_test_file: Optional[str] = field(
    default=None, metadata={"help": "A csv or a json file containing the small_test data."}
    )
    train_cache_file: Optional[str] = field(
        default=None, metadata={"help": "train dataset cache file."}
    )
    validation_cache_file: Optional[str] = field(
        default=None, metadata={"help": "validation dataset cache file."}
    )
    test_cache_file: Optional[str] = field(
    default=None, metadata={"help": "test dataset cache file."}
    )
    simple_test_cache_file: Optional[str] = field(
    default=None, metadata={"help": "simple_test dataset cache file."}
    )
    complex_test_cache_file: Optional[str] = field(
    default=None, metadata={"help": "complex_test dataset cache file."}
    )
    small_test_cache_file: Optional[str] = field(
    default=None, metadata={"help": "small_test dataset cache file."}
    )
    test_set: Optional[str] = field(
    default="test", metadata={"help": "which test set is chosen, options: test, simple_test, complex_test, small_test"}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.task_name == 'tab_fact':
        datasets = load_dataset('json', data_files={"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file, "simple_test": data_args.simple_test_file, "complex_test": data_args.complex_test_file, "small_test": data_args.small_test_file})
    
    if training_args.do_predict:
        datasets_for_predict = datasets[data_args.test_set]   # raw dataset without tokenization for prediction
    
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name == 'tab_fact':
        num_labels = 2
        is_regression = False
        label_list = ['refuted', 'entailed']
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def parse_table_text(table_text):
        return '\n'.join(['#'.join(item) for item in table_text])

    def get_map_tokenize(tokenizer, text, tokens):    
        text_list = text.strip().split()
        temp = 0
        map_list = []
        token_list = []
        for item in text_list:
            map_list.append(list(range(temp,temp+len(tokenizer.tokenize(item)),1)))
            token_list.append(item)
            temp += len(tokenizer.tokenize(item))
        assert temp == len(tokens)
        temp_list = [(item1, item2) for item1, item2 in zip(map_list, token_list) if item2 not in ['{', '}', ';', '[CLS]', '[SEP]', '.']]
        map_list = [item[0] for item in temp_list]
        token_list = [item[1] for item in temp_list]
        return map_list, token_list

    def preprocess_function(example):
        # Tokenize the texts

        func_list = list(APIs.keys()) + ['within', 'not_within', 'uniq', 'hop']
        type_map_dict = {'func':0, 'ent':1}
        temp = parse_table_text(example[sentence1_key])
        args = (
            (temp, ) if sentence2_key is None else (temp, example[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        parsed_program_list = batched_parse_program(example['program_aug'])
        program_id_list = []
        for idx in range(len(parsed_program_list)):
            program_id_list.extend([idx] * len(parsed_program_list[idx].word.split()))
        to_tokenize_programs = '[CLS] ' + ' . '.join(example['program_aug']) + ' [SEP]'
        to_tokenize_programs = to_tokenize_programs.replace('  ', ' ')
        map_list, token_list = get_map_tokenize(tokenizer, to_tokenize_programs, tokenizer.tokenize(to_tokenize_programs))
        assert len(program_id_list) == len(map_list) == len(token_list)
        map_dict = dict()
        for program_id, bert_id, token in zip(program_id_list, map_list, token_list):
            if program_id not in map_dict:
                map_dict[program_id] = bert_id
            else:
                map_dict[program_id].extend(bert_id)
        assert len(map_dict) == len(parsed_program_list)
        assert list(map_dict.keys()) == list(range(len(parsed_program_list)))
        for item in parsed_program_list:
            item.bert_index = map_dict[item.idx]
            item.type = 'func' if item.word in func_list else 'ent'
        parsed_program_list = [item for item in parsed_program_list if not item.bert_index[-1] >= max_length - len(tokenizer.tokenize(example[sentence2_key]))]  # trun
        # construct edges
        edge_list = []
        edge_list_ent = []
        perm_list = list(permutations(range(len(parsed_program_list)), 2))
        for i in range(len(perm_list)):
            if parsed_program_list[perm_list[i][0]].parent == parsed_program_list[perm_list[i][1]]:
                edge_list.append(tuple([item+1 for item in perm_list[i]]))
            if parsed_program_list[perm_list[i][0]].word == parsed_program_list[perm_list[i][1]].word != 'all_rows' and parsed_program_list[perm_list[i][0]].type == parsed_program_list[perm_list[i][1]].type == 'ent':
                edge_list_ent.append(tuple([item+1 for item in perm_list[i]]))

        if len(edge_list) == 0:
            to_tokenize_programs = '[CLS] [UNK] [SEP]'
            # edge_list.append((0,0))
            edge_list.append((1,1))
        for i in range(1, edge_list[-1][0]):
            if i not in [item[0] for item in edge_list]:
                edge_list.append((i,0))
        edge_list.sort(key=lambda x:x[0])
        if len(edge_list) < 200:
            for i in range(1,200):
                if i not in [item[0] for item in edge_list]:
                    edge_list.append((i,i))
        else:
            edge_list = edge_list[:199]
        edge_array = np.array(edge_list)

        if len(edge_list_ent) == 0:
            edge_list_ent.append((1,1))
        if len(edge_list_ent) < 200:
            for i in range(1,200):
                if i not in [item[0] for item in edge_list_ent]:
                    edge_list_ent.append((i,i))
        edge_list_ent = edge_list_ent[:199]
        edge_array_ent = np.array(edge_list_ent)

        args = (
            (to_tokenize_programs[6:-6], ) if sentence2_key is None else (to_tokenize_programs[6:-6], example[sentence2_key])
        )
        result_temp = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)
        for key in result_temp:
            result_temp[key+'_program'] = result_temp.pop(key)
        result.update(result_temp)
        sep_index_list = [key for key, value in list(enumerate(result['input_ids_program'])) if value == 102]
        assert len(sep_index_list) == 2
        statement_start_index_list = [sep_index_list[0]+1]
        statement_len_list = [sep_index_list[1] - sep_index_list[0]-1]
        result['statement_start_index'] = statement_start_index_list    # statement start index
        result['statement_len_index'] = statement_len_list     # statement len

        edge_first_index = [item[0] for item in edge_array]
        edge_second_index = [item[1] for item in edge_array]
        result['edge_first_index'] = edge_first_index     # start index of edge
        result['edge_second_index'] = edge_second_index    # end index of edge
        assert list(range(1, 200)) == edge_first_index

        edge_first_index_ent = [item[0] for item in edge_array_ent]
        edge_second_index_ent = [item[1] for item in edge_array_ent]
        result['edge_first_index_ent'] = edge_first_index_ent     # start index of edge
        result['edge_second_index_ent'] = edge_second_index_ent    # end index of edge
        assert len(edge_first_index_ent) == 199

        node_start_index_list = [0] + [item.bert_index[0] for item in parsed_program_list]
        node_len_list = [0] + [len(item.bert_index) for item in parsed_program_list]
        type_list = ['ent'] + [item.type for item in parsed_program_list]
        if len(node_start_index_list) < 200:                  # max node num: 200
            node_start_index_list.extend([0] * (200 - len(node_start_index_list)))
            node_len_list.extend([0] * (200 - len(node_len_list)))
            type_list.extend(['ent'] * (200 - len(type_list)))
        else:
            node_start_index_list = node_start_index_list[:200]
            node_len_list = node_len_list[:200]
            type_list = type_list[:200]
        
        node_len_mask_list = []
        for i in range(200):
            if node_len_list[i] < 10:   # assert all node token len is <= 10
                node_len_mask_list.append(list(range(node_start_index_list[i], node_len_list[i]+node_start_index_list[i])) + [0] * (10 - node_len_list[i]))
            else:
                node_len_mask_list.append(list(range(node_start_index_list[i], 10+node_start_index_list[i])))

        result['node_len_mask'] = node_len_mask_list

        result['type_id'] = [type_map_dict[item] for item in type_list]

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in example:
            result["label"] = [label_to_id[l] for l in example["label"]]
                
        return result

    if 'tapas' in model_args.model_name_or_path and data_args.task_name == 'tab_fact':
        def _format_pd_table(table_text: List) -> pd.DataFrame:
            df = pd.DataFrame(columns=table_text[0], data=table_text[1:])
            df = df.astype(str)
            return df
        features = Features({
            'attention_mask': Sequence(Value(dtype='int64')),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'label': ClassLabel(names=['refuted', 'entailed']),
            'statement': Value(dtype='string'),
            'table_caption': Value(dtype='string'),
            'table_id': Value(dtype='string'),
            'token_type_ids': Array2D(dtype="int64", shape=(512, 7))
        })

        datasets = datasets.map(
        lambda e: tokenizer(table=_format_pd_table(e['table_text']), queries=e['statement'],
                            truncation=True,
                            padding='max_length'),
        features=features,
        remove_columns=['table_text'],
    )

    else: 
        datasets = datasets.map(preprocess_function, batched=False, load_from_cache_file=not data_args.overwrite_cache, cache_file_names={'train':data_args.train_cache_file, 'validation':data_args.validation_cache_file, 'test':data_args.test_cache_file, 'simple_test':data_args.simple_test_cache_file, 'complex_test':data_args.complex_test_cache_file, 'small_test':data_args.small_test_cache_file})

    if data_args.task_name is not None and training_args.do_train:
        train_dataset = datasets["train"]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.task_name is not None and training_args.do_eval:
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name is not None and training_args.do_predict:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else data_args.test_set]


    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("accuracy")

    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        # test_evaluation
        for test_dataset, task in zip(test_datasets, tasks):
            test_result = trainer.evaluate(eval_dataset=test_dataset)

            output_test_file = os.path.join(training_args.output_dir, f"test_eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    for key, value in test_result.items():
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(test_result)


        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            output_test_file = os.path.join(training_args.output_dir, f"test_results_prediction_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("table_id\ttable_caption\tstatement\tgold\tprediction\n")
                    for pred, dat in zip(predictions, datasets_for_predict):
                        table_id, table_caption, statement, gold_label = dat['table_id'], dat['table_caption'], dat['statement'], dat['label']
                        if is_regression:
                            writer.write(f"{table_id}\t{table_caption}\t{statement}\t{gold_label}\t{pred:3.3f}\n")
                        else:
                            pred = str(pred)
                            writer.write(f"{table_id}\t{table_caption}\t{statement}\t{gold_label}\t{pred}\n")

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()