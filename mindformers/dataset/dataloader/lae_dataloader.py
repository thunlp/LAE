# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""LAE DataLoader."""
import json
import os
import random

from mindspore.dataset import GeneratorDataset

from mindformers.tools import logger
from ...tools.register import MindFormerRegister, MindFormerModuleType
from ...models.bert.bert_tokenizer import BertTokenizer

model_ms_path = "/data3/private/syt/bert/ms"
MAX_CASE = 4
MAX_SEQ_LENGTH = 120 #70
MAX_LENGTH = 256 # 144
MAX_ENCODE_LENGTH = 512
NEGATIVE_SAMPLE_SIZE = 15
TRAIN_SIZE = 20000
OVERLAP_MIN = 0.1

def overlap(a, b):
    ainb = 0
    bina = 0
    for word in a:
        if word in b:
            ainb += 1

    for word in b:
        if word in a:
            bina += 1

    return (ainb / len(a) + bina / len(b)) / 2

def load_case(data_path, train_size):
    pros = []
    cons = []
    sentences = []
    for root, dirs, files in os.walk(data_path):
        for _file in files:
            fp = open(os.path.join(root, _file), 'r')
            texts = json.loads(fp.read())
            for case in texts:
                pro = case[1]
                con = case[2]
                sentences.append((pro, con))

            if len(sentences) >= train_size:
                break

    random.shuffle(sentences)
    sentences = sentences[0:train_size]
    for pro, con in sentences:
        pros.append(pro)
        cons.append(con)
        
    return pros, cons

@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class LAEDataLoader:
    """LAE Dataloader"""
    _default_column_names = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]

    def __new__(cls, dataset_dir, column_names=None, stage="train", num_shards=1, shard_id=0):
        """
        LAE Dataloader API

        Args:
            dataset_dir: the directory to dataset
            column_names: the output column names, a tuple or a list of string with length 2
            stage: the supported `option` are in ["train"、"test"、"del"、"all"]

        Return:
            a GeneratorDataset for LAE dataset
        """
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if column_names is None:
            column_names = cls._default_column_names
            logger.info("The column_names to the LAEDataLoader is None, so assign it with default_column_names %s",
                        cls._default_column_names)

        # if column_names and not isinstance(column_names, (tuple, list)):
        #     raise TypeError(f"column_names should be a tuple or a list"
        #                     f" of string with length 2, but got {type(column_names)}")

        # if len(column_names) != 2:
        #     raise ValueError(f"the length of column_names should be 2,"
        #                      f" but got {len(column_names)}")

        if not isinstance(column_names[0], str) or not isinstance(column_names[1], str):
            raise ValueError(f"the item type of column_names should be string,"
                             f" but got {type(column_names[0])} and {type(column_names[1])}")

        dataset = LAEDataSet(dataset_dir, stage)
        return GeneratorDataset(dataset, column_names, num_shards=num_shards, shard_id=shard_id)


def read_text(train_file):
    """Read the text files and return a list."""
    with open(train_file) as fp:
        data = []
        for line in fp:
            line = line.strip()
            if line:
                data.append(line)
    return data

def pad_lists(lists, pad_length, max_num, pad_value=0):
    """
    Pads each sublist in the main list to a specific length with a specified pad value.

    Args:
    lists (list of lists): The main list containing sublists.
    pad_length (int): The desired length for each sublist.
    pad_value (any): The value to use for padding sublists that are shorter than pad_length.

    Returns:
    list of lists: A new list where each sublist has been padded to the specified length.
    """
    padded_lists = []
    for sublist in lists:
        # Calculate the number of padding elements needed
        extra_padding = pad_length - len(sublist)
        if extra_padding > 0:
            # Extend the sublist with the pad_value
            padded_sublist = sublist + [pad_value] * extra_padding
        else:
            # No padding needed, use the sublist as is
            padded_sublist = sublist[:pad_length]  # This also handles sublists longer than pad_length
        padded_lists.append(padded_sublist)
    while len(padded_lists) < max_num:
        padded_lists.append([pad_value] * pad_length)
    assert len(padded_lists) == max_num
    return padded_lists

def pad_lists_am(lists, pad_length, max_num, pad_value=0):
    """
    Pads each sublist in the main list to a specific length with a specified pad value.

    Args:
    lists (list of lists): The main list containing sublists.
    pad_length (int): The desired length for each sublist.
    pad_value (any): The value to use for padding sublists that are shorter than pad_length.

    Returns:
    list of lists: A new list where each sublist has been padded to the specified length.
    """
    padded_lists = []
    for sublist in lists:
        # Calculate the number of padding elements needed
        extra_padding = pad_length - len(sublist)
        if extra_padding > 0:
            # Extend the sublist with the pad_value
            padded_sublist = [1]*len(sublist) + [pad_value] * extra_padding
        else:
            # No padding needed, use the sublist as is
            padded_sublist = [1]*pad_length # This also handles sublists longer than pad_length
        padded_lists.append(padded_sublist)
    while len(padded_lists) < max_num:
        padded_lists.append([pad_value] * pad_length)
    return padded_lists

def pad_lists_type(lists, pad_length, max_num, pad_value=0):
    """
    Pads each sublist in the main list to a specific length with a specified pad value.

    Args:
    lists (list of lists): The main list containing sublists.
    pad_length (int): The desired length for each sublist.
    pad_value (any): The value to use for padding sublists that are shorter than pad_length.

    Returns:
    list of lists: A new list where each sublist has been padded to the specified length.
    """
    padded_lists = []
    while len(padded_lists) < max_num:
        padded_lists.append([pad_value] * pad_length)
    return padded_lists


class LAEDataSet:
    """LAE DataSet"""

    def __init__(self, dataset_dir, stage="train"):
        """
        LAEDataSet Dataset

        Args:
            dataset_dir: the directory to LAE dataset
            stage: the supported key word are in ["train"、"test"、"del"、"all"]

        Return:
            an iterable dataset for LAE dataset
        """
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")
        self.dataset_dir = dataset_dir
        self.stage = stage

        pros, cons = load_case(dataset_dir, TRAIN_SIZE)
        self.sentences = []
        tokenizer = BertTokenizer.from_pretrained('./pretrained_models/law_bert/vocab.txt')
        self.max_1 = -1
        self.max_2 = -1
        self.max_3 = -1
        max_len_stat = -1

        for i in range(len(pros)):
            if len(pros[i]) <= 1:
                continue

            sentences_case = []
            for ptr1, pro in enumerate(pros[i]):
            # *******************************Generating the positive sentence*******************************
                sentences = []
                for con in cons[i]:
                    sentences.append(tokenizer.encode(pro[0:MAX_SEQ_LENGTH] + "[SEP]" + con[0:MAX_SEQ_LENGTH]))
                    max_len_stat = max(max_len_stat, len(sentences[-1]))

                # *******************************Generating the negative sentence*******************************
                sentences_negative = []
                for _ in range(NEGATIVE_SAMPLE_SIZE):
                    while 1:
                        target_case = cons[random.randint(0, len(cons) - 1)]
                        target_sentences = target_case[random.randint(0, len(target_case) - 1)]
                        if overlap(target_sentences, pro) >= OVERLAP_MIN:
                            sentences_negative.append(tokenizer.encode(pro[0:MAX_SEQ_LENGTH] + "[SEP]" + target_sentences[0:MAX_SEQ_LENGTH]))
                            break

                sentences_case.append([sentences, sentences_negative])

            # *******************************Generating the divergence pair*******************************
            for ptr1 in range(len(pros[i])):
                ptr2 = random.randint(0, len(pros[i]) - 2)
                if ptr2 >= ptr1:
                    ptr2 += 1

                self.sentences.append([sentences_case[ptr1][0], sentences_case[ptr2][0], sentences_case[ptr1][1]])
                self.max_1 = max(self.max_1, len(sentences_case[ptr1][0]))
                self.max_2 = max(self.max_2, len(sentences_case[ptr2][0]))
                self.max_3 = max(self.max_3, len(sentences_case[ptr1][1]))

        for i in range(len(cons)):
            if len(cons[i]) <= 1:
                continue

            sentences_case = []
            for ptr1, con in enumerate(cons[i]):
            # *******************************Generating the positive sentence*******************************
                sentences = []
                for pro in pros[i]:
                    sentences.append(tokenizer.encode(pro[0:MAX_SEQ_LENGTH] + "[SEP]" + con[0:MAX_SEQ_LENGTH]))

                # *******************************Generating the negative sentence*******************************
                sentences_negative = []
                for _ in range(NEGATIVE_SAMPLE_SIZE):
                    while 1:
                        target_case = pros[random.randint(0, len(pros) - 1)]
                        target_sentences = target_case[random.randint(0, len(target_case) - 1)]
                        if overlap(target_sentences, con) >= OVERLAP_MIN:
                            sentences_negative.append(tokenizer.encode(target_sentences[0:MAX_SEQ_LENGTH] + "[SEP]" + con[0:MAX_SEQ_LENGTH]))
                            break
                        
                sentences_case.append([sentences, sentences_negative])

            # *******************************Generating the divergence pair*******************************
            for ptr1 in range(len(cons[i])):
                ptr2 = random.randint(0, len(cons[i]) - 2)
                if ptr2 >= ptr1:
                    ptr2 += 1

                self.sentences.append([sentences_case[ptr1][0], sentences_case[ptr2][0], sentences_case[ptr1][1]])
                self.max_1 = max(self.max_1, len(sentences_case[ptr1][0]))
                self.max_2 = max(self.max_2, len(sentences_case[ptr2][0]))
                self.max_3 = max(self.max_3, len(sentences_case[ptr1][1]))

        # self.sentences = self.sentences[:3]
        # print(self.sentences[10][2])
        # c1是正例的list、c2是另外一个正例的list、c3是负例的list
        # print(self.max_1, self.max_2, self.max_3)

    def __getitem__(self, item):
        sentences, sentences_pair, sentences_neg = self.sentences[item]

        sentences_am = pad_lists_am(sentences, MAX_LENGTH, self.max_1)
        sentences_pair_am = pad_lists_am(sentences_pair, MAX_LENGTH, self.max_2)
        sentences_neg_am = pad_lists_am(sentences_neg, MAX_LENGTH, self.max_3)

        sentences_type = pad_lists_type(sentences, MAX_LENGTH, self.max_1)
        sentences_pair_type = pad_lists_type(sentences_pair, MAX_LENGTH, self.max_2)
        sentences_neg_type = pad_lists_type(sentences_neg, MAX_LENGTH, self.max_3)

        sentences = pad_lists(sentences, MAX_LENGTH, self.max_1)
        sentences_pair = pad_lists(sentences_pair, MAX_LENGTH, self.max_2)
        sentences_neg = pad_lists(sentences_neg, MAX_LENGTH, self.max_3)

        return sentences, sentences_am, sentences_type, sentences_pair, sentences_pair_am, sentences_pair_type, sentences_neg, sentences_neg_am, sentences_neg_type

    def __len__(self):
        return len(self.sentences)


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class LAEDataLoaderEval:
    """LAE Dataloader Eval"""
    _default_column_names = ["c1", "c2", "c3"]

    def __new__(cls, dataset_dir, column_names=None, stage="train", num_shards=1, shard_id=0):
        """
        LAE Dataloader API

        Args:
            dataset_dir: the directory to dataset
            column_names: the output column names, a tuple or a list of string with length 2
            stage: the supported `option` are in ["train"、"test"、"del"、"all"]

        Return:
            a GeneratorDataset for LAE dataset
        """
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if column_names is None:
            column_names = cls._default_column_names
            logger.info("The column_names to the LAEDataLoader is None, so assign it with default_column_names %s",
                        cls._default_column_names)

        if not isinstance(column_names[0], str) or not isinstance(column_names[1], str):
            raise ValueError(f"the item type of column_names should be string,"
                             f" but got {type(column_names[0])} and {type(column_names[1])}")

        dataset = LAEDataEvalSet(dataset_dir, stage)

        return GeneratorDataset(dataset, column_names, num_shards=num_shards, shard_id=shard_id)

class LAEDataEvalSet:
    """LAE DataSet Eval"""

    def __init__(self, dataset_dir, stage="train"):
        """
        LAEDataSet Dataset

        Args:
            dataset_dir: the directory to LAE dataset
            stage: the supported key word are in ["train"、"test"、"del"、"all"]

        Return:
            an iterable dataset for LAE dataset
        """
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")
        self.dataset_dir = dataset_dir
        self.stage = stage
        tokenizer = BertTokenizer.from_pretrained('./pretrained_models/law_bert/vocab.txt')

        sentences = []
        root = dataset_dir
        for root, dirs, files in os.walk(root):
            for _file in files:
                fp = open(os.path.join(root, _file), 'r')
                cases = json.loads(fp.read())
                for case in cases:
                    sentences_unit = []
                    for sentence in case[1]:
                        sentences_unit.append(tokenizer.encode(case[0][0:MAX_SEQ_LENGTH] + "[SEP]" + sentence[0:MAX_SEQ_LENGTH]))
                    
                    sentences.append(sentences_unit)
        self.sentences = sentences

    def __getitem__(self, item):
        sentences = self.sentences[item]

        sentences_am = pad_lists_am(sentences, MAX_LENGTH, 5)
        sentences_type = pad_lists_type(sentences, MAX_LENGTH, 5)
        sentences = pad_lists(sentences, MAX_LENGTH, 5)

        return sentences, sentences_am, sentences_type

    def __len__(self):
        return len(self.sentences)