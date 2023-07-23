import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding


from .arguments import DataArguments
from .trainer import DenseTrainer

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: DenseTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.q_distil = json.load(open(self.data_args.q_distil))
        self.p_offset = json.load(open(self.data_args.p_offset))
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)
        
        query_vector = self.q_distil[group['query_id']]
        teacher_query = []
        teacher_query = self.get_distil_vector(query_vector)

        encoded_passages, teacher_passages = [], []
        group_positives = group['positives']
        group_negatives = group['negatives']
        group_positives_id = group['positives_id']
        group_negatives_id = group['negatives_id']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
            pos_vector = self.p_distil[group_positives_id[0]]
        else:
            index = (_hashed_seed + epoch) % len(group_positives)
            pos_psg = group_positives[index]
            pos_vector = self.get_item(group_positives_id[index])

        encoded_passages.append(self.create_one_example(pos_psg))
        passage_vector = self.get_distil_vector(pos_vector)
        teacher_passages.append(passage_vector)

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            negs_id = [x for x in group_negatives_id]
            negs_comb = list(zip(negs, negs_id))
            random.Random(_hashed_seed).shuffle(negs_comb)
            negs, negs_id = zip(*negs_comb)
            negs = negs * 2
            negs_id = negs_id * 2
            negs = negs[_offset: _offset + negative_size]
            negs_id = negs_id[_offset: _offset + negative_size]

        for neg_psg, neg_id in zip(negs, negs_id):
            encoded_passages.append(self.create_one_example(neg_psg))
            neg_vector = self.get_item(neg_id)
            passage_vector = self.get_distil_vector(neg_vector)
            teacher_passages.append(passage_vector)

        return encoded_query, encoded_passages, teacher_query, teacher_passages

    def get_item(self, idx):
        offset = self.p_offset[idx]
        with open(self.data_args.p_distil) as f:
            f.seek(offset)
            line = json.loads(f.readline())
        return line

    def get_distil_vector(self, data):
        idx, vals = list(zip(*data))
        idx, vals = torch.tensor(idx), torch.tensor(vals, dtype=torch.float32)
        vector = torch.zeros(30522, dtype=torch.float32)
        vector[idx] = vals/100.0
        return vector
 
class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.encode_plus(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 256

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        qt = [f[2] for f in features]
        dt = [f[3] for f in features]
        
        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        if isinstance(qt[0], list):
            qt = sum(qt, [])
        if isinstance(dt[0], list):
            dt = sum(dt, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        q_teacher = {"input_tensors": torch.stack(qt)}
        d_teacher = {"input_tensors": torch.stack(dt)}
        return q_collated, d_collated, q_teacher, d_teacher


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features
