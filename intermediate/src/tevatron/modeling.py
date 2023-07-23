import json
import os
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, AutoModelForMaskedLM, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


from typing import Optional, Dict, List

from arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
import logging

logger = logging.getLogger(__name__)


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True
    ):
        super(LinearPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q[:, 0])
        elif p is not None:
            return self.linear_p(p[:, 0])
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, 'pooler.pt')
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'pooler.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class SpladePooler(nn.Module):
    def __init__(
            self,
            top_k: int
    ):
        super(SpladePooler, self).__init__()
        self.top_k = top_k
        self._config = {'top_k': top_k}

    def forward(self, embeddings: Tensor, attention_mask: Tensor, top_k: int):
        x = torch.max(torch.log(1 + torch.relu(embeddings)) * attention_mask.unsqueeze(-1), dim=1).values
        kvals, kidx = x.topk(k=top_k, dim=1)
        topk = torch.zeros_like(x)
        topk[torch.arange(x.size(0))[:, None], kidx] = kvals
        return topk

class DenseModel(nn.Module):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()
        self._trainer = None
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = SpladePooler(model_args.top_k)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
   
        self.top_k = int(lm_q.cls.predictions.decoder.out_features * 0.01)

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
            query_mlm: Dict[str, Tensor] = None,
            passage_mlm: Dict[str, Tensor] = None
    ):
        q_mlm_loss = self.lm_q(**query_mlm).loss
        p_mlm_loss = self.lm_p(**passage_mlm).loss
        
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)
        
        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                p_reps=p_reps
            ) 
        
        if self.training:
            if self.train_args.negatives_x_device:
                q_reps = self.dist_gather_tensor(q_reps)
                p_reps = self.dist_gather_tensor(p_reps)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device \
                else self.train_args.per_device_train_batch_size

            scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
            scores = scores.view(effective_bsz, -1)
   
            target = torch.arange(
                scores.size(0),
                device=scores.device,
                dtype=torch.long
            )
            
            target = target * self.data_args.train_n_passages
            loss = self.cross_entropy(scores, target)
        
            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            
            loss = loss + q_mlm_loss + p_mlm_loss
           
           return DenseOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps
            )

        else:
            loss = None
            if query and passage:
                scores = (q_reps * p_reps).sum(1)
            else:
                scores = None

            return DenseOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps
            )

    def encode_passage(self, psg):
        if psg is None:
            return None, None

        psg_out = self.lm_p(**psg, return_dict=False)
        p_hidden = psg_out["logits"]
        p_reps = self.pooler(p_hidden, psg["attention_mask"], self.top_k)
        return p_reps


    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=False)
        q_hidden = qry_out["logits"]
        q_reps = self.pooler(q_hidden, qry["attention_mask"], self.top_k)
        return q_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = SpladePooler(
            model_args.top_k
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            if model_args.freeze_layers:
                for param in lm_q.bert.parameters():
                    param.requires_grad = False
                lm_q.cls.predictions.decoder.weight.requires_grad = True
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.decoder_weights:
            lm_q.cls.predictions.decoder = torch.load(model_args.decoder_weights)
            lm_p.cls.predictions.decoder = torch.load(model_args.decoder_weights)

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args
        )
        return model

    def save(self, output_dir: str):
        if self.model_args.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)
            #torch.save(self.lm_q.cls.predictions.decoder, os.path.join(output_dir, 'decoder.pt'))

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

