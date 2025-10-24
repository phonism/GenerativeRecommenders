"""
NoteLLM: A Retrievable Large Language Model for Note Recommendation
https://arxiv.org/pdf/2403.01744
"""
import torch
from torch import nn
import gin
from transformers import (
    EvalPrediction,
    Qwen2ForCausalLM,
    Qwen2TokenizerFast,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from transformers import cache_utils
from transformers.loss import loss_utils
from typing import Union, Optional, Dict, List, Callable, Tuple
import pdb


class DynamicCache(cache_utils.DynamicCache):
    """
    DynamicCache with device caching
    """

    def __init__(self, obj: cache_utils.DynamicCache):
        super().__init__()
        self.key_cache, self.value_cache = obj.key_cache, obj.value_cache
        self.fixed_seq_length = obj.get_seq_length()
        self.device_dict = {self.key_cache[0].device: self}

    def to(self, device: torch.device) -> "DynamicCache":
        """
        Get the DynamicCache object for the specified device.
        """
        obj = self.device_dict.get(device)
        if obj is None:
            obj = self.device_dict[device] = cache_utils.DynamicCache()
            obj.key_cache = [k.to(device) for k in self.key_cache]
            obj.value_cache = [v.to(device) for v in self.value_cache]
        obj.crop(self.fixed_seq_length)
        return obj


@gin.configurable 
class Query2Embedding(nn.Module):
    """Model"""

    def __init__(
        self,
        pretrained_path: str,
        freeze_lm: bool = False,
        item_token: str = "[EMB]",
        pad_token: str = "<|pad|>",
        eos_token: str = "<|endoftext|>",
        gradient_checkpointing: bool = True,
        tau: float = 3.0,
        alpha: float = 0.01,
        hardneg_r: float = 0.1
    ):
        super().__init__()

        self.tau = nn.Parameter(torch.tensor(tau))
        self.alpha, self.hardneg_r = alpha, hardneg_r
        self.item_token = item_token
        self.tokenizer: PreTrainedTokenizerBase = Qwen2TokenizerFast.from_pretrained(
            pretrained_path, pad_token=pad_token, eos_token=eos_token
        )
        self.model: PreTrainedModel = Qwen2ForCausalLM.from_pretrained(pretrained_path)
        if freeze_lm:
            for param in self.model.parameters():
                param.requires_grad = False
        self.tokenizer.add_tokens([item_token], special_tokens=True)
        self.emb_id = self.tokenizer.convert_tokens_to_ids(item_token)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.vocab_size = len(self.tokenizer)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def tokenize(self,
        query: list[str],
        category: list[str] = None,
        score: list[float] = [],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts.

        Args:
            query (list[str]): query texts.
            category (list[str]): category texts.
            score (list[float]): score.
        Returns:
            Dict[str, torch.Tensor]: tokenized texts.
        """
        token = self.tokenizer(
            query,
            category,
            padding=True,
            return_token_type_ids=category is not None,
            return_tensors="pt",
        )

        # emb special token position
        token["emb_token_idx"] = (token.input_ids == self.emb_id).int().argmax(1, keepdim=True)
        
        if category:
            token["labels"] = token.input_ids.where(token.pop("token_type_ids").bool(), -100)

        # build hardneg label
        if score:
            token["hardneg"] = torch.tensor(score) < self.hardneg_r
        return token

    def get_embedding(self, input_ids, attention_mask, emb_token_idx, past_key_values):
        """
        get embedding
        """
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        last_hidden_state, outputs["last_hidden_state"] = outputs["last_hidden_state"], None

        # extract embedding
        emb_token_idx = emb_token_idx.repeat(1, self.model.config.hidden_size).unsqueeze(1)
        outputs["sentence_embedding"] = last_hidden_state.gather(1, emb_token_idx).squeeze(1)
        
        norm_embedding = nn.functional.normalize(outputs.sentence_embedding, p=2, dim=1)
        return norm_embedding, last_hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        emb_token_idx: torch.LongTensor,
        labels: torch.LongTensor = None,
        hardneg: torch.BoolTensor = None,
        past_key_values: DynamicCache = None,
        return_loss: bool = True,
    ):
        """
        forward
        """
        if past_key_values is not None:
            past_key_values = past_key_values.to(input_ids.device)
            if past_key_values.key_cache[0].shape[0] != input_ids.shape[0]:
                past_key_values.batch_select_indices([0] * input_ids.shape[0])

            attention_mask = torch.cat([
                torch.ones(
                    attention_mask.shape[0],
                    past_key_values.get_seq_length(),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ), attention_mask
            ], dim=1)


        outputs = {}
        outputs["sentence_embedding"], last_hidden_state = self.get_embedding(
            input_ids,
            attention_mask,
            emb_token_idx,
            past_key_values
        )

        if not return_loss:
            return {"sentence_embedding": outputs["sentence_embedding"]}

        # vector similarity loss
        sim = torch.mm(
            nn.functional.normalize(outputs["sentence_embedding"][::2], p=2, dim=1),
            nn.functional.normalize(outputs["sentence_embedding"][1::2], p=2, dim=1).T,
        )
        log_softmax = -(sim * self.tau.exp()).softmax(dim=1).diag().log()
        
        if hardneg is not None:
            outputs["hardneg"] = hardneg
            if hardneg.any():
                # create a True tensor with the same shape as hardneg
                mask = torch.ones_like(hardneg, dtype=torch.bool).to(log_softmax.device)
                # set hardneg positions to False
                mask[hardneg] = False
                # use mask instead of ~hardneg
                log_softmax = torch.cat([
                    log_softmax[mask],
                    (sim[hardneg].mean(dim=1) + 1).log() * self.hardneg_r
                ])
        cl_loss = log_softmax.mean()

        if labels is None or (labels < 0).all():
            outputs["loss"] = cl_loss
            return outputs

        # category generation loss
        logits = self.model.lm_head(last_hidden_state)
        gen_loss = loss_utils.ForCausalLMLoss(
            logits=logits,
            labels=labels,
            vocab_size=self.model.config.vocab_size
        )
        outputs["loss"] = (cl_loss + gen_loss * self.alpha) / (1 + self.alpha)
        return outputs

    def save_pretrained(self, save_dir: str, **kwargs):
        """
        Save both tokenizer and model weights.

        Args:
            save_dir (str): directory to save the model and tokenizer.
            **kwargs: additional arguments to pass to the tokenizer and model.
        """
        self.model.save_pretrained(save_dir, **kwargs)
        self.tokenizer.save_pretrained(save_dir)
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Support two input methods:
        1. Tensor: directly as input_ids
        2. Dict: containing all parameters needed for model generation (e.g. input_ids, attention_mask)
        Additional parameters are passed to model.generate() through kwargs
        """
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        elif not isinstance(inputs, dict):
            raise ValueError("inputs must be Tensor or Dict[str, Tensor]")
        
        # call model generate
        return self.model.generate(**inputs, **kwargs)

    @staticmethod
    def compute_metrics(topk: int = 5, batch_size: int = 64, save_file: str = ''):
        """compute_metrics"""

        def compute_topk_acc(eval_prediction: EvalPrediction):
            """compute top k accuracy"""
            if isinstance(eval_prediction.predictions, tuple):
                pred, hardneg = eval_prediction.predictions
                pred = torch.tensor(pred)
                pred1, pred2 = pred[::2][~hardneg], pred[1::2][~hardneg]
            else:
                pred = torch.tensor(eval_prediction.predictions)
                pred1, pred2 = pred[::2], pred[1::2]

            correct_topk = 0
            for i in range(0, pred1.shape[0] // batch_size * batch_size, batch_size):
                sim = torch.mm(
                    nn.functional.normalize(pred1[i:i + batch_size], p=2, dim=1),
                    nn.functional.normalize(pred2[i:i + batch_size], p=2, dim=1).T,
                )
                _, topk_idx = sim.topk(topk, dim=0)
                true_idx = torch.arange(sim.shape[0], device=topk_idx.device)
                correct_topk += (topk_idx == true_idx).sum().item()
            topk_acc = correct_topk / pred1.shape[0]

            if save_file:
                with open(save_file, "a") as f:
                    print(topk_acc, file=f)
            return {"topk_acc": topk_acc}
        return compute_topk_acc
