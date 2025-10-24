"""
LCRec: Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation
https://arxiv.org/pdf/2311.09049
"""
import torch
from torch import nn
import json
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from typing import Union, Optional, Dict, List, Callable, Tuple

class LCRec(nn.Module):
    """
    LC-Rec implementation based on Qwen2 LLM backbone.

    This module integrates collaborative semantics by representing each
    item with a unique *special token* appended to the tokenizer's
    vocabulary. The model can be fine-tuned to predict the next-item
    token given a user interaction history (sequence of special tokens),
    enabling direct item generation without candidate sampling.

    References:
        Zheng *et al.* "Adapting Large Language Models by Integrating
        Collaborative Semantics for Recommendation" (ICDE 2024).
    """

    def __init__(
        self,
        pretrained_path: str,
    ) -> None:
        super().__init__()

        # Load tokenizer and LLM backbone
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained_path)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained_path)
    
    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing.
        """
        self.model.gradient_checkpointing_enable()
    
    def add_codebook_tokens(self, num_codebooks: int, codebook_size: int):
        """
        Add codebook tokens to the tokenizer.
        """
        num_added = 0
        for i in range(num_codebooks):
            for j in range(codebook_size):
                num_added += self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": [f"<C{i}_{j}>"]}
                )
        if num_added > 0: # means we added new tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.vocab_size = len(self.tokenizer)
        
    def tokenize(self, prompt: str, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts.

        Args:
            prompt (str): prompt to tokenize.
            *args: additional arguments to pass to the tokenizer.
            **kwargs: additional keyword arguments to pass to the tokenizer.
        Returns:
            Dict[str, torch.Tensor]: tokenized texts.
        """
        return self.tokenizer(prompt, *args, **kwargs)
    
    def decode(self, ids: torch.Tensor, *args, **kwargs) -> str:
        """
        Decode token IDs to text.

        Args:
            ids (torch.Tensor): token IDs to decode.
            *args: additional arguments to pass to the tokenizer.
            **kwargs: additional keyword arguments to pass to the tokenizer.
        Returns:
            str: decoded text.
        """
        return self.tokenizer.decode(ids, *args, **kwargs)
        
    def tokenize_sft_format(
        self,
        prompt: str,
        response: str = "",
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts.

        Args:
            prompt (str): prompt to tokenize.
            response (str): response to tokenize.
            device (torch.device, optional): device to move the tokens to. Defaults to torch.device("cpu").
        Returns:
            Dict[str, torch.Tensor]: tokenized texts.
        """
        prompt_ids = self.tokenizer(prompt).input_ids
        response_ids = self.tokenizer(response).input_ids
        input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
        input_ids = torch.LongTensor([input_ids]).to(device)
        return {
            "input_ids": input_ids,
            "prompt_seq_length": len(prompt_ids),
            "attention_mask": torch.ones_like(input_ids)
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward function.

        Args:
            input_ids (torch.Tensor): input IDs.
            attention_mask (torch.Tensor): attention mask.
            labels (torch.Tensor): labels.
            **generate_kwargs: additional arguments to pass to the model.
        Returns:
            Dict[str, torch.Tensor]: outputs.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
    def generate_topk(
        self,
        input_ids: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 3,
        beam_width: int = 10,
        topk: Optional[int] = None,
        allowed_token_fn: Optional[Callable[[int], bool]] = None,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[List[Tuple[torch.Tensor, float]]]:
        """
        batched beam search
        Args:
            input_ids (torch.Tensor): input IDs.
            attention_mask (torch.Tensor): attention mask.
            max_new_tokens (int): max new tokens.
            beam_width (int): beam width.
            topk (Optional[int]): topk.
            allowed_token_fn (Optional[Callable[[int], bool]]): allowed token function.
            eos_token_id (Optional[int]): end of sentence token ID.
            temperature (float): temperature.
        Returns:
            List[List[Tuple[torch.Tensor, float]]]: list of list of tuple of (ids, log_prob).
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        seq_lens = input_ids.size(1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        topk = topk or beam_width
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id

        # initialize beams
        beams = [[(input_ids[i], 0.0, False)] for i in range(batch_size)]  # [batch][beam]

        for _ in range(max_new_tokens):
            new_beams = []
            for b in range(batch_size):
                candidates = []
                all_finished = True
                for seq, score, finished in beams[b]:
                    if finished:
                        candidates.append((seq, score, True))
                        continue

                    all_finished = False
                    attn = attention_mask[b].unsqueeze(0)
                    out = self.model(input_ids=seq.unsqueeze(0), attention_mask=attn)
                    logits = out.logits[0, -1] / temperature
                    log_probs = F.log_softmax(logits, dim=-1)

                    next_scores, next_tokens = torch.topk(log_probs, beam_width)
                    for tok, tok_logp in zip(next_tokens.tolist(), next_scores.tolist()):
                        if allowed_token_fn and not allowed_token_fn(tok):
                            continue
                        new_seq = torch.cat([seq, torch.tensor([tok], device=device)])
                        new_score = score + tok_logp
                        new_finished = (tok == eos_token_id)
                        candidates.append((new_seq, new_score, new_finished))
                if not candidates:
                    # if all candidates are filtered out, keep the original beam
                    candidates = beams[b]
                # top beam_width
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beams.append(candidates[:beam_width])

            beams = new_beams

            # if all beams are finished, break
            if all(all(finished for (_, _, finished) in beam) for beam in beams):
                break

        # final result, each batch's beam sorted by score
        final_result = []
        for beam in beams:
            beam.sort(key=lambda x: x[1], reverse=True)
            final_result.append([(seq, score) for seq, score, _ in beam[:topk]])
        return final_result
