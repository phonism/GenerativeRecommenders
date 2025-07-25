"""
utils
"""
import argparse
import gin
import torch
from generative_recommenders.data.schemas import TokenizedSeqBatch
from einops import rearrange
from torch import Tensor


def reset_kv_cache(fn):
    """
    Reset the kv cache before and after the function call.
    """
    def inner(self, *args, **kwargs):
        """
        Set the model to evaluation mode and restore the original mode after the function call.
        """
        self.decoder.reset_kv_cache()
        out = fn(self, *args, **kwargs)
        self.decoder.reset_kv_cache()
        return out
    
    return inner


def reset_encoder_cache(fn):
    """
    Reset the encoder cache before and after the function call.
    """
    def inner(self, *args, **kwargs):
        """
        Set the model to evaluation mode and restore the original mode after the function call.
        """
        if self.jagged_mode:
            self.transformer.cached_enc_output = None
        out = fn(self, *args, **kwargs)
        if self.jagged_mode:
            self.transformer.cached_enc_output = None
        return out
    
    return inner


def eval_mode(fn):
    """
    Set the model to evaluation mode and restore the original mode after the function call.
    """
    def inner(self, *args, **kwargs):
        """
        Set the model to evaluation mode and restore the original mode after the function call.
        """
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def select_columns_per_row(x: Tensor, indices: Tensor) -> torch.Tensor:
    """
    Select columns from a tensor for each row.
    """
    assert x.shape[0] == indices.shape[0]
    assert indices.shape[1] <= x.shape[1]

    B = x.shape[0]
    return x[
        rearrange(torch.arange(B, device=x.device), "B -> B 1"), indices
    ]


def maybe_repeat_interleave(x, repeats, dim):
    """
    Repeat the tensor along the given dimension, but only for the first row.
    """
    if not isinstance(x, Tensor):
        return x
    return x.repeat_interleave(repeats, dim=dim)


def parse_config():
    """
    Parse the gin config file and set the parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to gin config file.")
    args = parser.parse_args()
    gin.parse_config_file(args.config_path)


@torch.no_grad
def compute_debug_metrics(batch: TokenizedSeqBatch, model_output=None, prefix: str="") -> dict:
    """
    Compute debug metrics for a batch of sequences.
    """
    seq_lengths = batch.seq_mask.sum(axis=1).to(torch.float32)
    prefix = prefix + "_"
    debug_metrics = {
        prefix + f"seq_length_p{q}": torch.quantile(seq_lengths, q=q).detach().cpu().item() 
        for q in [0.25, 0.5, 0.75, 0.9, 1]
    }
    if model_output is not None:
        loss_debug_metrics = {
            prefix + f"loss_{d}": model_output.loss_d[d].detach().cpu().item() \
            for d in range(batch.sem_ids_fut.shape[1])
        }
        debug_metrics.update(loss_debug_metrics)
    return debug_metrics