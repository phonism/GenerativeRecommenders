"""
Trainer for the retrieval decoder model.
"""
from collections import defaultdict
from einops import rearrange
import torch
from typing import Dict, List

class TopKAccumulator:
    """
    TopKAccumulator
    """
    def __init__(self, ks: list[int] = [1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self) -> None:
        """
        Reset the accumulator
        """
        self.total = 0
        self.metrics = defaultdict(int)

    def accumulate(self, actual: torch.Tensor, top_k: torch.Tensor) -> None:
        """
        Accumulate the metrics
        """
        B, D = actual.shape
        pos_match = (rearrange(actual, "b d -> b 1 d") == top_k)
        for i in range(D):
            match_found, rank = pos_match[..., :i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(matched_rank[matched_rank < k])
            
            match_found, rank = pos_match[..., i:i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])
        self.total += B
        
    def reduce(self) -> Dict[str, float]:
        """
        Reduce the metrics
        """
        return {k: v / self.total for k, v in self.metrics.items()}