"""
gumbel
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Tuple


def sample_gumbel(
    shape: Tuple,
    device: torch.device,
    eps: float = 1e-20,
) -> Tensor:
    """
    Sample from Gumbel(0, 1)

    Args:
        shape: Tuple
        device: torch.device
        eps: float
    Returns:
        Tensor
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(
    logits: Tensor,
    temperature: float,
    device: torch.device,
) -> Tensor:
    """
    Draw a sample from the Gumbel-Softmax distribution

    Args:
        logits: Tensor
        temperature: float
        device: torch.device
    Returns:
        Tensor
    """
    y = logits + sample_gumbel(logits.shape, device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample