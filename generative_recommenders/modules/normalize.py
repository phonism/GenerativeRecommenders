"""
normalize
"""
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import List


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """
    L2 normalization.
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)


class L2NormalizationLayer(nn.Module):
    """
    L2 normalization layer.
    """
    def __init__(
        self,
        dim: int = -1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        """
        return l2norm(x, dim=self.dim, eps=self.eps)


class RMSNorm(nn.Module):
    """
    RMS normalization layer.
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        """
        output = self._norm(x.float()).to(x.dtype)
        return output * self.weight


class SwishLayerNorm(nn.Module):
    """
    Swish layer normalization.
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        """
        return F.silu(self.ln(x))


class RootMeanSquareLayerNorm(nn.Module):
    """
    Root mean square layer normalization.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        make sure float32 for variance calculation
        """
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states