"""
Customisable Transformer encoder-decoder implementation.
"""
from __future__ import annotations
from typing import Optional, Callable, List, Tuple
import torch
import math
from torch import nn, Tensor

from generative_recommenders.modules.normalize import RMSNorm, RootMeanSquareLayerNorm

import math
import torch
from torch import nn, Tensor
from typing import Optional, Tuple

def _relative_position_bucket(
    relative_positions: Tensor,
    num_buckets: int = 32,
    max_distance: int = 128,
    bidirectional: bool = True,
) -> Tensor:
    """
    relative_position_bucket
    """
    ret = -relative_positions
    if bidirectional:
        num_buckets //= 2
        sign = (ret < 0).long()
        ret = ret.abs()
    else:
        ret = torch.clamp_min(ret, 0)

    max_exact = num_buckets // 2
    is_small = ret < max_exact
    large_val = max_exact + (
        (torch.log(ret.float() / max_exact + 1e-6) /
         math.log(max_distance / max_exact)) *
        (num_buckets - max_exact)
    ).long().clamp(max=num_buckets - max_exact - 1)

    ret = torch.where(is_small, ret, large_val)
    if bidirectional:
        ret = ret + sign * num_buckets
    return ret


class T5Attention(nn.Module):
    """
    T5 Attention
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
        has_relative_bias: bool = True,
        num_relative_buckets: int = 32,
        max_distance: int = 128,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.is_cross_attention = is_cross_attention
        self.has_relative_bias = has_relative_bias

        self.q = nn.Linear(d_model, d_model, bias=False)
        if is_cross_attention:
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
        else:
            self.kv = nn.Linear(d_model, 2 * d_model, bias=False)

        self.o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if has_relative_bias and not is_cross_attention:
            self.rel_bias = nn.Embedding(n_heads * num_relative_buckets, 1)
            self.num_relative_buckets = num_relative_buckets
            self.max_distance = max_distance
        else:
            self.rel_bias = None

    def _get_rel_bias(self, q_len: int, k_len: int, device: torch.device) -> Tensor:
        """
        _get_rel_bias
        """
        ctx = torch.arange(q_len, device=device)[:, None]
        mem = torch.arange(k_len, device=device)[None, :]
        rp = mem - ctx
        buckets = _relative_position_bucket(
            rp,
            self.num_relative_buckets,
            self.max_distance,
            bidirectional=True
        )
        buckets = buckets.unsqueeze(0).expand(self.n_heads, -1, -1)
        head_offset = (torch.arange(self.n_heads, device=device)
            * self.num_relative_buckets)[:, None, None]
        idx = buckets + head_offset

        bias = self.rel_bias(idx.flatten())
        bias = bias.view(self.n_heads, q_len, k_len)
        return bias.unsqueeze(0)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        forward
        """
        if self.is_cross_attention:
            k = self.k(key)
            v = self.v(value)
        else:
            key_value = query
            k, v = self.kv(key_value).chunk(2, dim=-1)

        q = self.q(query)

        # reshape -> (b,h,l,d)
        def split_heads(x):
            b, l, _ = x.size()
            return x.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))

        # (b, h, q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.rel_bias is not None:
            if position_bias is None:
                position_bias = self._get_rel_bias(q.size(-2), k.size(-2), q.device)
            scores = scores + position_bias
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], -1e9)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores + attn_mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(query.size(0), -1, self.d_model)
        out = self.o(out)
        return out, position_bias


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with configurable hidden dims and activation.
    """
    def __init__(
        self,
        dim: int,
        hidden_dims: int | List[int] = 2048,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the feed-forward network.
        """
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        layers: list[nn.Module] = []
        in_dim = dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h, bias=False), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  
        """
        Forward pass through the feed-forward network.
        """
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    Wrapper around `nn.MultiheadAttention` operating in batch-first mode.
    Provides both self-attention (query == key == value) and cross-attention.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        kv_dim: Optional[int] = None,
        bias: bool = True,
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kv_dim = kv_dim or dim
        """
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            kdim=self.kv_dim,
            vdim=self.kv_dim,
        )
        """
        self.attn = T5Attention(
            d_model=dim,
            n_heads=num_heads,
            dropout=dropout,
            is_cross_attention=is_cross_attention,
            has_relative_bias=True,
            num_relative_buckets=32,
            max_distance=128,
        )

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the multi-head attention.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        out, _ = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return out


class TransformerBlock(nn.Module):
    """
    A single Transformer layer that can operate as encoder (self-attn only) or
    decoder (self-attn + cross-attn).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        norm_cls: type[nn.Module] = RMSNorm,
        ff_hidden_dim: int = 2048,
        cross_attn: bool = False,
    ) -> None:
        """
        Initialize the transformer block.
        """
        super().__init__()
        self.cross_attn_enabled = cross_attn

        self.self_attn = MultiHeadAttention(dim, num_heads, dropout, bias=False)
        self.norm1 = norm_cls(dim)
        self.dropout1 = nn.Dropout(dropout)

        if cross_attn:
            self.cross_attn = MultiHeadAttention(
                dim, num_heads, is_cross_attention=True, dropout=dropout, bias=False)
            self.norm_cross = norm_cls(dim)
            self.dropout_cross = nn.Dropout(dropout)

        self.ff = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm2 = norm_cls(dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the transformer block.
        """
        # Self-attention
        attn = x + self.dropout1(
            self.self_attn(
                query=self.norm1(x),
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        )

        # Cross-attention (only if decoder)
        if self.cross_attn_enabled and context is not None:
            attn = attn + self.dropout_cross(
                self.cross_attn(
                    query=self.norm_cross(attn),
                    key=context,
                    value=context,
                    key_padding_mask=memory_key_padding_mask,
                )
            )
        # FFN
        attn = attn + self.dropout2(self.ff(self.norm2(attn)))
        return attn


class TransformerEncoder(nn.Module):
    """
    Transformer encoder.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.1,
        norm_cls: type[nn.Module] = RMSNorm,
        ff_hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim,
                num_heads,
                dropout,
                norm_cls=norm_cls,
                ff_hidden_dim=ff_hidden_dim,
                cross_attn=False,
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        src: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the transformer encoder.
        """
        for layer in self.layers:
            src = layer(src, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return src


class TransformerDecoder(nn.Module):
    """
    Transformer decoder.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.1,
        norm_cls: type[nn.Module] = RMSNorm,
        ff_hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim,
                num_heads,
                dropout,
                norm_cls=norm_cls,
                ff_hidden_dim=ff_hidden_dim,
                cross_attn=True,
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        tgt: Tensor,
        *,
        memory: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the transformer decoder.
        """
        for layer in self.layers:
            tgt = layer(
                tgt,
                context=memory,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return tgt


class TransformerEncoderDecoder(nn.Module):
    """
    High-level encoder-decoder wrapper (similar to `torch.nn.Transformer`)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_cls: type[nn.Module] = RMSNorm,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            dim=d_model,
            depth=num_encoder_layers,
            num_heads=nhead,
            dropout=dropout,
            norm_cls=norm_cls,
            ff_hidden_dim=dim_feedforward,
        )
        self.decoder = TransformerDecoder(
            dim=d_model,
            depth=num_decoder_layers,
            num_heads=nhead,
            dropout=dropout,
            norm_cls=norm_cls,
            ff_hidden_dim=dim_feedforward,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        *,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the transformer encoder-decoder.
        """
        if tgt_mask is None:
            T = tgt.size(1)
            tgt_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=tgt.device),
                diagonal=1
            )
        memory = self.encoder(src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        out = self.decoder(
            tgt,
            memory=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out