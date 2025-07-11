"""
rqvae
"""
import torch
import gin
from typing import List
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from enum import Enum
from einops import rearrange
from functools import cached_property
from generative_recommenders.modules.encoder import MLP
from generative_recommenders.modules.loss import CategoricalReconstuctionLoss
from generative_recommenders.modules.loss import ReconstructionLoss
from generative_recommenders.modules.loss import QuantizeLoss
from generative_recommenders.modules.normalize import l2norm
from generative_recommenders.modules.gumbel import gumbel_softmax_sample
from generative_recommenders.modules.kmeans import kmeans_init_
from generative_recommenders.modules.normalize import L2NormalizationLayer


torch.set_float32_matmul_precision('high')

@gin.constants_from_enum
class QuantizeForwardMode(Enum):
    """
    Quantize forward modes.
    """
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3
    SINKHORN = 4  


class QuantizeDistance(Enum):
    """
    Quantize distance modes.
    """
    L2 = 1
    COSINE = 2


class QuantizeOutput(NamedTuple):
    """
    Quantize output.
    """
    embeddings: Tensor
    ids: Tensor
    loss: Tensor


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, "b d -> b 1 d")
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e -
        2 * (e @ rearrange(w, "b d -> b d 1") @ rearrange(w, "b d -> b 1 d")) +
        2 * (e @ rearrange(u, "b d -> b d 1").detach() @ rearrange(q, "b d -> b 1 d").detach())
    ).squeeze()


@torch.no_grad()
def _sinkhorn_knopp(
    cost: Tensor,
    row_marginals: Tensor,
    col_marginals: Tensor,
    eps: float = 0.05,
    max_iter: int = 50
) -> Tensor:
    """
    cost: (B, K) cost matrix
    row_marginals: (B,) row marginals
    col_marginals: (K,) column marginals
    eps: entropy regularization temperature
    returns: (B, K) optimal transport matrix P, satisfies row and column constraints
    """
    K = torch.exp(-cost / eps)

    u = torch.ones_like(row_marginals)
    v = torch.ones_like(col_marginals)

    for _ in range(max_iter):
        u = row_marginals / (K @ v + 1e-8)
        v = col_marginals / (K.T @ u + 1e-8)

    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return P


class Quantize(nn.Module):
    """
    Quantize module.
    """
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        do_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        sim_vq: bool = False,  # https://arxiv.org/pdf/2411.02038
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        distance_mode: QuantizeDistance = QuantizeDistance.L2
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity()
        )

        self.quantize_loss = QuantizeLoss(commitment_weight)
        self._init_weights()

    @property
    def weight(self) -> Tensor:
        """
        Get embedding weight.
        """
        return self.embedding.weight

    @property
    def device(self) -> torch.device:
        """
        Get device.
        """
        return self.embedding.weight.device

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
    
    @torch.no_grad
    def _kmeans_init(self, x) -> None:
        kmeans_init_(self.embedding.weight, x=x)
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        """
        Get item embeddings by ids.
        """
        return self.out_proj(self.embedding(item_ids))

    def forward(self, x, temperature) -> QuantizeOutput:
        """
        Quantize forward pass.
        """
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x=x)

        codebook = self.out_proj(self.embedding.weight)
        if self.distance_mode == QuantizeDistance.L2:
            dist = (
                (x ** 2).sum(axis=1, keepdim=True) +
                (codebook.T ** 2).sum(axis=0, keepdim=True) -
                2 * x @ codebook.T
            )
        elif self.distance_mode == QuantizeDistance.COSINE:
            dist = -(
                x / x.norm(dim=1, keepdim=True) @
                (codebook.T / codebook.T.norm(dim=0, keepdim=True))
            )
        else:
            raise Exception("Unsupported Quantize distance mode.")
        _, ids = (dist.detach()).min(axis=1)

        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                weights = gumbel_softmax_sample(
                    -dist, temperature=temperature, device=self.device
                )
                emb = weights @ codebook
                emb_out = emb
            elif self.forward_mode == QuantizeForwardMode.STE:
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
            elif self.forward_mode == QuantizeForwardMode.ROTATION_TRICK:
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (x.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    x
                )
            elif self.forward_mode == QuantizeForwardMode.SINKHORN:
                # https://arxiv.org/pdf/2311.09049
                B, K = dist.shape
                max_d, min_d = dist.max(), dist.min()
                mid = (max_d + min_d) / 2
                amp = max_d - mid + 1e-5
                dist_norm = ((dist - mid) / amp).double()

                row_marginals = torch.full((B, ), 1. / B, device=self.device)
                col_marginals = torch.full((K, ), 1. / K, device=self.device)

                P = _sinkhorn_knopp(
                    cost=dist_norm,
                    row_marginals=row_marginals,
                    col_marginals=col_marginals,
                    eps=0.003,
                    max_iter=100
                ).detach()

                # hard ids & embeddings
                sk_ids = P.argmax(dim=-1)
                emb = self.get_item_embeddings(sk_ids)
                emb_out = x + (emb - x).detach()
                ids = sk_ids
            else:
                raise Exception("Unsupported Quantize forward mode.")
            loss = self.quantize_loss(query=x, value=emb)
        
        else:
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(query=x, value=emb_out)

        return QuantizeOutput(
            embeddings=emb_out,
            ids=ids,
            loss=loss
        )


class RqVaeOutput(NamedTuple):
    """
    RqVae output.
    """
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    """
    RqVae computed losses.
    """
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor




class RqVae(nn.Module):
    """
    RqVae model.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        codebook_last_layer_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 18,
    ) -> None:
        self._config = locals()
        
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features

        self.layers = []
        for i in range(n_layers):
            mode = codebook_mode if i < n_layers - 1 else codebook_last_layer_mode
            self.layers.append(
                Quantize(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    forward_mode=mode,
                    do_kmeans_init=codebook_kmeans_init,
                    codebook_normalize=i == 0 and codebook_normalize,
                    sim_vq=codebook_sim_vq,
                    commitment_weight=commitment_weight,
                    distance_mode=QuantizeDistance.L2
                )
            )
            
        self.layers = nn.ModuleList(modules=self.layers)

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=True
        )

        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features) if n_cat_features != 0
            else ReconstructionLoss()
        )
    
    @cached_property
    def config(self) -> dict:
        """
        Get the config of the model.
        """
        return self._config
    
    @property
    def device(self) -> torch.device:
        """
        Get the device of the model.
        """
        return next(self.encoder.parameters()).device
    
    def load_pretrained(self, path: str) -> None:
        """
        Load a pretrained RQVAE model.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model"])
        print(f"---Loaded RQVAE Iter {state['iter']}---")

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode a batch of data.
        """
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        """
        Decode a batch of data.
        """
        return self.decoder(x)

    def get_semantic_ids(
        self,
        x: Tensor,
        gumbel_t: float = 0.001
    ) -> RqVaeOutput:
        """
        Get the semantic ids of a batch of data.
        """
        res = self.encode(x)

        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []
        for layer in self.layers:
            residuals.append(res)
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss += quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb
            sem_ids.append(id)
            embs.append(emb)

        return RqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),
            residuals=rearrange(residuals, "b h d -> h d b"),
            sem_ids=rearrange(sem_ids, "b d -> d b"),
            quantize_loss=quantize_loss
        )

    @torch.compile(mode="reduce-overhead")
    def forward(self, batch: Tensor, gumbel_t: float) -> RqVaeComputedLosses:
        """
        Forward pass.

        Args:
            batch (Tensor): input batch
            gumbel_t (float): gumbel temperature
        Returns:
            RqVaeComputedLosses: computed losses
        """
        x = batch
        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals
        x_hat = self.decode(embs.sum(axis=-1))
        x_hat = torch.cat([l2norm(x_hat[..., :-self.n_cat_feats]), x_hat[..., -self.n_cat_feats:]], axis=-1)
        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = quantized.quantize_loss
        loss = (reconstuction_loss + rqvae_loss).mean()

        with torch.no_grad():
            # Compute debug ID statistics
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (~torch.triu(
                (rearrange(quantized.sem_ids, "b d -> b 1 d") \
                == rearrange(quantized.sem_ids, "b d -> 1 b d")).all(axis=-1), diagonal=1)
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids
        )


