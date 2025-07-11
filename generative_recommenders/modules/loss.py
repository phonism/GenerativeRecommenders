"""
loss functions
"""
from torch import nn
from torch import Tensor


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for RQ-VAE
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            x_hat (Tensor): reconstructed data
            x (Tensor): original data
        Returns:
            Tensor: reconstruction loss
        """
        return ((x_hat - x) ** 2).sum(axis=-1)


class CategoricalReconstuctionLoss(nn.Module):
    """
    Reconstruction loss for RQ-VAE with categorical features
    """
    def __init__(self, n_cat_feats: int) -> None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats
    
    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            x_hat (Tensor): reconstructed data
            x (Tensor): original data
        Returns:
            Tensor: reconstruction loss
        """
        reconstr = self.reconstruction_loss(
            x_hat[:, :-self.n_cat_feats],
            x[:, :-self.n_cat_feats]
        )
        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:, -self.n_cat_feats:],
                x[:, -self.n_cat_feats:],
                reduction='none'
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr


class QuantizeLoss(nn.Module):
    """
    Quantization loss for RQ-VAE
    """
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        """
        Forward pass
        
        Args:
            query (Tensor): quantized embeddings
            value (Tensor): embedding values
        Returns:
            Tensor: quantization loss
        """
        emb_loss = ((query.detach() - value) ** 2).sum(axis=[-1])
        query_loss = ((query - value.detach()) ** 2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss