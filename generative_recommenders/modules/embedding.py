"""
embeddings
"""
from torch import nn
import torch

class SemIdEmbedding(nn.Module):
    """
    Semantic ID embedding
    """
    def __init__(
        self,
        num_embeddings: int,
        sem_ids_dim: int,
        embeddings_dim: int,
    ) -> None:
        super().__init__()
        
        self.sem_ids_dim = sem_ids_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = sem_ids_dim * num_embeddings
        
        self.emb = nn.Embedding(
            num_embeddings=num_embeddings * self.sem_ids_dim + 1,
            embedding_dim=embeddings_dim,
            padding_idx=self.padding_idx
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        forward pass
        Args:
            input_ids: (B, T) (ids_0_0, ids_0_1, ..., ids_0_C, ids_1_0, ids_1_1, ..., ids_1_C, ...)
            token_type_ids: (B, T) (0, 1, ..., C - 1, 0, 1, ..., C - 1, ...)
        Returns:
            (B, T, D)
        """
        sem_ids = token_type_ids * self.num_embeddings + input_ids
        return self.emb(sem_ids)


class UserIdEmbedding(nn.Module):
    """
    User ID embedding
    """
    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embeddings_dim,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        forward pass
        Args:
            input_ids: (B, T)
        Returns:
            (B, T, D)
        """
        hash_ids = input_ids % self.num_embeddings
        return self.emb(hash_ids)