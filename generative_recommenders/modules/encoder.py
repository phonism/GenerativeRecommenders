"""
encoder module
"""
from torch import nn
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from generative_recommenders.modules.normalize import L2NormalizationLayer
from typing import List
from torch import nn
from torch import Tensor

class SentenceT5Encoder(nn.Module):
    """
    Sentence T5 Encoder
    """
    def __init__(self, model_name="./models_hub/sentence-t5-xl") -> None:
        """
        Initialize the SentenceT5Encoder

        Args:
            model_name: path to the model
        """
        super().__init__()
        full_model = SentenceTransformer(model_name)

        # get submodules
        self.tokenizer = full_model.tokenizer
        self.encoder_model = full_model._modules["0"].auto_model.encoder
        self.pooling = full_model._modules["1"]
        self.dense = full_model._modules["2"]             # Dense(1024→768)

        self.encoder_model.gradient_checkpointing_enable()
    
    def tokenize(self, texts) -> torch.Tensor:
        """
        Tokenize texts

        Args:
            texts: (B, T)
        Returns:
            torch.Tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.encoder_model.device)
        return inputs
    
    def encode(self, texts) -> torch.Tensor:
        """
        Encode texts to embeddings

        Args:
            texts: (B, T)
        Returns:
            torch.Tensor
        """
        inputs = self.tokenize(texts)
        return self.forward(inputs)

    def forward(self, batch_tokens):
        """
        Forward Pass

        Args:
            batch_tokens: (B, T, C)
        Returns:
            torch.Tensor
        """
        batch_tokens.fill_(1123)
        if batch_tokens.dim() == 3:
            B, T, C = batch_tokens.shape
            flat_tokens = batch_tokens.view(B * T, C)      # (B*T, C)
        elif batch_tokens.dim() == 2:
            B, C = batch_tokens.shape
            T = 1
            flat_tokens = batch_tokens                     # (B, C)
        else:
            raise ValueError(f"Expect 2-D or 3-D, got {batch_tokens.dim()}-D")
        attention_mask = (flat_tokens != 0).long()         # (B*T, C) or (B, C)
        # encode
        encoder_outputs = self.encoder_model(
            input_ids=flat_tokens,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = encoder_outputs.last_hidden_state  # (B, L, 1024)

        # pooling + proj
        features = {
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask
        }
        pooled = self.pooling(features)["sentence_embedding"]  # (B, 1024)
        dense = self.dense({"sentence_embedding": pooled})["sentence_embedding"]
        dense_norm = torch.nn.functional.normalize(dense, p=2, dim=-1)
        if T > 1:                                               # if it is 3-D
            dense_norm = dense_norm.view(B, T, -1)               # (B, T, D)
        return dense_norm


class ErnieEncoder(nn.Module):
    """
    Ernie Encoder
    """
    def __init__(self, model_name="./models_hub/ernie-3.0-medium-zh", out_dim=768) -> None:
        """
        Initialize the ErnieEncoder

        Args:
            model_name: path to the model
            out_dim: output dimension
        """
        super().__init__()
        self.out_dim = out_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.pooler = None
        #self.backbone.pooler.dense.bias.requires_grad_(False)
        #self.backbone.pooler.dense.weight.requires_grad_(False)
        #self.backbone.gradient_checkpointing_enable()
    
    def encode(self, texts) -> torch.Tensor:
        """
        encode texts to embeddings

        Args:
            texts: (B, T)
        Returns:
            torch.Tensor
        """
        batch_tokens = self.tokenize(texts)
        return self.forward(batch_tokens["input_ids"]).view(-1, self.out_dim)
    
    def tokenize(self, texts) -> torch.Tensor:
        """
        Tokenize texts

        Args:
            texts: (B, T)
        Returns:
            torch.Tensor
        """
        if isinstance(texts, str):
            texts = [[texts]]                      # (1,1)
        elif isinstance(texts[0], str):
            texts = [[t] for t in texts]           # (B,1)

        B, T = len(texts), len(texts[0])           # (B,T)
        flat_texts = [t for seq in texts for t in seq]     # length B*T

        batch_tokens = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=102400,
            return_tensors="pt"
        ).to(self.backbone.device)

        return batch_tokens

    def forward(self, batch_tokens) -> torch.Tensor:
        """
        Forward Pass

        Args:
            batch_tokens: (B, T, C)
        Returns:
            torch.Tensor
        """
        if batch_tokens.dim() == 3:
            B, T, C = batch_tokens.shape
            flat_tokens = batch_tokens.view(B * T, C)       # (B*T, C)
        elif batch_tokens.dim() == 2:
            B, C = batch_tokens.shape
            T = 1
            flat_tokens = batch_tokens                      # (B, C)
        else:
            raise ValueError(f"Expect 2-D or 3-D, got {batch_tokens.dim()}-D")
        attention_mask = (flat_tokens != 0).long()          # (B*T, C) or (B, C)

        # -------- backbone forward --------
        outputs = self.backbone(
            input_ids=flat_tokens,
            attention_mask=attention_mask,
            return_dict=True,
        )
        h_cls = outputs.last_hidden_state[:, 0]             # (B*T, H)

        z = h_cls

        if T > 1:                                           # if it is 3-D
            z = z.view(B, T, -1)                            # (B, T, D)
        return z                                            # (B, T, D) or (B, D)


class BgeEncoder(nn.Module):
    """
    BGE Encoder
    """
    def __init__(self, model_name="./models_hub/bge-base-zh"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.pooler = None
        self.backbone.gradient_checkpointing_enable()
        self.out_dim = self.backbone.config.hidden_size

    def tokenize(self, texts) -> torch.Tensor:
        """
        Tokenize texts

        Args:
            texts: (B, T)
        Returns:
            torch.Tensor
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(self.backbone.device)
    
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids
    ) -> torch.Tensor:
        """
        Forward Pass

        Args:
            input_ids: (B, T, C)
            attention_mask: (B, T, C)
            token_type_ids: (B, T, C)
        Returns:
            torch.Tensor
        """
        if input_ids.dim() == 3:
            B, T, C = input_ids.shape
            flat_tokens = input_ids.view(B * T, C)
            flat_attention_mask = attention_mask.view(B * T, C)
            flat_token_type_ids = token_type_ids.view(B * T, C)
        elif input_ids.dim() == 2:
            B, C = input_ids.shape
            T = 1
            flat_tokens = input_ids
            flat_attention_mask = attention_mask.view(B * T, C)
            flat_token_type_ids = token_type_ids.view(B * T, C)
        else:
            raise ValueError(f"Expect 2-D or 3-D, got {input_ids.dim()}-D")
        outputs = self.backbone(
            input_ids=flat_tokens,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )
        hidden_state = outputs.last_hidden_state[:, 0]
        hidden_state = torch.nn.functional.normalize(hidden_state, p=2, dim=-1)
        if T > 1:
            hidden_state = hidden_state.view(B, T, -1)
        return hidden_state

    @torch.no_grad()
    def encode(self, texts, device="cuda") -> torch.Tensor:
        """
        Encode texts to embeddings

        Args:
            texts: (B, T)
            device: device
        Returns:
            torch.Tensor
        """
        batch = self.tokenize(texts)
        return self.forward(**batch)


class MLP(nn.Module):
    """
    MLP Layer
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.0,
        normalize: bool = False
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout

        dims = [self.input_dim] + self.hidden_dims + [self.out_dim]
        
        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))
            if i != len(dims) - 2:
                self.mlp.append(nn.SiLU())
                if dropout != 0:
                    self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(L2NormalizationLayer() if normalize else nn.Identity())

    def forward(self, x: Tensor) -> torch.Tensor:
        """
        Forward Pass

        Args:
            x: (B, D)
        Returns:
            torch.Tensor
        """
        assert x.shape[-1] == self.input_dim, f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)

if __name__ == "__main__":
    bge = SentenceTransformer("./models_hub/bge-base-zh")
    bge_new = BgeEncoder("./models_hub/bge-base-zh")
    print(bge.encode(["hello world"])[0][:10])
    print(bge_new.encode(["hello world"])[0][:10])