"""
COBRA: Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations
https://arxiv.org/pdf/2503.02453
"""

import torch
from typing import NamedTuple, Optional, Dict
from torch import nn
import json
import torch.nn.functional as F
from generative_recommenders.models.rqvae import RqVae
from generative_recommenders.modules.encoder import BgeEncoder
import pdb

class CobraOutput(NamedTuple):
    """
    cobra output
    """
    loss: torch.Tensor
    loss_sparse: torch.Tensor
    loss_dense: torch.Tensor
    sparse_acc: torch.Tensor
    top5_acc: torch.Tensor
    vec_cos_sim: torch.Tensor
    codebook_entropy: torch.Tensor
    dense_loss_neg_number: torch.Tensor
    

class CobraEmbedding(nn.Module):
    """cobra embedding: e_t^1, e_t^2, ..., e_t^C, v_t
    ---------------------------------------
    ids  : (B, T*C)   # C = n_codebooks
    vecs : (B, T, Dv)
      -> out: (B, (C+1)*T, d_model)
    """
    def __init__(
        self,
        id_vocab_size: int,
        n_codebooks: int = 3,
        d_model: int = 768,
        max_len: int = 1024,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.C = n_codebooks
        self.pad_id = pad_id
        self.id_vocab_size = id_vocab_size
        # for each codebook, build a set of Embedding (parameters are not shared)
        self.id_embed = nn.Embedding(
            id_vocab_size * n_codebooks + 1, d_model, padding_idx=id_vocab_size * n_codebooks
        )
        # token-type: 0…C-1 (sparse), C (dense)
        self.type_embed = nn.Embedding(2, d_model)
        # absolute position: up to max_len
        self.pos_embed  = nn.Embedding(max_len, d_model)

    def forward(self, input_ids: torch.Tensor, input_vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass

        Args:
            input_ids  : (B, T*C)
            input_vecs : (B, T, Dv)
        Returns:
            (B, (C+1)*T, d_model)
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # 1) for each codebook, get embedding -> (B,T,D)
        id_token_type_ids = (torch.arange(T, device=device) % 3).unsqueeze(0).repeat(B, 1)
        emb_mask = (input_ids != self.pad_id)
        emb_input_ids = input_ids.clone()
        emb_input_ids[emb_mask] += id_token_type_ids[emb_mask] * self.id_vocab_size
        id_tok_list = self.id_embed(emb_input_ids)

        # 2) split id_tok_list into chunks
        chunks = id_tok_list.split(self.C, dim=1)  # list of [3, 768] tensors
        
        # 3) insert input_vecs into chunks
        output_chunks = []
        for i, chunk in enumerate(chunks):
            output_chunks.append(chunk)
            
            if i < input_vecs.shape[1]:
                output_chunks.append(input_vecs[:, i, :].unsqueeze(1))  # [B, 1, 768]
                
        # 4) concat chunks
        h = torch.cat(output_chunks, dim=1)  # (B,(C+1)*T,D)

        # 5) add position & type embeddings
        # position: repeat C+1 times for each t ⇒ [0 0 0 0 1 1 1 1 ...]
        pos_idx = torch.arange((T // self.C) + 1, device=device).repeat_interleave(self.C + 1)[:h.shape[1]]
        pos_idx = pos_idx.unsqueeze(0).repeat(B, 1)                      # (B,(C+1)*T)

        # type: first C are 0 (sparse), last 1 is 1 (dense)
        block_type = torch.tensor([0] * self.C + [1], device=device)
        type_idx = block_type.repeat((T // self.C) + 1).unsqueeze(0).repeat(B, 1)[..., :h.shape[1]]        # (B,(C+1)*T)
        
        mask = mask.unsqueeze(-1).float()
        h = h * mask
        h = h + self.pos_embed(pos_idx) * mask
        h = h + self.type_embed(type_idx) * mask
        return h 


class CobraDecoder(nn.Module):
    """
    TransformerDecoder
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate upper-triangular bool mask, shape (L, L)

        Args:
            seq_len: sequence length
            device: device
        Returns:
            (L, L) bool mask
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

    def forward(
        self,
        tgt: torch.Tensor,                 # (B, L, D)
        memory: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                     # --> (B, L, D)
        """
        Forward Pass

        Args:
            tgt: (B, L, D)
            memory: (B, L, D)
        Returns:
            (B, L, D)
        """
        L = tgt.size(1)
        causal_mask = self._causal_mask(L, tgt.device)

        # when memory=None, PyTorch still requires a Tensor; use an empty Tensor as placeholder
        if memory is None:
            memory = torch.zeros(
                (tgt.size(0), 0, tgt.size(2)),
                dtype=tgt.dtype,
                device=tgt.device,
            )
        
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out


class Cobra(torch.nn.Module):
    """
    Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations
    https://arxiv.org/pdf/2503.02453
    """
    def __init__(
        self,
        encoder_model_name: str = "./bge-base-zh",
        id_vocab_size: int = 512,
        n_codebooks: int = 3,
        d_model: int = 768,
        max_len: int = 1024,
        temperature=0.2,
        queue_size=1024,
    ) -> None:
        super().__init__()
        self.C = n_codebooks
        self.d_model = d_model
        self.pad_id = id_vocab_size * self.C
        self.encoder = BgeEncoder(model_name=encoder_model_name)
        self.cobra_emb = CobraEmbedding(
            id_vocab_size=id_vocab_size,
            d_model=d_model,
            max_len=max_len,
            pad_id=self.pad_id
        )
        self.decoder = CobraDecoder(d_model, n_layers=6, n_heads=12)
        self.sparse_head = nn.ModuleList([
            nn.Linear(d_model, id_vocab_size) for _ in range(n_codebooks)
        ])
        self.temperature = temperature
       
        self.register_buffer(
            "feat_queue",
            torch.randn(queue_size, d_model)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        self.feat_queue = F.normalize(self.feat_queue, dim=-1)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, new_feats: torch.Tensor) -> None:
        """
        Enqueue new features, dequeue the oldest ones (circular queue).

        Args:
            new_feats: (n, D)  already L2-normalized features
        """
        n = new_feats.size(0)
        K = self.queue_size
        ptr = int(self.queue_ptr)          

        # case 1: batch size >= queue size
        if n >= K:
            self.feat_queue.copy_(new_feats[-K:])
            self.queue_ptr[0] = 0
        return

        # case 2: no wrap-around, write directly [ptr : ptr+n)
        end_ptr = ptr + n
        if end_ptr <= K:
            self.feat_queue[ptr:end_ptr] = new_feats
        else:
            # case 3: need wrap-around, write in two segments
            first_len = K - ptr                 # how many can be written at the end of the queue
            self.feat_queue[ptr:] = new_feats[:first_len]
            self.feat_queue[:end_ptr - K] = new_feats[first_len:]

        # update pointer (guarantee 0 ≤ queue_ptr < K)
        self.queue_ptr[0] = end_ptr % K


    def interleave_seq_mask(self, seq_mask: torch.Tensor, n: int) -> torch.Tensor:
        """
        Interleave sequence mask
    
        Args:
            seq_mask: (B, L)
            n: number of codebooks
        Returns:
            (B, L + k)
        """
        B, L = seq_mask.shape
        
        k = L // n
        device = seq_mask.device
        dtype = seq_mask.dtype
        
        orig_pos = torch.arange(L, device=device)
        new_pos  = orig_pos + orig_pos // n
        
        g = torch.arange(k, device=device)
        ins_pos = g * (n + 1) + n
        prev_idx = g * n + (n - 1)
        ins_value = seq_mask[:, prev_idx]
        
        new_len = L + k
        out_mask = seq_mask.new_zeros(B, new_len, dtype=dtype)
        
        out_mask.scatter_(
            dim=1,
            index=new_pos.expand(B, -1),
            src=seq_mask
        )
        
        out_mask.scatter_(
            dim=1,
            index=ins_pos.expand(B, -1),
            src=ins_value
        )
        return out_mask
    
    def forward(self, 
        input_ids: torch.Tensor, 
        encoder_input_ids: torch.Tensor, 
        encoder_attention_mask: torch.Tensor,
        encoder_token_type_ids: torch.Tensor,
        mask=None
    ) -> CobraOutput:
        """
        Forward Pass

        Args:
            input_ids: (B, T*C)
            encoder_input_ids: (B, T, Dv)
            encoder_attention_mask: (B, T, Dv)
            encoder_token_type_ids: (B, T, Dv)
        Returns:
            CobraOutput
        """
        
        vecs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            token_type_ids=encoder_token_type_ids
        )

        B, T = input_ids.shape

        seq_mask = (input_ids != self.pad_id)
        
        seq_mask = self.interleave_seq_mask(seq_mask, self.C)
        # print("seq_mask:", seq_mask)
        # ---------- ① Decoder ----------
        emb = self.cobra_emb(input_ids, vecs, seq_mask)
        
        h = self.decoder(emb, tgt_key_padding_mask=~seq_mask)       # (B,L,D)
        L = h.size(1)

        T = T // self.C
        # ---------- ② Sparse ID Loss ----------
        loss_sparse = 0.0
        total_correct, total_top5, total_tokens = 0, 0, 0
        for c in range(self.C):
            if c == 0:
                # e_{t+1}^0  ←  v_t
                #  logits come from v_t (last step has no target)
                pos_c = torch.arange(0, T - 1, device=h.device) * (self.C + 1) + self.C
                logits = self.sparse_head[0](h[:, pos_c, :])        # (B, T-1, V)
                target_pos = torch.arange(1, T, device=h.device) * self.C
                target = input_ids[:, target_pos]                              # (B, T-1)
            else:
                # e_t^c  ←  e_t^{c-1}   (same step internal dependency)
                pos_c = torch.arange(0, T - 1, device=h.device) * (self.C + 1) + (c - 1)
                logits = self.sparse_head[c](h[:, pos_c, :])        # (B, T-1, V)
                target_pos = torch.arange(1, T, device=h.device) * self.C + c
                target = input_ids[:, target_pos]                              # (B, T-1)
            
            # cross entropy + padding mask
            loss_c = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=self.pad_id,
                reduction="sum"
            )
            valid_tokens = (target != self.pad_id).sum()
            loss_sparse += loss_c / valid_tokens.clamp(min=1)

            with torch.no_grad():
                valid_mask = target != self.pad_id                  # (B,T-1)
                pred_top1 = logits.argmax(-1)
                top1_correct = (pred_top1 == target) & valid_mask
                top5_correct = (logits.topk(5, -1).indices == target.unsqueeze(-1)).any(-1) & valid_mask

                total_correct += top1_correct.sum()
                total_top5 += top5_correct.sum()
                total_tokens += valid_mask.sum()

        loss_sparse = loss_sparse / self.C
        
        
        # ---------- ③ Dense InfoNCE ----------
        # vec token position: t*(C+1)+C, only take the first T-1 steps
        vec_pos = torch.arange(0, T, device=h.device) * (self.C + 1) + self.C - 1
        vec_pred = h[:, vec_pos, :self.d_model]                     # (B,T-1,Dv)
        vec_gt = vecs[:, :, :].detach()                             # (B,T-1,Dv)
        

        Q = B * T
        valid = seq_mask[..., ::(self.C + 1)].reshape(-1)
        vec_pred = vec_pred.reshape(Q, -1)[valid]
        vec_gt = vec_gt.reshape(Q, -1)[valid]
        vec_pred = F.normalize(vec_pred, p=2, dim=-1, eps=1e-12)
        vec_gt = F.normalize(vec_gt, p=2, dim=-1, eps=1e-12)

        
        # in-batch InfoNCE
        seq_ids_raw = torch.arange(B, device=h.device).unsqueeze(1)  # (B,1)
        seq_ids_raw = seq_ids_raw.expand(-1, T).reshape(-1)          # (Q,)
        seq_ids = seq_ids_raw[valid]                                 # (Q_valid,)
        same_seq = seq_ids.unsqueeze(0) == seq_ids.unsqueeze(1)
        same_seq.fill_diagonal_(False) 
        sim = (vec_pred @ vec_gt.T) / self.temperature
        sim = sim.masked_fill(same_seq, -1e4)
        labels = torch.arange(sim.size(0), device=sim.device)
        loss_dense = F.cross_entropy(sim, labels, reduction="mean")
        # calculate negative number
        dense_loss_neg_number = torch.tensor(((~same_seq).sum().item() - sim.size(0)) / Q, device=sim.device)

        """
        # cross-batch InfoNCE
        feat_neg = self.feat_queue.clone().detach()
        logits_pos = (vec_pred * vec_gt).sum(-1, keepdim=True)  # (Q,1)
        logits_neg = vec_pred @ feat_neg.t()             # (Q,K)
        logits = torch.cat([logits_pos, logits_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss_dense = F.cross_entropy(logits, labels, reduction="mean")

        with torch.no_grad():
            self._dequeue_and_enqueue(vec_gt)
        """

        # ---------- ④ Metrics ----------
        sparse_acc = total_correct.float() / total_tokens.clamp(min=1)
        top5_acc = total_top5.float() / total_tokens.clamp(min=1)

        vec_cos_sim = F.cosine_similarity(vec_pred, vec_gt).mean()

        # ---------- ⑤ Codebook entropy ----------
        with torch.no_grad():
            # ids: (B,T,C) -> (C, V)
            usage = torch.stack([F.one_hot(input_ids[:, c::3], self.pad_id + 1).sum((0, 1)).float()  for c in range(self.C)])  # (C, V)
            prob = usage / usage.sum(1, keepdim=True)
            codebook_entropy = -(prob * (prob.add(1e-12).log())).sum(1).mean()


        return CobraOutput(
            loss=loss_sparse + loss_dense,
            loss_sparse=loss_sparse,
            loss_dense=loss_dense,
            sparse_acc=sparse_acc,
            top5_acc=top5_acc,
            vec_cos_sim=vec_cos_sim,
            codebook_entropy=codebook_entropy, # TODO
            dense_loss_neg_number=dense_loss_neg_number
        )
    
    def generate(self, 
        input_ids: torch.Tensor, 
        encoder_input_ids: torch.Tensor, 
        encoder_attention_mask: torch.Tensor,
        encoder_token_type_ids: torch.Tensor,
        mask=None
    ) -> CobraOutput:
        """
        generate next token and useremb
        Args:
            input_ids: (B, T*C)
            encoder_input_ids: (B, T, Dv)
            encoder_attention_mask: (B, T, Dv)
            encoder_token_type_ids: (B, T, Dv)
        Returns:
            token and useremb
        """
        vecs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            token_type_ids=encoder_token_type_ids
        )
        print("vecs.shape", vecs.shape)
        print("input_ids.shape", input_ids.shape)
        B, T = input_ids.shape

        seq_mask = (input_ids != self.pad_id)
        print("seq_mask:", seq_mask.shape)
        seq_mask = self.interleave_seq_mask(seq_mask, self.C)
        print("seq_mask:", seq_mask)
        # ---------- ① Decoder ----------
        ## Todo: beam serach + batch decoder
        emb = self.cobra_emb(input_ids, vecs, seq_mask)
        print(emb.shape)
        h = self.decoder(emb, tgt_key_padding_mask=~seq_mask)       # (B,L,D)
    
    def generate_itemvec(self, 
        encoder_input_ids: torch.Tensor, 
        encoder_attention_mask: torch.Tensor,
        encoder_token_type_ids: torch.Tensor)
        """
        generate itememb
        Args:
            encoder_input_ids: (B, T, Dv)
            encoder_attention_mask: (B, T, Dv)
            encoder_token_type_ids: (B, T, Dv)
        Returns:
            itememb
        """
        vecs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            token_type_ids=encoder_token_type_ids
        )
        vecs = F.normalize(vecs, p=2, dim=-1, eps=1e-12)
        return vecs
    
if __name__ == "__main__":
    cobra = Cobra()
    #cobra.load_state_dict(torch.load("./out/cobra/decoder/cobra_480000.pt")).cuda()
    cobra.cuda()
    cobra.eval()
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 512 * 3, 512 * 3, 512 * 3],
    ]).cuda()
    encoder_input_ids = torch.tensor([
        [
            [2, 3, 4, 0, 0, 0],
            [5, 1, 2, 4, 0, 0],
            [5, 1, 2, 4, 2, 1],
        ],
        [
            [5, 1, 2, 3, 7, 8],
            [5, 6, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ],
    ]).cuda()
    encoder_attention_mask = torch.tensor([
        [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]],
    ]).cuda()
    encoder_token_type_ids = torch.tensor([
        [[0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]],
    ]).cuda()
    out = cobra.generate(
        input_ids=input_ids,
        encoder_input_ids=encoder_input_ids,
        encoder_attention_mask=encoder_attention_mask,
        encoder_token_type_ids=encoder_token_type_ids,
    )
