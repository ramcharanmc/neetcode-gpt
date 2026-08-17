import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.Embedding_dim = embedding_dim
        self.Attention_dim = attention_dim
        self.Key_gen = nn.Linear(self.Embedding_dim, self.Attention_dim, bias=False)
        self.Query_gen = nn.Linear(self.Embedding_dim, self.Attention_dim, bias=False)
        self.Value_gen = nn.Linear(self.Embedding_dim, self.Attention_dim, bias=False)
        

        

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        embedded = torch.as_tensor(embedded)
        k = self.Key_gen(embedded)   # (B, T, attention_dim)
        q = self.Query_gen(embedded) # (B, T, attention_dim)
        v = self.Value_gen(embedded) # (B, T, attention_dim)

        # Attention scores: (Q @ K^T) / sqrt(d_k)
        scores = q @ torch.transpose(k, 1, 2)
        context_length, attention_dim = k.shape[1], k.shape[2]
        scores = scores / (attention_dim ** 0.5)

        # Causal mask: prevent attending to future tokens
        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim=2)

        return torch.round(scores @ v, decimals=4)
        pass
