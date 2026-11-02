from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import Head


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 8
    n_embd: int = 32
    n_head: int = 4
    n_layer: int = 1
    dropout: float = 0.0
    ffwd: bool = True
    residual: bool = True


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList(
            [Head(cfg.n_embd, head_size, cfg.block_size, cfg.dropout) for _ in range(cfg.n_head)]
        )
        self.proj = nn.Linear(head_size * cfg.n_head, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg):
        raise NotImplementedError

    def forward(self, idx, targets = None):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        raise NotImplementedError
