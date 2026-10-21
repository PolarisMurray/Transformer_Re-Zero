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
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class FeedForward(nn.Module):
    def __init__(self, cfg):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, cfg):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg):
        raise NotImplementedError

    def forward(self, idx, targets=None):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        raise NotImplementedError
