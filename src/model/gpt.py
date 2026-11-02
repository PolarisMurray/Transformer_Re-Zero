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
        super().__init__()
        self.residual = cfg.residual
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg) if cfg.ffwd else None
        if cfg.residual:
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        if self.residual:
            return self.forward_residual(x)
        return self.forward_plain(x)

    def forward_residual(self, x):
        x = x + self.sa(self.ln1(x))
        if self.ffwd is not None:
            x = x + self.ffwd(self.ln2(x))
        return x

    def forward_plain(self, x):
        x = self.sa(x)
        if self.ffwd is not None:
            x = self.ffwd(x)
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd) if cfg.residual else nn.Identity()
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets = None):
        T = idx.shape[1]
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        return logits, self.compute_loss(logits, targets)

    def compute_loss(self, logits, targets):
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
