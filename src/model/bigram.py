import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        raise NotImplementedError

    def forward(self, idx, targets=None):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        raise NotImplementedError
