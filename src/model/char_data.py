from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "input.txt"


class CharTokenizer:
    def __init__(self, text):
        raise NotImplementedError

    def encode(self, s):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError


def load_data(path = DATA):
    raise NotImplementedError


def get_batch(data, batch_size, block_size, device = "cpu"):
    raise NotImplementedError
