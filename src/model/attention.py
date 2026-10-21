import torch
import torch.nn as nn
from torch.nn import functional as F


def average_loop(x):
    raise NotImplementedError


def average_matmul(x):
    raise NotImplementedError


def average_softmax(x):
    raise NotImplementedError


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def main():
    import sys

    import matplotlib.pyplot as plt

    from char_data import ROOT, load_data
    from train import build_model

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "experiments" / "deep.pt"
    text = sys.argv[2].replace("\\n", "\n") if len(sys.argv) > 2 else "First Citizen:\nBefore we proceed"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if ckpt["cfg"]["model"] != "gpt":
        sys.exit("bigram 没有 attention，换一个实验的 .pt 文件")

    tok, _, _ = load_data()
    model = build_model(ckpt["cfg"], ckpt["vocab_size"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    text = text[-model.cfg.block_size:]
    model(torch.tensor([tok.encode(text)]))
    labels = ["\\n" if c == "\n" else "␣" if c == " " else c for c in text]

    n_layer, n_head = len(model.blocks), model.cfg.n_head
    fig, axes = plt.subplots(n_layer, n_head, figsize=(3.2 * n_head, 3.2 * n_layer), squeeze=False)
    for l, block in enumerate(model.blocks):
        for h, head in enumerate(block.sa.heads):
            ax = axes[l][h]
            ax.imshow(head.wei[0], cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"layer {l}  head {h}", fontsize=9)
            ax.set_xticks(range(len(labels)), labels, fontsize=6)
            ax.set_yticks(range(len(labels)), labels, fontsize=6)
    fig.tight_layout()
    out = ROOT / "experiments" / f"attention_{ckpt['preset']}.png"
    fig.savefig(out, dpi=150)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
