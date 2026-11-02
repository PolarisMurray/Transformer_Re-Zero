import torch
import torch.nn as nn
from torch.nn import functional as F
## example:
x = torch.tensor(
    [
        [1, 0, 0, 1, 0], #cat
        [0, 1, 0, 1, 0], # sat
        [0, 0, 1, 0, 1] # on
    ]
)

# print(x.shape)
# Batch: 第几条序列, Time: 第几个Token(how many tokens), Channel: Token的第几个特征(how many features)

def average_loop(x):
    B, T, C = x.shape
    out = torch.zeros_like(x)

    for b in range(B):
        for t in range(T):
            prefix = x[b, :t + 1, :] # here the shape of batch B is removed
            avg = prefix.mean(dim=0)
            out[b, t, :] = avg
    return out


def average_matmul(x):
    B, T, C = x.shape
    wei = torch.tril(torch.ones(T, T))
    Sum = wei.sum(dim=1, keepdim=True)
    wei = wei / Sum
    out = wei @ x
    return out


def average_softmax(x):
    B, T, C = x.shape
    scores = torch.zeros(T, T)
    mask = torch.tril(torch.ones(T, T, dtype = bool))
    scores = scores.masked_fill(~mask, float('-inf'))
    wei = scores.softmax(dim=1)
    out = wei @ x
    return out


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size, dtype=bool))
        )

        self.dropout= nn.Dropout(dropout)
        self.wei = None


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        
        scores = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)

        # masking
        mask = self.tril[:T, :T]
        scores = scores.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        self.wei = attn.detach()
        attn = self.dropout(attn)
        out = attn @ v

        return out
        


def main():
    import sys

    import matplotlib.pyplot as plt

    from char_data import ROOT, load_data
    from train import build_model

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "experiments" / "deep.pt"
    text = sys.argv[2].replace("\\n", "\n") if len(sys.argv) > 2 else "First Citizen:\nBefore we proceed"
    ckpt = torch.load(ckpt_path, map_location = "cpu")
    if ckpt["cfg"]["model"] != "gpt":
        sys.exit("bigram has no attention, pick a checkpoint from another preset")

    tok, _, _ = load_data()
    model = build_model(ckpt["cfg"], ckpt["vocab_size"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    text = text[-model.cfg.block_size:]
    model(torch.tensor([tok.encode(text)]))
    labels = ["\\n" if c == "\n" else "␣" if c == " " else c for c in text]

    n_layer, n_head = len(model.blocks), model.cfg.n_head
    fig, axes = plt.subplots(n_layer, n_head, figsize = (3.2 * n_head, 3.2 * n_layer), squeeze = False)
    for l, block in enumerate(model.blocks):
        for h, head in enumerate(block.sa.heads):
            ax = axes[l][h]
            ax.imshow(head.wei[0], cmap = "Blues", vmin = 0, vmax = 1)
            ax.set_title(f"layer {l}  head {h}", fontsize = 9)
            ax.set_xticks(range(len(labels)), labels, fontsize = 6)
            ax.set_yticks(range(len(labels)), labels, fontsize = 6)
    fig.tight_layout()
    out = ROOT / "experiments" / f"attention_{ckpt['preset']}.png"
    fig.savefig(out, dpi = 150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
