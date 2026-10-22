import argparse
import time

import torch

from char_data import ROOT, get_batch, load_data
from bigram import BigramLanguageModel
from gpt import GPT, GPTConfig


EXP_DIR = ROOT / "experiments"


@torch.no_grad()
def estimate_loss(model, splits, batch_size, block_size, device, eval_iters = 200):
    raise NotImplementedError


def train(model, splits, cfg, device, eval_interval = 500, eval_iters = 200, log = print):
    raise NotImplementedError


SMALL = dict(
    model = "gpt",
    batch_size = 32,
    block_size = 8,
    n_embd = 32,
    dropout = 0.0,
    lr = 1e-3,
    max_iters = 5000,
)

PRESETS = {
    "bigram":      dict(SMALL, model = "bigram", lr = 1e-2),
    "single_head": dict(SMALL, n_head = 1, n_layer = 1, ffwd = False, residual = False),
    "multi_head":  dict(SMALL, n_head = 4, n_layer = 1, ffwd = False, residual = False),
    "ffwd":        dict(SMALL, n_head = 4, n_layer = 1, ffwd = True,  residual = False),
    "deep_no_res": dict(SMALL, n_head = 4, n_layer = 4, ffwd = True,  residual = False),
    "deep":        dict(SMALL, n_head = 4, n_layer = 4, ffwd = True,  residual = True),
    "medium": dict(
        model = "gpt",
        batch_size = 32,
        block_size = 64,
        n_embd = 128,
        n_head = 4,
        n_layer = 4,
        ffwd = True,
        residual = True,
        dropout = 0.1,
        lr = 1e-3,
        max_iters = 5000,
    ),
    "big": dict(
        model = "gpt",
        batch_size = 64,
        block_size = 256,
        n_embd = 384,
        n_head = 6,
        n_layer = 6,
        ffwd = True,
        residual = True,
        dropout = 0.2,
        lr = 3e-4,
        max_iters = 5000,
    ),
}


def build_model(cfg, vocab_size):
    if cfg["model"] == "bigram":
        return BigramLanguageModel(vocab_size)

    gpt_cfg = GPTConfig(
        vocab_size = vocab_size,
        block_size = cfg["block_size"],
        n_embd = cfg["n_embd"],
        n_head = cfg["n_head"],
        n_layer = cfg["n_layer"],
        dropout = cfg["dropout"],
        ffwd = cfg["ffwd"],
        residual = cfg["residual"],
    )
    return GPT(gpt_cfg)


def pick_device(cfg):
    small_model = cfg["model"] == "bigram" or cfg["n_embd"] < 128
    if small_model:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def sample(model, tok, device, n = 500):
    model.eval()
    start = torch.zeros((1, 1), dtype = torch.long, device = device)
    ids = model.generate(start, n)[0].tolist()
    model.train()
    return tok.decode(ids)


def save_sample(name, losses, before, after):
    text = f"val loss {losses['val']:.4f}\n"
    text += f"\n--- before training ---\n{before}\n"
    text += f"\n--- after training ---\n{after}\n"
    path = EXP_DIR / f"{name}.txt"
    path.write_text(text, encoding = "utf-8")


def save_checkpoint(name, cfg, vocab_size, model):
    ckpt = {
        "preset": name,
        "cfg": cfg,
        "vocab_size": vocab_size,
        "model": model.state_dict(),
    }
    torch.save(ckpt, EXP_DIR / f"{name}.pt")


def append_result(name, cfg, n_params, losses, minutes, device):
    path = EXP_DIR / "results.md"
    if not path.exists():
        header = "| preset | params | steps | train loss | val loss | time | device |\n"
        header += "|---|---|---|---|---|---|---|\n"
        path.write_text(header, encoding = "utf-8")

    row = (
        f"| {name} | {n_params:,} | {cfg['max_iters']} "
        f"| {losses['train']:.4f} | {losses['val']:.4f} "
        f"| {minutes:.1f} min | {device} |\n"
    )
    with path.open("a", encoding = "utf-8") as f:
        f.write(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices = PRESETS)
    parser.add_argument("--iters", type = int)
    parser.add_argument("--device", default = "auto")
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.preset

    cfg = dict(PRESETS[name])
    if args.iters is not None:
        cfg["max_iters"] = args.iters

    device = args.device
    if device == "auto":
        device = pick_device(cfg)

    torch.manual_seed(1337)
    tok, train_data, val_data = load_data()
    splits = {"train": train_data, "val": val_data}

    model = build_model(cfg, tok.vocab_size).to(device)
    n_params = count_params(model)
    print(f"[{name}] {n_params:,} params, device = {device}")

    before = sample(model, tok, device, n = 200)
    print("\n--- before training ---")
    print(before)

    t0 = time.time()
    losses = train(model, splits, cfg, device)
    minutes = (time.time() - t0) / 60

    after = sample(model, tok, device, n = 500)
    print("\n--- after training ---")
    print(after)

    EXP_DIR.mkdir(exist_ok = True)
    save_sample(name, losses, before, after)
    save_checkpoint(name, cfg, tok.vocab_size, model)
    append_result(name, cfg, n_params, losses, minutes, device)
    print(f"\nsaved experiments/{name}.txt and experiments/{name}.pt")
    print("appended results to experiments/results.md")


if __name__ == "__main__":
    main()
