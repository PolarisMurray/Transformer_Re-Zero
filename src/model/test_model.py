## Lesson-by-lesson checks for the hand-written model
## Usage (run from the Transformer_Re-Zero directory):
##   python src/model/test_model.py              check every lesson in src/model
##   python src/model/test_model.py 1 2          check only lessons 1 and 2
##   python src/model/test_model.py --ref        check the reference answers (to make sure the tests themselves are right)

import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
argv = sys.argv[1:]

if "--ref" in argv:
    sys.path.insert(0, str(ROOT / "answers" / "model_reference"))   # reference answers (gitignored, local only)
    WHO = "answers/model_reference"
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    WHO = "src/model"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


## ------------------------------------------------------------
def lesson0():
    from char_data import CharTokenizer, get_batch, load_data

    tok = CharTokenizer("hello world")
    check(tok.vocab_size == 8, f"'hello world' has 8 distinct characters, but vocab_size is {tok.vocab_size}")
    check(tok.stoi[" "] == 0 and tok.itos[0] == " ", "sort the characters before numbering them: space is the smallest, so it should be id 0")
    check(tok.encode("hello") == [3, 2, 4, 4, 5], f"encode('hello') should be [3, 2, 4, 4, 5], got {tok.encode('hello')}")
    check(tok.decode([7, 5, 6, 4, 1]) == "world", "decode([7, 5, 6, 4, 1]) should be 'world'")

    tok, train_data, val_data = load_data()
    check(tok.vocab_size == 65, f"shakespeare has 65 distinct characters, got {tok.vocab_size}")
    check(isinstance(train_data, torch.Tensor) and train_data.dtype == torch.long, "train_data should be a tensor with dtype=torch.long")
    check(len(train_data) + len(val_data) == 1115394, "train + val together should be the whole text, 1,115,394 characters")
    check(len(train_data) == int(0.9 * 1115394), f"train should be the first 90% ({int(0.9 * 1115394):,}), got {len(train_data):,}")
    check(tok.decode(train_data[:14].tolist()) == "First Citizen:", "the start of train_data should decode to 'First Citizen:'")

    data = torch.arange(1000)
    x, y = get_batch(data, 4, 8)
    check(x.shape == (4, 8) and y.shape == (4, 8), f"x and y should both be (4, 8), got {tuple(x.shape)} and {tuple(y.shape)}")
    check(torch.equal(y, x + 1), "y should be x shifted by one (with arange as data, that means y == x + 1)")
    check(not torch.equal(get_batch(data, 4, 8)[0], get_batch(data, 4, 8)[0]), "two batches are identical: the start offsets should be random")

    small = torch.arange(10)        # length 10, block_size 9: the only valid start is 0
    for _ in range(50):
        x, y = get_batch(small, 2, 9)
        check(x.shape == (2, 9) and y.shape == (2, 9) and (x[:, 0] == 0).all(),
              "wrong upper bound for the start offset: with length 10 and block_size 9, only start 0 gives a full x and y")


## ------------------------------------------------------------
def lesson1():
    from bigram import BigramLanguageModel

    torch.manual_seed(0)
    m = BigramLanguageModel(65)
    check(n_params(m) == 65 * 65, f"bigram is a single 65x65 table, so it should have 4225 params, got {n_params(m)}")

    idx = torch.randint(0, 65, (4, 8))
    targets = torch.randint(0, 65, (4, 8))
    logits, loss = m(idx)
    check(logits.shape == (4, 8, 65), f"logits should be (4, 8, 65), got {tuple(logits.shape)}")
    check(loss is None, "loss should be None when no targets are given")

    logits, loss = m(idx, targets)
    check(logits.shape == (4, 8, 65), "with targets, the returned logits should still be (B, T, C); only flatten them to compute the loss")
    check(loss.dim() == 0, "loss should be a scalar")
    expect = F.cross_entropy(logits.reshape(-1, 65), targets.reshape(-1))
    check(torch.allclose(loss, expect), f"wrong loss: yours is {loss.item():.4f}, expected {expect.item():.4f}")
    check(4.0 < loss.item() < 5.5, f"loss right after init should be around 4.2 ~ 5, got {loss.item():.4f}")

    start = torch.zeros((2, 1), dtype = torch.long)
    out = m.generate(start, 20)
    check(out.shape == (2, 21), f"generate(start of shape (2, 1), 20) should return (2, 21), got {tuple(out.shape)}")
    check((out[:, 0] == 0).all(), "generate should keep the original start and append new tokens after it")
    check(((out >= 0) & (out < 65)).all(), "generated ids are outside the vocabulary")
    torch.manual_seed(1)
    a = m.generate(start, 50)
    torch.manual_seed(2)
    b = m.generate(start, 50)
    check(not torch.equal(a, b), "different random seeds gave identical output: sample with torch.multinomial, don't use argmax")


## ------------------------------------------------------------
def lesson2():
    from bigram import BigramLanguageModel
    from char_data import load_data
    from train import estimate_loss, train

    torch.manual_seed(0)
    tok, train_data, val_data = load_data()
    splits = {"train": train_data, "val": val_data}

    class Spy(BigramLanguageModel):
        modes = []

        def forward(self, idx, targets = None):
            self.modes.append(self.training)
            return super().forward(idx, targets)

    m = Spy(tok.vocab_size)
    out = estimate_loss(m, splits, 8, 8, "cpu", eval_iters = 10)
    check(set(out) == {"train", "val"}, f"estimate_loss should return {{'train': ..., 'val': ...}}, got keys {set(out)}")
    check(all(isinstance(v, float) for v in out.values()), "estimate_loss should return python floats (use .item() or .mean().item())")
    check(len(Spy.modes) == 20, f"eval_iters=10 with two splits should call the model 20 times, got {len(Spy.modes)}")
    check(not any(Spy.modes), "the model should be in eval mode during evaluation: call model.eval() first")
    check(m.training, "call model.train() at the end of estimate_loss to switch back to training mode")

    torch.manual_seed(0)
    m = BigramLanguageModel(tok.vocab_size)
    logs = []
    cfg = dict(lr = 1e-2, max_iters = 300, batch_size = 32, block_size = 8)
    out = train(m, splits, cfg, "cpu", eval_interval = 100, eval_iters = 20, log = logs.append)
    check(len(logs) == 4, f"max_iters=300 and eval_interval=100 should evaluate at steps 0/100/200/300, 4 times total, got {len(logs)}")
    check(set(out) == {"train", "val"}, "train should return the result of the last estimate_loss")
    check(out["val"] < 4.0, f"bigram with lr=1e-2 for 300 steps should bring val loss from 4.7 to below 4, got {out['val']:.4f}")


## ------------------------------------------------------------
def lesson3():
    from attention import Head, average_loop, average_matmul, average_softmax

    torch.manual_seed(0)
    x = torch.randn(2, 5, 3)
    expect = torch.stack([torch.stack([x[b, :t + 1].mean(0) for t in range(5)]) for b in range(2)])
    for f in (average_loop, average_matmul, average_softmax):
        out = f(x)
        check(out.shape == x.shape, f"{f.__name__} should return the same shape as its input {tuple(x.shape)}, got {tuple(out.shape)}")
        check(torch.allclose(out, expect, atol = 1e-6), f"{f.__name__} is wrong: out[b, t] should be the mean of x[b, 0..t]")

    head = Head(16, 8, block_size = 10)
    for name in ("key", "query", "value"):
        layer = getattr(head, name, None)
        check(isinstance(layer, nn.Linear) and layer.bias is None and layer.weight.shape == (8, 16),
              f"self.{name} should be nn.Linear(16, 8, bias=False)")
    check("tril" in dict(head.named_buffers()), "register the lower-triangular mask with self.register_buffer('tril', ...)")
    check(n_params(head) == 3 * 16 * 8, f"Head(16, 8) should have only 3*16*8 = 384 params, got {n_params(head)} (tril must not be a parameter)")

    x = torch.randn(2, 6, 16)        # T=6 is shorter than block_size=10
    out = head(x)
    check(out.shape == (2, 6, 8), f"Head output should be (B, T, head_size) = (2, 6, 8), got {tuple(out.shape)}")
    wei = head.wei
    check(wei is not None and wei.shape == (2, 6, 6), "forward should store the attention weights in self.wei, shape (B, T, T)")
    check(torch.allclose(wei.sum(-1), torch.ones(2, 6), atol = 1e-5), "each row of self.wei should sum to 1 (after softmax)")
    check((wei.triu(1) == 0).all(), "the upper triangle of self.wei should be all zeros: no looking into the future")

    q, k, v = head.query(x), head.key(x), head.value(x)
    raw = q @ k.transpose(-2, -1)
    mask = torch.tril(torch.ones(6, 6)) == 0
    good = F.softmax(raw.masked_fill(mask, float("-inf")) * 8 ** -0.5, dim = -1)
    if not torch.allclose(wei, good, atol = 1e-5):
        noscale = F.softmax(raw.masked_fill(mask, float("-inf")), dim = -1)
        if torch.allclose(wei, noscale, atol = 1e-5):
            raise AssertionError("forgot to divide by sqrt(head_size) (multiply by head_size ** -0.5)")
        raise AssertionError("wrong attention weights: should be softmax(mask(q @ k^T / sqrt(head_size)))")
    check(torch.allclose(out, good @ v, atol = 1e-5), "output should be wei @ v")

    x2 = x.clone()
    x2[:, 4] += 10.0
    check(torch.allclose(head(x2)[:, :4], out[:, :4], atol = 1e-5), "changing the input at position 4 changed outputs at positions 0~3: it is peeking at the future")


## ------------------------------------------------------------
def lesson4():
    from gpt import GPT, Block, FeedForward, GPTConfig, MultiHeadAttention

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size = 65, block_size = 8, n_embd = 32, n_head = 4, n_layer = 2)
    x = torch.randn(2, 8, 32)

    mha = MultiHeadAttention(cfg)
    check(isinstance(getattr(mha, "heads", None), nn.ModuleList) and len(mha.heads) == 4,
          "self.heads should be an nn.ModuleList of 4 Heads")
    check(n_params(mha) == 4 * 3 * 32 * 8 + 32 * 32 + 32,
          f"MultiHeadAttention should have 4128 params (4 heads + proj), got {n_params(mha)}")
    check(mha(x).shape == (2, 8, 32), "MultiHeadAttention output should be (B, T, n_embd)")

    ff = FeedForward(cfg)
    check(n_params(ff) == 32 * 128 + 128 + 128 * 32 + 32, f"FeedForward should have 8352 params (32 -> 128 -> 32), got {n_params(ff)}")
    check(ff(x).shape == (2, 8, 32), "FeedForward output should be (B, T, n_embd)")

    plain = Block(GPTConfig(n_embd = 32, n_head = 4, ffwd = False, residual = False))
    check(n_params(plain) == 4128, "with ffwd=False and residual=False, Block should contain only one MultiHeadAttention")
    check(torch.allclose(plain(x), plain.sa(x)), "without residual and ffwd, Block output should just be self.sa(x)")

    blk = Block(cfg)
    check(n_params(blk) == 4128 + 8352 + 2 * 64, f"a full Block should have 12608 params (sa + ffwd + two LayerNorms), got {n_params(blk)}")
    h = x + blk.sa(blk.ln1(x))
    check(torch.allclose(blk(x), h + blk.ffwd(blk.ln2(h)), atol = 1e-6),
          "with residual it should be x = x + sa(ln1(x)), then x = x + ffwd(ln2(x))")

    presets = {   # matches the presets in train.py
        "single_head": (dict(n_head = 1, n_layer = 1, ffwd = False, residual = False), 8609),
        "multi_head":  (dict(n_head = 4, n_layer = 1, ffwd = False, residual = False), 8609),
        "ffwd":        (dict(n_head = 4, n_layer = 1, ffwd = True, residual = False), 16961),
        "deep":        (dict(n_head = 4, n_layer = 4, ffwd = True, residual = True), 54977),
    }
    for name, (kw, expect) in presets.items():
        n = n_params(GPT(GPTConfig(vocab_size = 65, block_size = 8, n_embd = 32, **kw)))
        check(n == expect, f"the GPT for preset {name} should have {expect:,} params, got {n:,}")

    model = GPT(cfg)
    idx = torch.randint(0, 65, (2, 5))       # T=5 is shorter than block_size
    logits, loss = model(idx)
    check(logits.shape == (2, 5, 65) and loss is None, "GPT forward should return (logits (B, T, vocab), None), same as bigram")
    targets = torch.randint(0, 65, (2, 5))
    logits, loss = model(idx, targets)
    check(torch.allclose(loss, F.cross_entropy(logits.reshape(-1, 65), targets.reshape(-1))), "wrong GPT loss")

    idx2 = idx.clone()
    idx2[:, 3] = (idx2[:, 3] + 1) % 65
    check(torch.allclose(model(idx2)[0][:, :3], logits[:, :3], atol = 1e-5), "changing token 3 changed outputs at earlier positions: it is peeking at the future")

    try:
        out = model.generate(torch.zeros((1, 1), dtype = torch.long), 30)
    except IndexError:
        raise AssertionError("generate raised index out of range: once the sequence is longer than block_size, the position table runs out. "
                             "Crop to the last block_size tokens each step: idx[:, -block_size:]")
    check(out.shape == (1, 31), "after generating 30 tokens the result should be (1, 31)")


## ------------------------------------------------------------
LESSONS = {0: lesson0, 1: lesson1, 2: lesson2, 3: lesson3, 4: lesson4}


def main():
    chosen = [int(a) for a in argv if a.isdigit()] or list(LESSONS)
    print(f"checking {WHO}\n")
    for n in chosen:
        try:
            LESSONS[n]()
            print(f"lesson {n} ✅ passed")
        except NotImplementedError:
            print(f"lesson {n} ⏳ not written yet")
        except AssertionError as e:
            print(f"lesson {n} ❌ {e}")
        except Exception:
            print(f"lesson {n} 💥 error:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
