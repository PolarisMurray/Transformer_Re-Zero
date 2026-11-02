## 逐课检查手写的 tokenizer
## 用法（在 Transformer_Re-Zero 目录下运行）:
##   python src/tokenizer/test_bpe.py                 检查 bpe_scratch.py 的所有课
##   python src/tokenizer/test_bpe.py 2 3             只检查第2、3课
##   python src/tokenizer/test_bpe.py --ref           检查参考答案（用来确认测试本身没问题）
##   python src/tokenizer/test_bpe.py --text 某文件    第5课用这个文件做全文对比（默认 data/raw/input.txt）

import sys
import time
import traceback
from pathlib import Path

import regex as re
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
argv = sys.argv[1:]

if "--ref" in argv:
    sys.path.insert(0, str(ROOT / "answers"))   # 参考答案放在 answers/（已 gitignore，只在本地）
    import bpe_reference as bpe
else:
    import bpe_scratch as bpe

DATA = ROOT / "data" / "raw" / "input.txt"
if "--text" in argv:
    DATA = Path(argv[argv.index("--text") + 1])

## 故意放进各种刁钻情况: 缩写、数字、连续空格、tab、换行、中文、emoji、行尾空格
SAMPLE = (
    "Hello world! I've got 3 apples, you'll get 42.\n\n"
    "    indented   text\twith tabs  \n"
    "中文也可以：你好，世界！ 😀 emoji\n"
    "SHOUTING and MiXeD case... ¿Qué tal?   "
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


## ------------------------------------------------------------
def lesson2():
    check(bpe.get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1},
          "get_stats([1,2,3,1,2]) 应该是 {(1,2): 2, (2,3): 1, (3,1): 1}")
    check(bpe.get_stats([7]) == {}, "只有一个元素时没有 pair，应该返回 {}")
    counts = bpe.get_stats([1, 2])
    bpe.get_stats([1, 2], counts)
    check(counts == {(1, 2): 2}, "传入 counts 时要在它上面累加，而不是新建一个 dict")

    check(bpe.merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99],
          "merge([1,2,3,1,2], (1,2), 99) 应该是 [99, 3, 99]")
    check(bpe.merge([1, 1, 1], (1, 1), 7) == [7, 1],
          "merge([1,1,1], (1,1), 7) 应该是 [7, 1]: 合并过的元素不能再参与下一次匹配")
    check(bpe.merge([5, 6], (1, 2), 9) == [5, 6], "找不到 pair 时原样返回")


## ------------------------------------------------------------
def lesson3():
    ## 维基百科上的经典例子
    tok = bpe.BasicTokenizer()
    tok.train("aaabdaaabac", 256 + 3)
    check(tok.merges == {(97, 97): 256, (256, 97): 257, (257, 98): 258},
          f"在 'aaabdaaabac' 上训练 3 次合并，merges 应该是 {{(97,97):256, (256,97):257, (257,98):258}}，"
          f"实际是 {tok.merges}（并列第一时取先出现的那一对）")
    check(tok.vocab[258] == b"aaab", f"vocab[258] 应该是 b'aaab'，实际是 {tok.vocab.get(258)!r}")
    check(tok.encode("aaabdaaabac") == [258, 100, 258, 97, 99],
          f"encode('aaabdaaabac') 应该是 [258, 100, 258, 97, 99]，实际是 {tok.encode('aaabdaaabac')}")
    check(tok.decode([258, 100, 258, 97, 99]) == "aaabdaaabac", "decode 没有还原出原文")

    ## 在混合文本上训练，往返测试
    tok = bpe.BasicTokenizer()
    tok.train(SAMPLE, 300)
    check(len(tok.vocab) == 300, f"vocab_size=300 时 vocab 应该有 300 项，实际 {len(tok.vocab)}")
    ids = tok.encode(SAMPLE)
    check(tok.decode(ids) == SAMPLE, "decode(encode(SAMPLE)) 和原文不一致")
    check(len(ids) < len(SAMPLE.encode("utf-8")), "训练过的文本，encode 之后应该比原始字节数短")
    unseen = "Tokenizers are fun 吗？"
    check(tok.decode(tok.encode(unseen)) == unseen, "没见过的文本也要能往返还原")
    check(tok.decode([0xE4]) == "�", "decode 半个汉字（单独一个 0xE4 字节）时应该返回 '�' 而不是报错")


## ------------------------------------------------------------
def lesson4():
    gpt2_pat = tiktoken.get_encoding("gpt2")._pat_str
    chunks = re.findall(bpe.GPT2_SPLIT_PATTERN, "Hello world, I've   got  it.")
    check(chunks == ["Hello", " world", ",", " I", "'ve", "  ", " got", " ", " it", "."],
          f"切分结果不对: {chunks}")
    check(re.findall(bpe.GPT2_SPLIT_PATTERN, SAMPLE) == re.findall(gpt2_pat, SAMPLE),
          "GPT2_SPLIT_PATTERN 的切分结果和 tiktoken 不一致")

    tok = bpe.RegexTokenizer()
    text = SAMPLE + "hello world " * 30
    tok.train(text, 300)
    check(len(tok.vocab) == 300, f"vocab_size=300 时 vocab 应该有 300 项，实际 {len(tok.vocab)}")
    check(tok.decode(tok.encode(text)) == text, "decode(encode(text)) 和原文不一致")
    for idx in range(256, 300):
        v = tok.vocab[idx]
        if v.strip():   # 含有非空白字符的 token，空格只允许出现在开头
            check(b" " not in v[1:],
                  f"vocab[{idx}] = {v!r} 跨过了块的边界，说明合并没有被限制在块内部")
    check(b"o w" not in tok.vocab.values(), "学出了 b'o w'，说明 'hello' 和 ' world' 被合并到一起了")


## ------------------------------------------------------------
def lesson5():
    ref = tiktoken.get_encoding("gpt2")
    tok = bpe.GPT2Tokenizer()
    check(tok.n_vocab == 50257, f"n_vocab 应该是 50257，实际 {tok.n_vocab}")

    for s in ["Hello world", "", " ", "\n\n\n", "   leading", "trailing   ", SAMPLE]:
        mine, theirs = tok.encode_ordinary(s), ref.encode_ordinary(s)
        check(mine == theirs, f"encode_ordinary({s!r}) 和 tiktoken 不一致\n  你的:     {mine}\n  tiktoken: {theirs}")
        check(tok.decode(mine) == s, f"decode 没有还原出 {s!r}")

    if not DATA.exists():
        print(f"   （跳过全文对比: 没找到 {DATA}，下载 tiny shakespeare 放到那里就会自动对比）")
        return
    text = DATA.read_text(encoding="utf-8")
    t0 = time.time()
    mine = tok.encode_ordinary(text)
    t1 = time.time()
    theirs = ref.encode_ordinary(text)
    t2 = time.time()
    if mine != theirs:
        i = next((k for k, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
        raise AssertionError(
            f"全文第 {i} 个 token 开始不一致（你的 {len(mine)} 个 vs tiktoken {len(theirs)} 个）\n"
            f"  你的:     {[tok.decode([t]) for t in mine[max(0, i - 3):i + 3]]}\n"
            f"  tiktoken: {[ref.decode([t]) for t in theirs[max(0, i - 3):i + 3]]}")
    check(tok.decode(mine) == text, "全文 decode 之后和原文不一致")
    print(f"   全文 {len(text):,} 字符 -> {len(mine):,} tokens，与 tiktoken 完全一致"
          f"（手写 {t1 - t0:.2f}s，tiktoken {t2 - t1:.2f}s）")


## ------------------------------------------------------------
def lesson6():
    ref = tiktoken.get_encoding("gpt2")
    tok = bpe.GPT2Tokenizer()
    s = "Hello<|endoftext|>World <|endoftext|>"
    for allowed in ["all", {"<|endoftext|>"}]:
        mine, theirs = tok.encode(s, allowed_special=allowed), ref.encode(s, allowed_special=allowed)
        check(mine == theirs, f"encode({s!r}, allowed_special={allowed!r}) 和 tiktoken 不一致\n"
                              f"  你的:     {mine}\n  tiktoken: {theirs}")
    check(tok.encode(s, allowed_special="all").count(50256) == 2, "<|endoftext|> 应该编码成一个 id 50256")
    check(tok.decode(tok.encode(s, allowed_special="all")) == s, "带特殊 token 的文本 decode 没有还原")
    check(tok.encode("plain text") == ref.encode("plain text"), "不传 allowed_special 时应该和 encode_ordinary 一样")


## ------------------------------------------------------------
LESSONS = {2: lesson2, 3: lesson3, 4: lesson4, 5: lesson5, 6: lesson6}


def main():
    skip = {argv.index("--text") + 1} if "--text" in argv else set()
    chosen = [int(a) for k, a in enumerate(argv) if a.isdigit() and k not in skip] or list(LESSONS)
    print(f"检查 {bpe.__name__}.py\n")
    for n in chosen:
        try:
            LESSONS[n]()
            print(f"第{n}课 ✅ 通过")
        except NotImplementedError:
            print(f"第{n}课 ⏳ 还没写")
        except AssertionError as e:
            print(f"第{n}课 ❌ {e}")
        except Exception:
            print(f"第{n}课 💥 代码报错:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
