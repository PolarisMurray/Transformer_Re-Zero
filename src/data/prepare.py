import tiktoken


## load GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

print("Vocabulary size:", enc.n_vocab)
assert enc.n_vocab == 50257

## Disposable scripts: Original -> Token
## 一次性的


##1，  读取文件，把整个txt读成一个字符串，打印字符数

##2，  拿tokenizer，用tiktoken，取“GPT2"，打印它的词表大小， 确认是50257

##3，  做往返测试（比较重要）：拿一句话先encode，然后再decode，确认结果和原文完全一致
##     然后打印这句话的token_id, 看看BPE怎么切分的，比如空格是否被并进了后面那个词里面


##4，  编码全文：把整个文本encode成为一个token_id列表，打印token总数（应该在33.8万左右）

##5， 切分训练集和验证集：前90%作为训练，后10%作为验证(Validation)

##6， 转换类型并保存，把两个部分，分别转换成numpy数组，数据类型用uint16，存成
## data/processed/train.bin 和 val.bin （uint16 = 65535 > 50257，空间是int64 的 1/4）


## from pathlib import Path

## text = Path("requirements.txt").read_text(encoding="utf-8")

## print(len(text))
## more professional way

## part_1


with open("filename.txt", "r", encoding = "utf-8") as f:
    text = f.read()

print(len(text))


## part_2


