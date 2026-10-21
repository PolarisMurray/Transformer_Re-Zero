## 手写 GPT-2 BPE Tokenizer —— 练习版（你来写）
## 教程: 上层文件夹 Re0_transformer/report&教程/tokenizer_lessons.md
## 每写完一课运行一次检查:  python src/tokenizer/test_bpe.py
## 参考答案: answers/bpe_reference.py（先自己写，卡住超过 20 分钟再看）
## 每个函数里的 raise NotImplementedError 是占位符，写好之后删掉它

import regex as re   # 不是标准库的 re，它支持 \p{L} 这种 Unicode 类别


## ============================================================
## Lession: 2BPE 的两个基本操作
## ============================================================

def get_stats(ids, counts=None):
    ## id is a list
    #for i in range(len(ids) - 1):
    #    a = ids[i]
    #    b = ids[i + 1]

    #    print(a, b)

    #using zip
    if counts is None:
        counts = {}
    for x, y in zip(id, id[1:]):
        pair = (x, y)
        counts[pair] = counts.get(pair, 0) + 1

    print(counts)

    raise NotImplementedError


def merge(ids, pair, idx):
    ## 把 ids 里所有连续出现的 pair 换成新 id idx
    ## merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
    i = 0
    newids = []
    while i < len(ids) - 1:
        if (ids[i], ids[i + 1]) == pair:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    if i < len(ids):
        newids.append(ids[i])
    raise NotImplementedError


## ============================================================
## 最朴素的 BPE —— 直接在整段文本的字节上训练
## ============================================================

class BasicTokenizer:

    def __init__(self):
        self.merges = {}                                   # (int, int) -> int，训练时学到的合并规则，按学到的顺序排
        self.vocab = {i: bytes([i]) for i in range(256)}   # int -> bytes，用来 decode

    def train(self, text, vocab_size, verbose=False):
        ##1，  把 text 用 utf-8 编码成字节，再转成 int 列表 ids（每个数都在 0~255）

        ##2，  一共要合并 vocab_size - 256 次，每一次（第 i 次）:
        ##     a. 用 get_stats 统计 ids 里所有相邻对（如果已经没有 pair 了就提前结束）
        ##     b. 找出出现次数最多的那一对 pair    提示: max(stats, key=stats.get)
        ##     c. 给它分配新 id: idx = 256 + i
        ##     d. 用 merge 把 ids 里的 pair 全部换成 idx
        ##     e. 记下规则 self.merges[pair] = idx
        ##     f. 记下这个新 token 对应的字节 self.vocab[idx] = 左边的字节 + 右边的字节

        ##3，  verbose 为 True 时把每一次合并打印出来，看看它学到了什么

        raise NotImplementedError

    def decode(self, ids):
        ##1，  每个 id 在 self.vocab 里查出 bytes，全部拼起来（b"".join）

        ##2，  用 utf-8 解码成字符串。注意: 一个汉字是 3 个字节，
        ##     如果只给了其中一部分会解码失败，加上 errors="replace" 用 � 代替

        raise NotImplementedError

    def encode(self, text):
        ##1，  先把 text 变成字节的 int 列表 ids（和 train 第1步一样）

        ##2，  只要 ids 还有至少 2 个元素，就循环:
        ##     a. 统计当前 ids 里所有相邻对
        ##     b. 在这些 pair 里，找 self.merges 里编号最小的那个
        ##        （训练时先学到的规则必须先用，顺序错了结果就不对）
        ##        提示: min(stats, key=lambda p: self.merges.get(p, float("inf")))
        ##     c. 如果找到的 pair 根本不在 self.merges 里，说明没有能合并的了，退出循环
        ##     d. 否则用 merge 合并它

        ##3，  返回 ids

        raise NotImplementedError


## ============================================================
## 第4课: 先用正则切块，再在每块内部做 BPE（GPT-2 的做法）
## ============================================================

## GPT-2 原版的切分正则，来自 openai/gpt-2 仓库的 src/encoder.py（直接给出，读懂即可）
##   's|'t|'re|'ve|'m|'ll|'d   英文缩写单独成块: I've -> "I", "'ve"
##   ' ?\p{L}+'                 一个可选空格 + 一串字母: " hello"  （空格就是这样并进后一个词的）
##   ' ?\p{N}+'                 一个可选空格 + 一串数字
##   ' ?[^\s\p{L}\p{N}]+'       一个可选空格 + 一串标点符号
##   '\s+(?!\S)'                一串空白，但留下最后一个空格给后面的词
##   '\s+'                      剩下的空白
## 先在 Python 里试一下: re.findall(GPT2_SPLIT_PATTERN, "Hello world, I've   got  it.")
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class RegexTokenizer(BasicTokenizer):

    def __init__(self, pattern=GPT2_SPLIT_PATTERN):
        super().__init__()
        self.pattern = re.compile(pattern)

    def train(self, text, vocab_size, verbose=False):
        ## 和 BasicTokenizer.train 几乎一样，区别是: 先切块，合并只发生在块内部

        ##1，  用 self.pattern.findall(text) 把文本切成块的列表 chunks

        ##2，  每一块单独变成字节 int 列表，得到 ids_list（列表的列表）

        ##3，  合并 vocab_size - 256 次，每一次:
        ##     a. 新建空 dict stats，对 ids_list 里的每一块调用 get_stats(ids, stats) 累加
        ##        这样只会统计块内部的相邻对，块和块之间不会被统计
        ##     b. 后面和第3课一样: 找最多的 pair、分配 idx、记录 merges 和 vocab
        ##     c. 区别: 要对 ids_list 里的每一块都做 merge

        raise NotImplementedError

    def encode(self, text):
        ##1，  用正则把 text 切成块

        ##2，  每一块单独调用父类的 encode（super().encode(块)），结果依次拼到一个大列表里

        ##3，  返回这个大列表
        ##     decode 不用重写: 把所有 token 的字节拼起来就是原文，和切没切块无关

        raise NotImplementedError


## ============================================================
## 第5课 + 第6课: 加载 GPT-2 训练好的词表，结果和 tiktoken 一模一样
## ============================================================

class GPT2Tokenizer:
    ## 接口和 tiktoken 保持一致: encode / decode / n_vocab
    ## 所以以后 prepare.py 里要切换，只需要换掉创建 enc 的那一行

    def __init__(self):
        import tiktoken
        ## 只借用 tiktoken 里存好的 GPT-2 词表数据，切分和合并的算法全部自己写
        ## ranks: bytes -> id，一共 50256 个。id 越小，说明这个 token 在训练时越早被合并出来
        ## 先在 Python 里看看: list(self.ranks.items())[:10] 和 list(self.ranks.items())[250:270]
        self.ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks
        self.special_tokens = {"<|endoftext|>": 50256}
        self.pattern = re.compile(GPT2_SPLIT_PATTERN)

        self.decoder = {i: b for b, i in self.ranks.items()}
        for s, i in self.special_tokens.items():
            self.decoder[i] = s.encode("utf-8")
        self.n_vocab = len(self.decoder)                   # 50256 + 1 = 50257
        self.cache = {}                                    # 同一个词会出现上千次，算一次就记住

    def _bpe(self, chunk):
        ## 第5课: 对一个块（字符串）做 BPE，返回 id 列表
        ## 和第3课最大的区别: GPT-2 的 id 0 是 b'!'，不是 b'\x00'，字节值不能直接当 id。
        ## 所以这里全程用 bytes 操作，最后一步再查 ranks 变成 id

        ##1，  如果 chunk 已经在 self.cache 里，直接返回

        ##2，  把 chunk 编码成字节，拆成单字节 bytes 的列表 parts
        ##     "hi" -> [b'h', b'i']        提示: bytes([b]) 把一个 int 变回单字节 bytes

        ##3，  只要 parts 还有至少 2 个元素，就循环:
        ##     a. 遍历每一对相邻的 parts[i] + parts[i+1]，拼起来去 self.ranks 里查
        ##     b. 记下查得到的里面 rank 最小的位置 best_i（rank 最小 = 训练时最早学到）
        ##     c. 一个都查不到就退出循环
        ##     d. 否则把 parts[best_i] 和 parts[best_i+1] 合成一个元素

        ##4，  每个 part 查 self.ranks 得到 id，存进 self.cache 再返回

        raise NotImplementedError

    def encode_ordinary(self, text):
        ## 第5课: 正则切块，每块调用 self._bpe，结果拼起来

        raise NotImplementedError

    def decode(self, ids):
        ## 第5课: 和 BasicTokenizer.decode 一样，只是查的是 self.decoder

        raise NotImplementedError

    def encode(self, text, allowed_special=()):
        ## 第6课: 处理特殊 token "<|endoftext|>"（GPT-2 用它分隔两篇文档）
        ## allowed_special="all" 或者 {"<|endoftext|>"} 时，它要编码成一个 id 50256
        ## 否则（默认）就和 encode_ordinary 一样

        ##1，  根据 allowed_special 算出这次允许的特殊 token: dict allowed（字符串 -> id）

        ##2，  allowed 为空，直接 return self.encode_ordinary(text)

        ##3，  拼出一个正则，用括号把所有特殊 token 包起来:  "(" + "|".join(re.escape(s) ...) + ")"
        ##     用 re.split 切开 text。因为有括号，切出来的结果里会保留特殊 token 本身
        ##     re.split("(<\|endoftext\|>)", "a<|endoftext|>b") -> ["a", "<|endoftext|>", "b"]

        ##4，  遍历切出来的每一段: 是特殊 token 就放它的 id，否则用 encode_ordinary

        raise NotImplementedError
