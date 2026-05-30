import sentencepiece as spm
import os

# 确保你在正确的目录下
print("开始训练英文 BPE 模型...")
spm.SentencePieceTrainer.train(
    input='en2cn/train_en.txt',      # 输入文件路径
    model_prefix='en_bpe',           # 输出模型的前缀名
    vocab_size=8000,                 # 词表大小
    character_coverage=1.0,          # 字符覆盖率
    model_type='bpe'                 # 模型类型
)

print("英文模型训练完成！\n开始训练中文 Unigram 模型...")
spm.SentencePieceTrainer.train(
    input='en2cn/train_zh.txt',      # 输入文件路径
    model_prefix='zh_bpe',           # 输出模型的前缀名
    vocab_size=8000,                 # 词表大小
    character_coverage=1.0,          # 字符覆盖率
    model_type='unigram'             # 中文通常建议用 unigram，或者 bpe 也可以
)

print("全部完成！你现在应该能看到 en_bpe.model 和 zh_bpe.model 了。")