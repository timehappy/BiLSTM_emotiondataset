import re
import numpy as np
import random
from collections import Counter
from typing import List, Dict

class TextPreprocessor:
    """
    文本预处理类：负责文本清洗、分词、词汇表构建和序列转换
    """ 
    def __init__(self, vocab_size: int = 10000, max_length: int = 64):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # 特殊标记
        self.PAD_TOKEN = '<PAD>'  # 填充标记
        self.UNK_TOKEN = '<UNK>'  # 未知词标记
        # 词汇表映射
        self.word2idx = {}
        self.idx2word = {}
    
    #文本处理
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # 处理缩写
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)
        # 规范化标点
        text = re.sub(r'!+', ' ! ', text)
        text = re.sub(r'\?+', ' ? ', text)
        
        text = re.sub(r"[^\w\s!?]", ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        return text.split()
    
    def build_vocab(self, texts: List[str]) -> None:
        """构建词汇表"""
        
        # counter统计词频
        word_counts = Counter()
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            word_counts.update(tokens) #词频计数更新
        
        # 按词频降序排序
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN]

        vocab_words += [word for word, _ in sorted_words[:self.vocab_size - 2]] #添加词
        
        # 创建映射
        self.word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"词汇表构建完成，大小: {len(self.word2idx)}")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """将文本转换为索引序列"""
        # 清洗文本
        cleaned = self.clean_text(text)
        # 分词
        tokens = self.tokenize(cleaned)
        
        # 获取每个词的索引，若不在词汇表，则使用unk索引
        sequence = []
        for token in tokens[:self.max_length]:
            idx = self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
            sequence.append(idx)
        
        # 填充或截断
        if len(sequence) < self.max_length:
            #序列长度小于max_length，用pad_token填充索引
            sequence = sequence + [self.word2idx[self.PAD_TOKEN]] * (self.max_length - len(sequence))
        else:
            # 截断
            sequence = sequence[:self.max_length]
        
        return sequence
    
    def augment_text(self, text: str) -> str:
        """无依赖文本增强：随机打乱词序+随机重复词（不改变语义核心）"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        
        # 长度<3的短文本不增强（避免语义失真）
        if len(tokens) < 3:
            return cleaned_text
        
        augmented_tokens = tokens.copy()
        
        # 1. 随机打乱词序（仅打乱中间部分，保留首尾词，避免语义完全混乱）
        if len(tokens) > 3:
            # 提取中间部分（去掉首尾1个词）
            middle_tokens = augmented_tokens[1:-1]
            random.shuffle(middle_tokens)
            # 重组：首尾词不变，中间打乱
            augmented_tokens = [augmented_tokens[0]] + middle_tokens + [augmented_tokens[-1]]
        
        # 2. 随机重复1个词（增强情感词权重）
        if random.random() < 0.3:  # 30%概率重复
            # 只重复情感相关词（简单判断：长度≥2的词，避免重复标点）
            repeatable_tokens = [idx for idx, token in enumerate(augmented_tokens) if len(token) >= 2]
            if repeatable_tokens:
                idx = random.choice(repeatable_tokens)
                augmented_tokens.insert(idx + 1, augmented_tokens[idx])
        
        return ' '.join(augmented_tokens)