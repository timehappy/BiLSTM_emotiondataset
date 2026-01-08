import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Attention(nn.Module):
    """注意力机制"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1) #创建线性层（输入维度，输出维度）
    
    def forward(self, lstm_output):
        #通过线性层和tanh激活函数以及softmax归一化得到注意力权重
        attention_weights = torch.tanh(self.attention(lstm_output)) 
        attention_weights = F.softmax(attention_weights, dim=1) #每个时间步权重和1

        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  #上下文向量
        return context_vector, attention_weights

class BiLSTMClassifier(nn.Module):
    """BiLSTM情感分类器（添加注意力机制）"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, 
                 num_layers, num_classes, dropout_rate=0.5, 
                 pretrained_embeddings=None, use_attention=True):
        super(BiLSTMClassifier, self).__init__()
        self.use_attention = use_attention
        
        # 1. 嵌入层,使用Word2Vec
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.4) #抑制过拟合

        if pretrained_embeddings is not None:
            #嵌入层使用预训练的Word2Vec词向量
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # 允许词向量  微调
        else:
            raise ValueError("未使用Word2Vec词向量！")
        
        # 2. BiLSTM层
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 3. 注意力层
        if use_attention:
            self.attention = Attention(hidden_size)
        
        # 4. 分类层
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  #批归一化，输入特征数为hidden_size*2
        self.fc = nn.Linear(hidden_size * 2, num_classes) #全连接层，将特征映射到类别数
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name: #权重参数，正交初始化
                nn.init.orthogonal_(param)
            elif 'bias' in name: #偏置参数，初始化为0
                nn.init.constant_(param, 0.0)

        #全连接层使用Xavier均匀初始化，偏置初始化为0
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, x):
        # 词嵌入
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # BiLSTM层处理词向量
        lstm_out, _ = self.lstm(embedded)
        
        if self.use_attention: #使用注意力机制
            context_vector, attention_weights = self.attention(lstm_out)
            features = context_vector
        else:
            #不使用注意力机制，则取前向LSTM的最后一个时间步和反向LSTM的第一个时间步
            features = torch.cat([lstm_out[:, -1, :self.lstm.hidden_size], 
                                 lstm_out[:, 0, self.lstm.hidden_size:]], dim=1)
        
        # 批归一化和分类
        features = self.batch_norm(features)
        features = self.dropout(features)
        logits = self.fc(features) #全连接
        
        return logits


def load_pretrained_embeddings(word2idx, embedding_dim=300, word2vec_path=None):
    """加载Word2Vec词向量"""
    
    if not word2vec_path or not os.path.exists(word2vec_path):
        raise FileNotFoundError(f"Word2Vec文件不存在: {word2vec_path}")
    
    vocab_size = len(word2idx)
    
    try:
        # 使用gensim加载
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        
        print(f"Word2Vec模型加载成功 (维度={model.vector_size})")
        
        #调整维度配置，最终以预训练维度为准
        if model.vector_size != embedding_dim:
            print(f"调整维度: {embedding_dim} -> {model.vector_size}")
            embedding_dim = model.vector_size
        
        # 创建词向量矩阵，初始化为0
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        found_vectors = [] #存储找到的词向量，用于计算均值
        
        for word, idx in word2idx.items():
            if word in model: #如果词在word2vec模型
                embedding_matrix[idx] = model[word]
                found_vectors.append(model[word])
            elif word.lower() in model:
                embedding_matrix[idx] = model[word.lower()]
                found_vectors.append(model[word.lower()])
        
        # 未匹配词使用已找到向量的均值（减少噪声，比随机初始化更稳定）
        if found_vectors:
            mean_vec = np.mean(found_vectors, axis=0)
            for word, idx in word2idx.items():
                #如果词不在模型，其小写也不在模型
                if word not in model and word.lower() not in model:
                    embedding_matrix[idx] = mean_vec
        
        # 特殊标记处理
        if '<PAD>' in word2idx: #零向量
            embedding_matrix[word2idx['<PAD>']] = np.zeros(embedding_dim)
        if '<UNK>' in word2idx:
            # 使用随机向量，避免过度平滑
            embedding_matrix[word2idx['<UNK>']] = np.random.normal(scale=0.1, size=(embedding_dim,))
        
        return torch.tensor(embedding_matrix, dtype=torch.float32)
        
    except Exception as e:
        print(f"加载失败: {e}")
        raise