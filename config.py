# 配置文件:定义超参数和路径
import os
import torch

class Config:
    # 数据集选择
    DATASET_NAME = 'isear'
    
    # 数据集路径
    if DATASET_NAME == 'sst-2':
        DATA_DIR = 'data/sst-2/'
        NUM_CLASSES = 2
        LABEL_MAP = {0: 'negative', 1: 'positive'}
    else:  # isear
        DATA_DIR = 'data/isear/'
        NUM_CLASSES = 7
        LABEL_MAP = {
            0: 'joy',
            1: 'fear', 
            2: 'anger',
            3: 'sadness',
            4: 'disgust',
            5: 'shame',
            6: 'guilt'
        }
    
    # 训练参数（优化）[sst-2 -> isear]
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00008  # 学习率 0.0003->0.00008
    EPOCHS = 20  # 增加训练轮次
    DROPOUT_RATE = 0.55  # 降低dropout率 0.7->0.55
    HIDDEN_SIZE = 64  # 增加隐藏层大小 256->64
    NUM_LAYERS = 1 #2->1
    
    # 文本处理参数
    MAX_LENGTH = 120  # 序列长度 128->120
    VOCAB_SIZE = 12000  # 增加词汇表大小 10000->12000
    EMBEDDING_DIM = 300 
     
    # 词向量设置
    USE_PRETRAINED_EMBEDDING = True
    WORD2VEC_PATH = 'word2vec/GoogleNews-vectors-negative300.bin'
    
    # 训练过程保存
    MODEL_SAVE_PATH = 'models/saved/'
    LOG_DIR = 'logs/'
    
    # 设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 优化器参数
    WEIGHT_DECAY = 2e-3 #1e-3 ->2e-3
      # L2正则化
    GRAD_CLIP = 1.0  # 梯度裁剪
    
    # 早停参数
    PATIENCE = 8   #10->8 
    MIN_DELTA = 0.001  # 最小改善阈值 0.002->0.001
    
    # 学习率调度器参数
    USE_SCHEDULER = True #启用学习率调度器
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    SCHEDULER_MODE = 'max'  # 监控准确率
    
    # 类别权重（处理不平衡）
    USE_CLASS_WEIGHTS = True
    
    # 注意力机制
    USE_ATTENTION = True ###

    @classmethod
    def update_from_args(cls, args):
        """根据命令行参数更新配置"""
        for key, value in vars(args).items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)