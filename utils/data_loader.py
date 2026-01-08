import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import csv

class EmotionDataset(Dataset):
    """情感数据集类:加载文本数据，并进行预处理"""
    def __init__(self, texts, labels, preprocessor, is_training=True):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.is_training = is_training
        
        # 训练集增强：每个样本生成1个增强版本，扩充数据量（样本数×2）
        if is_training:
            augmented_texts = [preprocessor.augment_text(text) for text in texts]
            self.texts += augmented_texts
            self.labels += labels  # 标签不变，仅扩充文本
            print(f"增强后训练集规模：{len(self.texts)}条")
        
        # 在增强后的text基础上构建词汇表（仅训练集）
        if is_training and not preprocessor.word2idx:
            self.preprocessor.build_vocab(self.texts)
        
        # 预转换：文本->索引
        self.sequences = [preprocessor.text_to_sequence(text) for text in self.texts]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx): #获取指定idx的样本
        return {
            'sequence': torch.tensor(self.sequences[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_sst2_data(data_dir, preprocessor):
    """加载SST-2数据集"""
    # 加载训练集
    train_df = pd.read_csv(f"{data_dir}/train.tsv", sep='\t')
    train_texts = train_df['sentence'].tolist()
    train_labels = train_df['label'].tolist()
    # 加载验证集
    dev_df = pd.read_csv(f"{data_dir}/dev.tsv", sep='\t')
    dev_texts = dev_df['sentence'].tolist()
    dev_labels = dev_df['label'].tolist()
    # 加载测试集
    test_df = pd.read_csv(f"{data_dir}/test.tsv", sep='\t')
    test_texts = test_df['sentence'].tolist()
    test_labels = test_df['label'].tolist()
    
    # 创建数据集
    train_dataset = EmotionDataset(train_texts, train_labels, preprocessor, is_training=True)
    dev_dataset = EmotionDataset(dev_texts, dev_labels, preprocessor, is_training=False)
    test_dataset = EmotionDataset(test_texts, test_labels, preprocessor, is_training=False)
    
    return train_dataset, dev_dataset, test_dataset

def load_isear_data(data_dir, preprocessor, val_size=0.2):
    """加载ISEAR数据集（七分类）"""
    
    try:
        # 尝试读取数据
        try:
            # 读取训练集
            train_df = pd.read_csv(f"{data_dir}/train.csv", sep='\t', quoting=csv.QUOTE_NONE)
            # 读取测试集
            test_df = pd.read_csv(f"{data_dir}/test.csv", sep='\t', quoting=csv.QUOTE_NONE)
        except Exception as e:
            print(f"读取失败: {e}")
            # 尝试其他分隔符
            train_df = pd.read_csv(f"{data_dir}/train.csv", on_bad_lines='skip')
            test_df = pd.read_csv(f"{data_dir}/test.csv", on_bad_lines='skip')
        
        
        # 提取文本和标签
        train_texts = train_df.iloc[:, 0].astype(str).tolist()

        # 使用Label ID列作为标签（如果存在），否则使用Label列
        if 'Label ID' in train_df.columns:
            train_labels = train_df['Label ID'].astype(int).tolist()
        elif 'Label' in train_df.columns:
            # 如果Label是字符串，需要映射到数字
            if train_df['Label'].dtype == object:
                label_mapping = {
                    'joy': 1, 'fear': 2, 'anger': 3, 'sadness': 4,
                    'disgust': 5, 'shame': 6, 'guilt': 7
                }
                #调整标签索引从0开始
                train_labels = [label_mapping[label.lower()] - 1 for label in train_df['Label']]
            else:
                train_labels = train_df['Label'].astype(int).tolist()
        else: 
            #若该列不是字符串类型，使用最后一列作为标签
            train_labels = train_df.iloc[:, -1].astype(int).tolist()
        
        test_texts = test_df.iloc[:, 0].astype(str).tolist()
        if 'Label ID' in test_df.columns:
            test_labels = test_df['Label ID'].astype(int).tolist()
        elif 'Label' in test_df.columns:
            if test_df['Label'].dtype == object:
                label_mapping = {
                    'joy': 1, 'fear': 2, 'anger': 3, 'sadness': 4,
                    'disgust': 5, 'shame': 6, 'guilt': 7
                }
                test_labels = [label_mapping[label.lower()] - 1 for label in test_df['Label']]
            else:
                test_labels = test_df['Label'].astype(int).tolist()
        else:
            test_labels = test_df.iloc[:, -1].astype(int).tolist()
        
        # 验证标签范围（ISEAR是7分类，标签应该是0-6）
        unique_labels = set(train_labels + test_labels)
        
        # 如果标签是1-7，转换为0-6
        if min(train_labels + test_labels) == 1 and max(train_labels + test_labels) == 7:
            train_labels = [label - 1 for label in train_labels]
            test_labels = [label - 1 for label in test_labels]
        
        # 检查标签范围
        final_unique_labels = set(train_labels + test_labels)
        print(f"转换后标签: {sorted(final_unique_labels)}")
        
        # 划分训练集和验证集
        print(f"划分数据集: 训练集={len(train_texts)}条, 验证集比例={val_size}")
        
        # 使用分层抽样，确保每个类别在训练集和验证集中都有代表
        if len(set(train_labels)) > 1: #从训练集中划分0.2比例作为验证集
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, 
                test_size=val_size, 
                random_state=42,
                stratify=train_labels
            )
        else:
            # 如果只有一个类别，不使用分层抽样
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, 
                test_size=val_size, 
                random_state=42
            )
        
        print(f"划分后: 训练集={len(train_texts)}条, 验证集={len(val_texts)}条, 测试集={len(test_texts)}条")
        
        # 创建数据集
        train_dataset = EmotionDataset(train_texts, train_labels, preprocessor, is_training=True)
        dev_dataset = EmotionDataset(val_texts, val_labels, preprocessor, is_training=False)
        test_dataset = EmotionDataset(test_texts, test_labels, preprocessor, is_training=False)
        
        return train_dataset, dev_dataset, test_dataset
        
    except Exception as e:
        print(f"加载ISEAR数据失败: {e}")
        # 直接抛出异常，而不是使用示例数据
        raise ValueError(f"无法加载ISEAR数据集: {e}")

def load_data(dataset_name, data_dir, preprocessor, batch_size):
    """统一的数据加载函数"""
    # 根据数据集名称选择加载函数
    if dataset_name == 'sst-2':
        train_dataset, dev_dataset, test_dataset = load_sst2_data(data_dir, preprocessor)
    elif dataset_name == 'isear':
        train_dataset, dev_dataset, test_dataset = load_isear_data(data_dir, preprocessor, val_size=0.2)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, dev_loader, test_loader