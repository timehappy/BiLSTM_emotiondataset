import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import swanlab
import numpy as np
from collections import Counter

from config import Config
from utils.preprocessing import TextPreprocessor
from utils.data_loader import load_data
from models.bilstm_model import BiLSTMClassifier, load_pretrained_embeddings


def compute_class_weights(train_loader):
    """计算类别权重处理不平衡"""
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['label'].numpy())
    
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts) #统计类别数
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    
    # 逆频率加权（核心）
    for cls, count in class_counts.items():
        weight = total_samples / (count * num_classes)
        class_weights[cls] = weight
    
    # 关键：限制权重最大值为3.0（避免少数类权重过大，导致模型过度偏向）
    class_weights = torch.clamp(class_weights, max=3.0)
    
    print(f"类别分布：{dict(class_counts)}")
    print(f"修正后类别权重：{class_weights.tolist()}")
    return class_weights

def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    """单轮训练"""
    model.train() #模型设置为训练模式
    total_loss, all_preds, all_labels = 0, [], [] #初始化总损失、预测列表和真实列表标签
    
    for batch in dataloader:
        sequences, labels = batch['sequence'].to(device), batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪防止爆炸
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step() #模型参数更新
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1) #获取预测类别:最大值索引
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

def evaluate(model, dataloader, criterion, device):
    """模型评估"""
    model.eval() #评估模式
    total_loss, all_preds, all_labels = 0, [], []
    
    with torch.no_grad():
        for batch in dataloader:
            sequences, labels = batch['sequence'].to(device), batch['label'].to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels

def train_model(config):
    """主训练函数"""
    # 初始化SwanLab
    swanlab.init(
        project="BiLSTM-Emotion-Classification",
        config={
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "hidden_size": config.HIDDEN_SIZE,
            "dropout_rate": config.DROPOUT_RATE,
            "dataset": config.DATASET_NAME,
            "epochs": config.EPOCHS,
            "embedding_dim": config.EMBEDDING_DIM,
            "early_stopping_patience": config.PATIENCE,
            "use_attention": config.USE_ATTENTION,
            "use_class_weights": config.USE_CLASS_WEIGHTS
        }
    )

    
    # 数据加载
    preprocessor = TextPreprocessor(vocab_size=config.VOCAB_SIZE, max_length=config.MAX_LENGTH)
    train_loader, val_loader, test_loader = load_data(config.DATASET_NAME, config.DATA_DIR, preprocessor, config.BATCH_SIZE)
    
    print(f"数据: 训练集{len(train_loader.dataset)} | 验证集{len(val_loader.dataset)} | 测试集{len(test_loader.dataset)}")
    print(f"词汇表大小: {len(preprocessor.word2idx)}")
    
    #加载预训练词向量：加载Word2Vec模型，并构建嵌入矩阵
    pretrained_embeddings = None
    if config.USE_PRETRAINED_EMBEDDING:
        pretrained_embeddings = load_pretrained_embeddings(
            preprocessor.word2idx,
            embedding_dim=config.EMBEDDING_DIM,
            word2vec_path=config.WORD2VEC_PATH
        )
    
    # 模型初始化
    model = BiLSTMClassifier(
        vocab_size=len(preprocessor.word2idx),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained_embeddings=pretrained_embeddings,
        use_attention=config.USE_ATTENTION
    ).to(config.DEVICE)
    
    
    # 计算类别权重
    if config.USE_CLASS_WEIGHTS:    #若启用类别函数
        class_weights = compute_class_weights(train_loader)
        class_weights = class_weights.to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights) #使用带权重的交叉熵损失函数
        print(f"使用类别权重处理不平衡")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=config.SCHEDULER_MODE,
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            verbose=True,
            min_lr=1e-6
        )
    
    # 训练循环+早停
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0 #早停计数器
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(config.EPOCHS):
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config.GRAD_CLIP
        )
        
        # 验证
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, config.DEVICE
        )
        
        # 学习率调度（基于验证准确率）
        if scheduler:
            scheduler.step(val_acc)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # SwanLab记录指标
        swanlab.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "train/f1": train_f1,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/f1": val_f1,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch+1)
        
        # 打印进度
        print(f"Epoch {epoch+1:02d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        
        # 早停机制：若当前验证准确率比最佳验证准确率高出min_delta，更新最佳模型
        if val_acc > best_val_acc + config.MIN_DELTA:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'preprocessor': preprocessor,
                'config': config,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history
            }, os.path.join(config.MODEL_SAVE_PATH, f"best_model_{config.DATASET_NAME}.pth"))
            
            print(f"已保存最佳模型 (val_acc={val_acc:.4f}, val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{config.PATIENCE}")
            
            if patience_counter >= config.PATIENCE:
                print(f"\n{'='*60}")
                print(f"早停触发！连续 {config.PATIENCE} 轮验证准确率未改善")
                print(f"最佳验证准确率: {best_val_acc:.4f}")
                print(f"{'='*60}")
                break
    
    # 加载最佳模型
    best_model_path = os.path.join(config.MODEL_SAVE_PATH, f"best_model_{config.DATASET_NAME}.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型 (val_acc={checkpoint['val_acc']:.4f})")
    
    # 测试集评估
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, config.DEVICE
    )
    
    print(f"\n{'='*60}")
    print(f"保留的测试结果:")
    print(f"  测试损失: {test_loss:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  测试F1分数: {test_f1:.4f}")
    print(f"{'='*60}")
    
    # 计算各类别准确率
    from sklearn.metrics import classification_report
    if config.DATASET_NAME == 'isear':
        target_names = [config.LABEL_MAP[i] for i in range(config.NUM_CLASSES)]
    else:
        target_names = ['negative', 'positive']
    
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=target_names, digits=4))
    
    return model, preprocessor, history, test_acc

if __name__ == "__main__":
    # 参数配置
    print(f"config参数配置:")
    print(f"  数据集: {Config.DATASET_NAME}")
    print(f"  训练轮次: {Config.EPOCHS}")
    print(f"  学习率: {Config.LEARNING_RATE}")
    print(f"  批大小: {Config.BATCH_SIZE}")
    print(f"  隐藏层大小: {Config.HIDDEN_SIZE}")
    print(f"  Dropout: {Config.DROPOUT_RATE}")
    print(f"  注意力机制: {'启用' if Config.USE_ATTENTION else '禁用'}")
    
    # 检查是否使用word2vec词向量
    if Config.USE_PRETRAINED_EMBEDDING and Config.WORD2VEC_PATH:
        if not os.path.exists(Config.WORD2VEC_PATH):
            print(f"\n警告：词向量文件不存在 {Config.WORD2VEC_PATH}")
            print("将使用随机初始化词向量")
            Config.USE_PRETRAINED_EMBEDDING = False
    
    # 开始训练
    model, preprocessor, history, test_acc = train_model(Config)
    
    # 绘制训练历史
    try:
        from utils.visualization import plot_training_history
        plot_training_history(history, Config.DATASET_NAME)
    except ImportError:
        print("无法导入可视化模块，跳过绘图")