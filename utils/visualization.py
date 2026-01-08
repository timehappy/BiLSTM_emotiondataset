#可视化
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import os

def plot_training_history(history, dataset_name):
    """绘制训练历史（补全F1曲线）"""
    # 创建results目录
    os.makedirs('results', exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1分数曲线（补全：检查history是否有train_f1/val_f1）
    if 'train_f1' in history and 'val_f1' in history:
        axes[2].plot(epochs, history['train_f1'], 'b-', label='Train F1')
        axes[2].plot(epochs, history['val_f1'], 'r-', label='Val F1')
        axes[2].set_title('Training and Validation F1 Score')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('F1 Score')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.suptitle(f'Training History - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/training_history_{dataset_name}.png')
    plt.show()

def plot_embedding_visualization(embeddings, labels, label_names, title="Word Embeddings Visualization"):
    """使用t-SNE可视化词嵌入"""
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    # 为每个类别使用不同的颜色
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=label_names[label], alpha=0.6)
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/embedding_visualization.png')
    plt.show()

def plot_class_distribution(labels, label_names, dataset_name):
    """绘制类别分布"""
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([label_names[i] for i in unique], counts)
    plt.title(f'Class Distribution - {dataset_name}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 在柱状图上显示数量
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'results/class_distribution_{dataset_name}.png')
    plt.show()

def plot_attention_heatmap(attention_weights, tokens, title="Attention Heatmap"):
    """绘制注意力热图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(attention_weights, cmap='viridis')
    
    # 设置坐标轴
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticklabels(tokens)
    
    # 添加颜色条
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('results/attention_heatmap.png')
    plt.show()