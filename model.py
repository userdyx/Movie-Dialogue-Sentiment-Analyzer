# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import pickle
import os
import gc  # 添加垃圾回收
import argparse
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 内存优化函数
def optimize_memory():
    """优化内存使用"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=1, 
                 n_layers=1, bidirectional=False, dropout=0.2):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text的形状: [batch size, sequence length]
        
        # 嵌入层
        embedded = self.dropout(self.embedding(text))
        # embedded的形状: [batch size, sequence length, embedding dim]
        
        # LSTM层
        output, (hidden, cell) = self.lstm(embedded)
        # output的形状: [batch size, sequence length, hidden dim * num directions]
        # hidden的形状: [num layers * num directions, batch size, hidden dim]
        
        # 如果是双向LSTM，拼接两个方向的最后一个隐藏状态
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        # hidden的形状: [batch size, hidden dim * num directions]
            
        # 全连接层
        return self.fc(self.dropout(hidden))

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, 
               checkpoint_interval=1, early_stopping_patience=3):
    """训练模型"""
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 记录训练和验证的损失和准确率
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 记录最佳验证准确率和没有改善的轮次数
    best_val_acc = 0
    no_improvement = 0
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        # 使用tqdm显示进度条
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            # 解包批次数据
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # 前向传播
            optimizer.zero_grad()
            predictions = model(inputs).squeeze(1)
            
            # 计算损失
            loss = criterion(predictions, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            predicted_labels = torch.round(torch.sigmoid(predictions))
            correct = (predicted_labels == labels).float().sum()
            accuracy = correct / len(labels)
            
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
            
            # 清理内存
            del inputs, labels, predictions, loss, predicted_labels
            optimize_memory()
        
        # 计算平均损失和准确率
        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 验证模式
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                # 前向传播
                predictions = model(inputs).squeeze(1)
                
                # 计算损失
                loss = criterion(predictions, labels)
                
                # 计算准确率
                predicted_labels = torch.round(torch.sigmoid(predictions))
                correct = (predicted_labels == labels).float().sum()
                accuracy = correct / len(labels)
                
                val_loss += loss.item()
                val_acc += accuracy.item()
                
                # 清理内存
                del inputs, labels, predictions, loss, predicted_labels
                optimize_memory()
            
        # 计算平均验证损失和准确率
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Training loss: {epoch_loss:.4f}, Training accuracy: {epoch_acc:.4f}')
        print(f'  Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}')
        
        # 是否保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print('  Best model saved!')
            no_improvement = 0
        else:
            no_improvement += 1
            
        # 是否提前停止训练
        if no_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs without improvement")
            break
            
        # 定期保存检查点
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f'  Checkpoint saved: {checkpoint_path}')
            
        # 每个epoch后清理内存
        optimize_memory()
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    return model

def evaluate_model(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # 前向传播
            predictions = model(inputs).squeeze(1)
            
            # 计算损失
            loss = criterion(predictions, labels)
            test_loss += loss.item()
            
            # 保存预测结果
            predicted_labels = torch.round(torch.sigmoid(predictions))
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 清理内存
            del inputs, labels, predictions, loss, predicted_labels
            optimize_memory()
    
    # 计算平均损失
    test_loss /= len(test_loader)
    
    # 计算各项指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # 打印结果
    print(f'Test loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 score: {f1:.4f}')
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_predictions)
    
    return test_loss, accuracy, precision, recall, f1

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train LSTM Model for Sentiment Analysis')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--early_stopping', type=int, default=3, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载预处理好的数据
    try:
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_val = np.load('data/X_val.npy')
        y_val = np.load('data/y_val.npy')
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        
        # 加载词汇表
        with open('word_to_idx.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)
        
        vocab_size = len(word_to_idx)
        print(f'Vocabulary size: {vocab_size}')
        print(f'Training data size: {len(X_train)}')
        print(f'Validation data size: {len(X_val)}')
        print(f'Test data size: {len(X_test)}')
        
    except FileNotFoundError:
        print("Preprocessed data files not found. Please run data_preprocessing.py first.")
        return
    
    # 创建PyTorch数据集和加载器
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
    
    # 使用命令行参数的batch_size
    batch_size = args.batch_size
    print(f'Using batch size: {batch_size}')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 创建模型
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    output_dim = 1  # 二分类问题
    n_layers = args.n_layers
    bidirectional = args.bidirectional
    dropout = args.dropout
    
    print(f'Model parameters: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, '
          f'n_layers={n_layers}, bidirectional={bidirectional}, dropout={dropout}')
    
    model = LSTMModel(
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        output_dim, 
        n_layers, 
        bidirectional, 
        dropout
    )
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练模型
    epochs = args.epochs
    print(f'Training for {epochs} epochs...')
    
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        device, 
        epochs=epochs,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping_patience=args.early_stopping
    )
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pt'))
    
    # 在测试集上评估模型
    test_loss, accuracy, precision, recall, f1 = evaluate_model(
        model, 
        test_loader, 
        criterion, 
        device
    )
    
    # 保存评估结果
    with open('evaluation_results.txt', 'w') as f:
        f.write(f'Test loss: {test_loss:.4f}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 score: {f1:.4f}\n')

if __name__ == "__main__":
    main() 