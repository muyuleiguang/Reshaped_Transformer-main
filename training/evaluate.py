# evaluate.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from training.loss import compute_loss
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from metrics.metrics import (
    classification_accuracy,
    classification_precision,
    classification_recall,
    classification_f1,
    mean_squared_error_metric,
    mean_absolute_error_metric,
    dynamic_time_warping
)

def evaluate(model, data_loader):
    """
    在给定数据集上评估模型性能。
    参数：
        model: 待评估的多任务模型
        data_loader: 数据加载器（如验证集或测试集的DataLoader）
    返回：
        metrics: 包含各种评估指标的字典
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_class_preds = []
    all_class_labels = []
    all_reg_preds = []
    all_reg_labels = []
    total_loss = 0.0
    total_class_loss = 0.0
    total_reg_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for X, y_class, y_reg in data_loader:
            X = X.to(device)
            y_class = y_class.to(device)
            y_reg = y_reg.to(device)
            
            # 前向传播
            outputs = model(X)
            class_logits, reg_output = outputs
            
            # 计算损失
            loss_tuple = compute_loss(outputs, (y_class, y_reg), model)
            loss, class_loss, reg_loss, _, _ = loss_tuple
            
            # 累计损失
            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            total_class_loss += class_loss.item() * batch_size
            total_reg_loss += reg_loss.item() * batch_size
            total_samples += batch_size
            
            # 获取分类预测
            _, class_preds = torch.max(class_logits, 1)
            
            # 收集预测和标签
            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_labels.extend(y_class.cpu().numpy())
            all_reg_preds.append(reg_output.cpu().numpy())
            all_reg_labels.append(y_reg.cpu().numpy())
    
    # 将收集的数据转换为numpy数组
    all_class_preds = np.array(all_class_preds)
    all_class_labels = np.array(all_class_labels)
    all_reg_preds = np.vstack(all_reg_preds)
    all_reg_labels = np.vstack(all_reg_labels)
    
    # 计算分类指标
    accuracy = accuracy_score(all_class_labels, all_class_preds)
    precision = precision_score(all_class_labels, all_class_preds, average='weighted', zero_division=0)
    recall = recall_score(all_class_labels, all_class_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_class_labels, all_class_preds, average='weighted', zero_division=0)
    
    # 计算回归指标
    mse = mean_squared_error(all_reg_labels, all_reg_preds)
    mae = mean_absolute_error(all_reg_labels, all_reg_preds)
    
    # 计算动态时间弯曲（DTW）距离
    # 为了减少计算量，可以选择数据的子集或抽样
    sample_size = min(100, len(all_reg_preds))
    indices = np.random.choice(len(all_reg_preds), sample_size, replace=False)
    
    dtw_distances = []
    for i in indices:
        distance, _ = fastdtw(all_reg_preds[i], all_reg_labels[i], dist=euclidean)
        dtw_distances.append(distance)
    
    avg_dtw = np.mean(dtw_distances)
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    avg_class_loss = total_class_loss / total_samples
    avg_reg_loss = total_reg_loss / total_samples
    
    # 构建指标字典
    metrics = {
        'loss_total': avg_loss,
        'loss_class': avg_class_loss,
        'loss_reg': avg_reg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mse': mse,
        'mae': mae,
        'dtw': avg_dtw
    }
    
    return metrics
