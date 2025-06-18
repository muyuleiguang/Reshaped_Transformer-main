#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轴承故障诊断模型训练脚本
使用重塑的Transformer模型进行多任务学习
"""

import os
import torch
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
from models.model import MultiTaskModel
from training.train import train
from training.evaluate import evaluate
from bearing_dataset import BearingDataset

def setup_logging(log_dir='logs/'):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log'), mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def main():
    """主训练函数"""
    print("=" * 60)
    print("轴承故障诊断 - 重塑的Transformer模型训练")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    
    # 超参数配置
    config = {
        'batch_size': 8,           # 批大小（较小的批大小适合内存有限的环境）
        'learning_rate': 0.001,     # 学习率
        'epochs': 50,               # 训练轮数
        'embed_dim': 128,           # 嵌入维度
        'num_heads': 8,             # 注意力头数
        'num_layers': 3,            # Transformer层数
        'ffn_dim': 256,             # 前馈网络维度
        'window_size': 16,          # 局部注意力窗口大小
        'dropout': 0.1,             # Dropout比例
        'kernel_sizes': [3, 5, 7, 9],  # 多尺度卷积核大小
        'period': 100,              # 周期先验值
        'patience': 15              # 早停耐心值
    }
    
    # 打印配置信息
    logging.info("模型配置:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 创建数据加载器
    logging.info("加载数据集...")
    try:
        train_dataset = BearingDataset(data_dir='.', split='train')
        val_dataset = BearingDataset(data_dir='.', split='val')
        test_dataset = BearingDataset(data_dir='.', split='test')
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        logging.info(f"训练集样本数: {len(train_dataset)}")
        logging.info(f"验证集样本数: {len(val_dataset)}")
        logging.info(f"测试集样本数: {len(test_dataset)}")
        
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        return
    
    # 创建模型
    logging.info("创建模型...")
    try:
        model = MultiTaskModel(
            in_channels=1,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            ffn_dim=config['ffn_dim'],
            window_size=config['window_size'],
            dropout=config['dropout'],
            num_classes=10,             # 10类故障类型
            num_regression=1024,        # 回归任务输出长度
            kernel_sizes=config['kernel_sizes'],
            period=config['period']
        ).to(device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"模型总参数量: {total_params:,}")
        logging.info(f"可训练参数量: {trainable_params:,}")
        
    except Exception as e:
        logging.error(f"模型创建失败: {e}")
        return
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 开始训练
    logging.info("开始训练...")
    try:
        best_model = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            log_dir='logs/',
            epochs=config['epochs'],
            patience=config['patience']
        )
        
        logging.info("训练完成!")
        
        # 在测试集上评估
        logging.info("在测试集上评估...")
        test_metrics = evaluate(best_model, test_loader)
        
        # 打印测试结果
        logging.info("=" * 40)
        logging.info("测试集评估结果:")
        logging.info(f"总损失: {test_metrics['loss_total']:.4f}")
        logging.info(f"分类损失: {test_metrics['loss_class']:.4f}")
        logging.info(f"回归损失: {test_metrics['loss_reg']:.4f}")
        logging.info(f"准确率: {test_metrics['accuracy']*100:.2f}%")
        logging.info(f"精确率: {test_metrics['precision']:.4f}")
        logging.info(f"召回率: {test_metrics['recall']:.4f}")
        logging.info(f"F1分数: {test_metrics['f1_score']:.4f}")
        logging.info(f"MSE: {test_metrics['mse']:.4f}")
        logging.info(f"MAE: {test_metrics['mae']:.4f}")
        logging.info(f"DTW距离: {test_metrics['dtw']:.4f}")
        logging.info("=" * 40)
        
    except Exception as e:
        logging.error(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("程序执行完毕!")

if __name__ == "__main__":
    main() 