# 快速开始指南

## ⚡ 5分钟快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <项目地址>
cd Reshaped_Transformer-main

# 安装依赖（选择其一）
# 方案1: CPU版本（推荐入门）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 方案2: GPU版本（推荐训练）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install joblib scikit-learn fastdtw tensorboard numpy scipy pandas
```

### 2. 数据检查

```bash
# 检查数据是否存在
ls *dualtask.joblib

# 应该看到以下文件：
# trainX_dualtask.joblib     trainYclass_dualtask.joblib     trainYtrend_dualtask.joblib
# valX_dualtask.joblib       valYclass_dualtask.joblib       valYtrend_dualtask.joblib  
# testX_dualtask.joblib      testYclass_dualtask.joblib      testYtrend_dualtask.joblib
```

### 3. 开始训练

```bash
# 方式1: 一键训练（推荐）
python train_model.py

# 方式2: 完整训练脚本
python main.py --mode train --epochs 50 --batch_size 16
```

### 4. 监控训练

在另一个终端窗口：

```bash
# 启动TensorBoard
tensorboard --logdir=logs/

# 打开浏览器访问: http://localhost:6006
```

## 🎯 预期结果

### 训练输出示例

```
============================================================
轴承故障诊断 - 重塑的Transformer模型训练
============================================================
2024-xx-xx 10:00:00,000 INFO 模型配置:
2024-xx-xx 10:00:00,000 INFO   batch_size: 16
2024-xx-xx 10:00:00,000 INFO   learning_rate: 0.001
2024-xx-xx 10:00:00,000 INFO   epochs: 50
...
2024-xx-xx 10:00:00,000 INFO 训练集样本数: 1386
2024-xx-xx 10:00:00,000 INFO 验证集样本数: 462
2024-xx-xx 10:00:00,000 INFO 测试集样本数: 462
2024-xx-xx 10:00:00,000 INFO 模型总参数量: 1,189,263

Epoch 1/50: Train Loss=2.5000 Val Loss=2.2000 Acc=30.00%
Epoch 2/50: Train Loss=2.0000 Val Loss=1.8000 Acc=45.00%
...
Epoch 25/50: Train Loss=0.5000 Val Loss=0.4500 Acc=92.50%
```

### 性能指标

训练完成后，您应该看到类似的结果：

```
========================================
测试集评估结果:
总损失: 0.4500
分类损失: 0.2000
回归损失: 0.2500
准确率: 92.50%
精确率: 0.9200
召回率: 0.9250
F1分数: 0.9225
MSE: 0.0850
MAE: 0.2100
DTW距离: 15.50
========================================
```

## 🔧 常见问题解决

### Q1: ModuleNotFoundError: No module named 'torch'

```bash
# 解决方案：重新安装PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Q2: CUDA out of memory

```bash
# 解决方案1：减小批大小
python train_model.py  # 已设置batch_size=16

# 解决方案2：使用CPU训练
python main.py --mode train --batch_size 8
```

### Q3: 训练速度太慢

```bash
# 解决方案1：减少epoch数快速验证
python main.py --mode train --epochs 10

# 解决方案2：使用GPU（如果有）
python main.py --mode train  # 会自动检测GPU
```

### Q4: 无法找到数据文件

```bash
# 检查当前目录
pwd

# 确保在正确的项目目录
cd Reshaped_Transformer-main

# 验证数据文件
python -c "import joblib; print('数据加载正常' if len(joblib.load('trainX_dualtask.joblib')) > 0 else '数据文件有问题')"
```

## 📊 结果分析

### 1. TensorBoard可视化

训练开始后，在浏览器打开 `http://localhost:6006` 查看：

- **Loss曲线**: 观察训练和验证损失变化
- **准确率曲线**: 监控分类任务性能
- **任务权重**: 查看不确定性加权的动态变化

### 2. 日志分析

训练日志保存在 `logs/training.log`：

```bash
# 查看训练进度
tail -f logs/training.log

# 搜索最佳结果
grep "最佳" logs/training.log
```

### 3. 模型文件

训练完成后检查生成的文件：

```bash
ls logs/
# best_model.pth          - 最佳模型权重
# events.out.tfevents.*   - TensorBoard日志
# training.log            - 训练日志
```

## 🚀 下一步

### 1. 模型评估

```bash
# 仅运行评估（加载已训练模型）
python main.py --mode eval
```

### 2. 自定义参数

```bash
# 调整模型结构
python main.py --embed_dim 256 --num_heads 16 --num_layers 4

# 调整训练策略
python main.py --learning_rate 0.0005 --batch_size 32 --epochs 100
```

### 3. 实时预测

```python
# 创建预测脚本 predict.py
from models.model import MultiTaskModel
import torch
import joblib

# 加载模型
model = MultiTaskModel(...)
model.load_state_dict(torch.load('logs/best_model.pth'))
model.eval()

# 加载测试数据
test_data = joblib.load('testX_dualtask.joblib')

# 进行预测
with torch.no_grad():
    sample = torch.FloatTensor(test_data[0:1]).unsqueeze(-1)
    class_logits, reg_out = model(sample)
    
    predicted_class = torch.argmax(class_logits, dim=1).item()
    confidence = torch.softmax(class_logits, dim=1).max().item()
    
    print(f"预测故障类型: {predicted_class}")
    print(f"置信度: {confidence:.4f}")
```

## 📝 提示和技巧

### 性能优化

1. **使用更大的批大小**: 如果内存允许，增加batch_size到32或64
2. **启用混合精度**: 在支持的GPU上使用AMP加速训练
3. **数据并行**: 多GPU训练时使用DataParallel

### 模型调优

1. **学习率调度**: 尝试CosineAnnealing或StepLR
2. **正则化**: 调整dropout和weight_decay
3. **模型结构**: 根据数据特点调整window_size和embed_dim

### 实验管理

1. **版本控制**: 使用git管理代码版本
2. **实验记录**: 记录每次实验的参数和结果
3. **模型对比**: 保存多个模型版本进行对比

开始您的轴承故障诊断之旅吧！🎉 