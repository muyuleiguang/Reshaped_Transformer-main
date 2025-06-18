# 基于重塑Transformer的轴承故障诊断系统

## 项目简介

本项目基于重塑的Transformer架构，实现了一个高效的轴承故障诊断多任务学习系统。该系统能够同时完成故障分类和趋势预测两个任务，为工业设备健康监测提供了先进的解决方案。

## 主要创新点

### 1. 多尺度卷积嵌入 (Multi-Scale Convolution Embedding)
- 采用CoMer式的多尺度卷积核（3×3, 5×5, 7×7, 9×9）
- 并行提取不同时间尺度的振动信号特征
- 有效捕获轴承故障的多频率特征

### 2. T-PE位置先验编码 (Temporal Position Encoding with Prior)
- 融合周期性先验信息的位置编码
- 可学习的位置偏置参数
- 适应轴承振动信号的周期性特征

### 3. 门控稀疏注意力 (Gated Sparse Attention)
- 基于距离衰减的注意力偏置
- 门控机制动态选择重要连接
- 局部窗口限制，提高计算效率

### 4. 不确定性加权多任务损失 (Uncertainty-Weighted Multi-task Loss)
- 自适应调整分类和回归任务权重
- 基于任务不确定性的动态平衡
- 提高多任务学习效果

## 模型架构

```
输入振动信号 (1024维)
      ↓
多尺度卷积嵌入层
      ↓
T-PE位置编码
      ↓
重塑Transformer编码器 (3层)
 - 门控稀疏注意力
 - 前馈网络
      ↓
   多任务输出头
  ↙           ↘
分类任务      回归任务
(10类故障)   (1024维趋势)
```

## 数据集说明

### 故障类型 (10类)
0. 正常状态 (de_normal)
1. 7mil内圈故障 (de_7_inner)
2. 7mil滚珠故障 (de_7_ball)
3. 7mil外圈故障 (de_7_outer)
4. 14mil内圈故障 (de_14_inner)
5. 14mil滚珠故障 (de_14_ball)
6. 14mil外圈故障 (de_14_outer)
7. 21mil内圈故障 (de_21_inner)
8. 21mil滚珠故障 (de_21_ball)
9. 21mil外圈故障 (de_21_outer)

### 数据格式
- **输入**: 1024个时间步的振动信号
- **分类标签**: 故障类型编号 (0-9)
- **回归标签**: 1024个时间步的未来趋势预测

## 文件结构

```
Reshaped_Transformer-main/
├── models/
│   └── model.py                    # 主模型定义
├── training/
│   ├── train.py                    # 训练循环
│   ├── evaluate.py                 # 评估函数
│   └── loss.py                     # 损失函数
├── metrics/
│   └── metrics.py                  # 评估指标
├── logs/                           # 训练日志和模型保存
├── dataset/                        # 数据集相关
├── bearing_dataset.py              # 数据集加载器
├── main.py                         # 主训练脚本
├── train_model.py                  # 简化训练脚本
├── test.py                         # 测试脚本
├── data_12k_10c.csv               # 原始数据
├── train*_dualtask.joblib          # 训练数据
├── val*_dualtask.joblib            # 验证数据
├── test*_dualtask.joblib           # 测试数据
└── requirement.txt                 # 依赖包列表
```

## 快速开始

### 1. 环境配置

```bash
# 安装Python依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install joblib scikit-learn fastdtw tensorboard numpy scipy pandas
```

### 2. 简单训练

```bash
# 使用简化脚本快速训练
python train_model.py
```

### 3. 完整训练（可配置参数）

```bash
# 训练模式
python main.py --mode train --epochs 100 --batch_size 32 --learning_rate 0.001

# 仅评估模式
python main.py --mode eval
```

### 4. 参数配置

```bash
python main.py \
    --epochs 100 \              # 训练轮数
    --batch_size 32 \           # 批大小
    --learning_rate 0.001 \     # 学习率
    --embed_dim 128 \           # 嵌入维度
    --num_heads 8 \             # 注意力头数
    --num_layers 3 \            # Transformer层数
    --window_size 16 \          # 注意力窗口大小
    --kernel_sizes 3,5,7,9 \    # 卷积核大小
    --period 100                # 周期先验值
```

## 训练监控

### TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir=logs/

# 浏览器访问: http://localhost:6006
```

### 监控指标

- **训练损失**: 总损失、分类损失、回归损失
- **任务权重**: 分类任务权重、回归任务权重
- **验证指标**: 准确率、精确率、召回率、F1分数
- **回归指标**: MSE、MAE、DTW距离

## 模型性能

### 分类任务
- **准确率**: 90%+
- **F1分数**: 0.9+
- **多类别平衡**: 加权平均指标

### 回归任务
- **MSE**: < 0.1
- **MAE**: < 0.2
- **DTW距离**: 时间序列相似度

### 模型规模
- **总参数量**: ~1.2M
- **训练时间**: ~50 epochs
- **推理速度**: 实时处理

## 技术特点

### 1. 高效性
- 稀疏注意力机制减少计算复杂度
- 局部窗口注意力，O(n×w) 复杂度
- 多尺度并行处理

### 2. 鲁棒性
- 不确定性加权自适应调整任务重要性
- 多任务学习提高泛化能力
- 周期性先验增强时序建模

### 3. 可解释性
- 注意力权重可视化
- 任务权重动态变化监控
- 多尺度特征分析

## 扩展应用

本模型可扩展应用于：

1. **其他机械故障诊断**
   - 齿轮箱故障检测
   - 电机故障诊断
   - 泵类设备监测

2. **信号处理任务**
   - 时间序列预测
   - 异常检测
   - 模式识别

3. **工业物联网**
   - 设备健康管理
   - 预测性维护
   - 智能监控系统

## 参考文献

1. Attention Is All You Need (Transformer原论文)
2. CoMer: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition
3. Multi-Task Learning Using Uncertainty to Weigh Losses

## 更新日志

### v1.0 (2024年)
- 初始版本发布
- 基础多任务学习功能
- 重塑Transformer架构实现

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

## 许可证

本项目遵循 MIT 许可证。 