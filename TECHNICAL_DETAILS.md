# 技术实现详细说明

## 1. 模型架构详解

### 1.1 多尺度卷积嵌入层

```python
class MultiScaleConvEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_sizes=[3, 5, 7, 9]):
```

**设计思路：**
- 使用不同尺寸的卷积核并行提取特征
- 卷积核大小：3, 5, 7, 9 对应不同的时间感受野
- 输出通道数平均分配，确保总维度为 embed_dim

**技术细节：**
- 使用 `padding = k // 2` 保持序列长度不变
- BatchNorm1d + GELU 激活函数
- 最后拼接所有尺度的特征

### 1.2 T-PE位置先验编码

```python
class TPEPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, period=None, dropout=0.1):
```

**创新点：**
- 标准正弦/余弦位置编码 + 周期性先验
- 可学习的位置偏置参数 `position_bias`
- 适配轴承信号的周期性特征

**数学公式：**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# 周期性增强
if period is not None:
    PE += 0.5 * sin(pos * 2π / period)
```

### 1.3 门控稀疏注意力机制

```python
class GatedSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=16, dropout=0.1):
```

**核心特性：**

1. **距离衰减偏置**
   ```python
   # 高斯衰减函数
   dist = torch.abs(positions - i)
   sigma = window_size / 2
   bias[i] = -0.5 * (dist ** 2) / (sigma ** 2)
   ```

2. **门控机制**
   ```python
   # 计算门控因子
   gate_input = torch.cat([q_i, k_gate], dim=-1)
   gates = self.gate_proj(gate_input).squeeze(-1)
   attn_weights = attn_weights * gates
   ```

3. **局部窗口**
   - 只计算窗口内的注意力
   - 复杂度从 O(n²) 降到 O(n×w)

### 1.4 不确定性加权损失

```python
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_cls = nn.Parameter(torch.zeros(1))
        self.log_sigma_reg = nn.Parameter(torch.zeros(1))
```

**理论基础：**
- 基于贝叶斯不确定性的多任务学习
- 自动调整任务权重，避免手动调参

**损失计算：**
```python
precision_cls = torch.exp(-self.log_sigma_cls)
precision_reg = torch.exp(-self.log_sigma_reg)

weighted_cls_loss = precision_cls * class_loss + self.log_sigma_cls
weighted_reg_loss = precision_reg * reg_loss + self.log_sigma_reg
```

## 2. 数据预处理流程

### 2.1 信号预处理

```python
def bandpass_filter(data, lowcut=200, highcut=5900, fs=12000, order=4):
    """带通滤波"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def normalize(data):
    """0–1 归一化"""
    return (data - data.min()) / (data.max() - data.min() + 1e-8)
```

### 2.2 样本生成策略

```python
def create_dualtask_samples(signal, label, Lhist=1024, Lpred=1024, stride=512):
    """
    滑窗生成训练样本
    - Lhist: 历史窗口长度 (输入)
    - Lpred: 预测窗口长度 (标签)
    - stride: 滑窗步长
    """
```

**数据增强：**
- 滑窗采样增加样本数量
- 步长512提供50%重叠
- 保持时序连续性

## 3. 训练策略

### 3.1 早停机制

```python
if val_total_loss < best_val_loss:
    best_val_loss = val_total_loss
    best_model_wts = model.state_dict().copy()
    epochs_no_improve = 0
else:
    epochs_no_improve += 1
    if epochs_no_improve >= patience:
        break
```

### 3.2 学习率调度

推荐使用 CosineAnnealingLR 或 ReduceLROnPlateau：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
```

### 3.3 模型保存策略

```python
# 保存最佳模型
torch.save(model.state_dict(), "best_model.pth")

# 保存检查点
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': best_val_loss,
}
torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")
```

## 4. 评估指标详解

### 4.1 分类指标

```python
# 多类别分类指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
```

### 4.2 回归指标

```python
# 回归误差指标
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

# 动态时间弯曲距离
from fastdtw import fastdtw
distance, _ = fastdtw(pred_sequence, true_sequence, dist=euclidean)
```

### 4.3 DTW距离说明

DTW (Dynamic Time Warping) 适合评估时间序列预测：
- 允许时间轴上的弹性匹配
- 对相位偏移和速度变化鲁棒
- 比MSE更适合时序相似度评估

## 5. 超参数调优指南

### 5.1 模型结构参数

| 参数 | 默认值 | 调优范围 | 影响 |
|------|--------|----------|------|
| embed_dim | 128 | 64-512 | 模型容量，计算复杂度 |
| num_heads | 8 | 4-16 | 注意力多样性 |
| num_layers | 3 | 2-6 | 模型深度，表达能力 |
| window_size | 16 | 8-64 | 局部感受野大小 |
| dropout | 0.1 | 0.05-0.3 | 正则化强度 |

### 5.2 训练参数

| 参数 | 默认值 | 调优范围 | 说明 |
|------|--------|----------|------|
| batch_size | 16 | 8-64 | 内存允许的最大值 |
| learning_rate | 1e-3 | 1e-4 to 1e-2 | Adam优化器 |
| weight_decay | 1e-4 | 1e-5 to 1e-3 | L2正则化 |

### 5.3 调优策略

1. **粗调阶段**：
   - 先固定其他参数，调整学习率
   - 使用较小的模型快速实验
   - 观察损失曲线和收敛趋势

2. **精调阶段**：
   - 使用验证集确定最优结构参数
   - 网格搜索或贝叶斯优化
   - 考虑计算资源约束

## 6. 部署和推理

### 6.1 模型转换

```python
# 转换为ONNX格式
dummy_input = torch.randn(1, 1024, 1)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=['input'], output_names=['classification', 'regression']
)
```

### 6.2 实时推理

```python
class RealTimePredictor:
    def __init__(self, model_path):
        self.model = MultiTaskModel(...)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, signal):
        with torch.no_grad():
            signal = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(-1)
            class_logits, reg_out = self.model(signal)
            
            # 故障分类
            fault_class = torch.argmax(class_logits, dim=1).item()
            confidence = torch.softmax(class_logits, dim=1).max().item()
            
            # 趋势预测
            trend_prediction = reg_out.squeeze().numpy()
            
            return {
                'fault_class': fault_class,
                'confidence': confidence,
                'trend': trend_prediction
            }
```

## 7. 故障排除

### 7.1 常见问题

1. **内存不足**
   - 减小batch_size
   - 减少embed_dim
   - 使用梯度累积

2. **训练不收敛**
   - 检查学习率设置
   - 验证数据预处理
   - 增加正则化

3. **过拟合**
   - 增大dropout
   - 使用早停
   - 数据增强

### 7.2 调试技巧

```python
# 1. 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# 2. 监控激活值
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__}: {output.norm()}")

# 注册钩子函数
model.register_forward_hook(hook_fn)
```

## 8. 扩展开发

### 8.1 添加新的注意力机制

```python
class CustomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # 实现自定义注意力
        
    def forward(self, x):
        # 注意力计算逻辑
        return attention_output
```

### 8.2 集成其他模型

```python
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_backbone = TransformerBackbone(...)
        self.cnn_branch = ConvolutionalBranch(...)
        self.fusion_layer = FusionLayer(...)
        
    def forward(self, x):
        transformer_features = self.transformer_backbone(x)
        cnn_features = self.cnn_branch(x)
        fused_features = self.fusion_layer(transformer_features, cnn_features)
        return self.output_head(fused_features)
```

这个技术文档提供了模型的深层实现细节，可以帮助理解和进一步开发这个轴承故障诊断系统。 