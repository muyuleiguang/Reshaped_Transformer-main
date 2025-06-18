# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 创新点1: CoMer式多尺度卷积嵌入
class MultiScaleConvEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # 计算每个卷积路径的输出通道数，使总和为embed_dim
        out_channels = embed_dim // len(kernel_sizes)
        remainder = embed_dim % len(kernel_sizes)
        
        for i, k in enumerate(kernel_sizes):
            # 最后一个卷积可能有额外通道来达到总的embed_dim
            channels = out_channels + (remainder if i == len(kernel_sizes) - 1 else 0)
            padding = k // 2  # 保持序列长度不变
            
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, channels, kernel_size=k, padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.GELU()
                )
            )
    
    def forward(self, x):
        # x 形状: [batch, seq_len, in_channels]
        # 转换为卷积期望的形状
        x = x.permute(0, 2, 1)  # [batch, in_channels, seq_len]
        
        # 并行应用多个卷积路径
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
            
        # 在特征维度上拼接，并恢复原始维度顺序
        x = torch.cat(outputs, dim=1)  # [batch, embed_dim, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, embed_dim]
        
        return x

# 创新点2: T-PE位置先验编码
class TPEPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, period=None, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 标准的位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加周期性先验信息
        if period is not None:
            # 根据周期添加额外的编码分量
            period_term = torch.sin(position * (2 * math.pi / period))
            periodic_weight = 0.5
            pe = pe + periodic_weight * period_term
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # 位置先验偏置的可学习参数
        self.position_bias = nn.Parameter(torch.zeros(1, max_len, d_model))
        
    def forward(self, x):
        # x 形状: [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :] + self.position_bias[:, :seq_len, :]
        return self.dropout(x)

# 创新点2: 门控稀疏注意力
class GatedSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=16, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        # 查询、键、值的线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 门控网络 - 决定哪些连接是重要的
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid()
        )
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 注意力先验偏置，基于距离的衰减函数
        self.register_buffer(
            "bias", self._build_bias(window_size)
        )
        
    def _build_bias(self, window_size):
        # 创建基于距离的注意力偏置
        seq_len = window_size * 2 + 1
        bias = torch.zeros(seq_len, seq_len)
        
        # 使用高斯衰减函数作为偏置
        positions = torch.arange(0, seq_len)
        for i in range(seq_len):
            # 距离越远，偏置越小（负值）
            dist = torch.abs(positions - i)
            # 标准差调整关注范围
            sigma = window_size / 2
            bias[i] = -0.5 * (dist ** 2) / (sigma ** 2)
            
        return bias
        
    def forward(self, x):
        # x 形状: [batch, seq_len, embed_dim]
        bsz, seq_len, _ = x.shape
        
        # 线性投影
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力得分: [batch, num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加位置先验偏置
        bias = self.bias
        if seq_len > bias.size(0):
            # 如果序列长度超过预定义的偏置大小，则需要扩展偏置
            padding = seq_len - bias.size(0)
            bias = F.pad(bias, (0, padding, 0, padding), value=float('-inf'))
        else:
            # 如果序列长度小于预定义的偏置大小，则取偏置的子集
            bias = bias[:seq_len, :seq_len]
            
        attn_weights = attn_weights + bias.unsqueeze(0).unsqueeze(0)
        
        # 门控机制 - 计算查询和键之间的相关性
        # 将 q 和 k 重塑以便计算门控因子
        q_gate = q.transpose(1, 2).reshape(bsz, seq_len, -1)  # [batch, seq_len, num_heads*head_dim]
        k_gate = k.transpose(1, 2).reshape(bsz, seq_len, -1)  # [batch, seq_len, num_heads*head_dim]
        
        gate_inputs = []
        for i in range(seq_len):
            q_i = q_gate[:, i:i+1].expand(-1, seq_len, -1)  # 扩展到所有位置
            gate_input = torch.cat([q_i, k_gate], dim=-1)  # [batch, seq_len, 2*embed_dim]
            gate_inputs.append(gate_input)
            
        gate_inputs = torch.stack(gate_inputs, dim=1)  # [batch, seq_len, seq_len, 2*embed_dim]
        gates = self.gate_proj(gate_inputs).squeeze(-1)  # [batch, seq_len, seq_len]
        
        # 应用门控到注意力权重
        gates = gates.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch, num_heads, seq_len, seq_len]
        attn_weights = attn_weights * gates
        
        # 创建局部窗口掩码
        local_mask = torch.ones(seq_len, seq_len, device=attn_weights.device) * float('-inf')
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            local_mask[i, start:end] = 0
            
        attn_weights = attn_weights + local_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax 归一化
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 加权聚合值向量
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        return attn_output

# Transformer 编码器块：使用门控稀疏注意力
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, window_size=16, dropout=0.1):
        super().__init__()
        # 使用门控稀疏注意力
        self.attn = GatedSparseAttention(embed_dim, num_heads, window_size, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Pre-LN 结构: 先归一化，然后应用自注意力
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residual + x
        
        # 前馈网络部分
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

# Transformer 主干网络
class TransformerBackbone(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, ffn_dim, window_size=16, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ffn_dim, window_size, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 创新点3: 多任务损失的不确定性加权器
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 可学习的任务不确定性参数
        self.log_sigma_cls = nn.Parameter(torch.zeros(1))
        self.log_sigma_reg = nn.Parameter(torch.zeros(1))
        
    def forward(self, class_loss, reg_loss):
        # 根据不确定性动态调整权重
        precision_cls = torch.exp(-self.log_sigma_cls)
        precision_reg = torch.exp(-self.log_sigma_reg)
        
        # 加权损失
        weighted_cls_loss = precision_cls * class_loss + self.log_sigma_cls
        weighted_reg_loss = precision_reg * reg_loss + self.log_sigma_reg
        
        # 总损失
        total_loss = weighted_cls_loss + weighted_reg_loss
        return total_loss, precision_cls.item(), precision_reg.item()

# 多任务模型：包含重塑的Transformer主干和多任务输出头
class MultiTaskModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        ffn_dim=256,
        window_size=16,
        dropout=0.1,
        num_classes=10,
        num_regression=1024,
        kernel_sizes=[3, 5, 7, 9],
        period=None
    ):
        super().__init__()
        # 多尺度卷积嵌入
        self.embedding = MultiScaleConvEmbedding(in_channels, embed_dim, kernel_sizes)
        
        # T-PE位置先验编码
        self.positional_encoding = TPEPositionalEncoding(embed_dim, max_len=5000, period=period, dropout=dropout)
        
        # Transformer 主干网络
        self.backbone = TransformerBackbone(
            embed_dim, num_heads, num_layers, ffn_dim, window_size, dropout
        )
        
        # 任务特定输出层
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_regression)
        )
        
        # 不确定性权重损失函数
        self.uncertainty_weighter = UncertaintyWeightedLoss()

    def forward(self, x):
        # 输入 x 形状: [batch, seq_len, in_channels]
        
        # 多尺度卷积嵌入
        x = self.embedding(x)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 通过Transformer主干网络
        x = self.backbone(x)
        
        # 分类分支: 使用序列的[CLS]位置或全局池化特征
        # 这里使用最后一个时间步的特征作为分类特征
        class_feature = x[:, -1]
        class_out = self.classifier(class_feature)
        
        # 回归分支: 使用整个序列预测趋势
        # 这里可以根据需要从整个序列中选择某些时间步进行预测
        # 或对整个序列应用回归头
        reg_out = self.regressor(x[:, -1])
        
        return class_out, reg_out

# 测试模型功能，生成样例数据并验证模型运行正常
if __name__ == "__main__":
    print("测试模型是否能正常运行...")
    
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    
    # 生成样例数据
    batch_size = 4
    seq_len = 1024  # 序列长度
    in_channels = 1  # 单通道振动信号
    num_classes = 10  # 故障类别数
    num_regression = 512  # 趋势预测长度
    
    # 生成随机振动数据
    # 模拟振动信号：添加一些周期性成分，以模拟轴承振动
    t = torch.linspace(0, 8*np.pi, seq_len).unsqueeze(0).unsqueeze(-1)
    freq_components = [1.0, 2.0, 5.0, 10.0]  # 不同频率成分
    
    # 生成不同类别的合成振动信号
    X = torch.zeros(batch_size, seq_len, in_channels)
    y_class = torch.randint(0, num_classes, (batch_size,))
    
    for i in range(batch_size):
        signal = torch.sin(freq_components[0] * t)
        # 根据类别添加不同频率成分
        for j, freq in enumerate(freq_components[1:]):
            if y_class[i] % (j+2) == 0:  # 根据类别选择特定频率成分
                signal += 0.5 * torch.sin(freq * t + torch.rand(1) * np.pi)
        
        # 添加一些高斯噪声
        noise = 0.1 * torch.randn_like(signal)
        X[i] = signal + noise
    
    # 生成趋势预测标签 (假设趋势是未来的振动值)
    future_steps = num_regression
    y_reg = torch.zeros(batch_size, future_steps)
    
    # 生成简单的未来趋势：振幅缓慢变化的正弦波
    for i in range(batch_size):
        t_future = torch.linspace(8*np.pi, 10*np.pi, future_steps)
        trend = torch.sin(t_future) * (1.0 + 0.1 * y_class[i].float())  # 类别影响趋势幅度
        y_reg[i] = trend + 0.05 * torch.randn(future_steps)  # 添加轻微噪声
    
    print(f"输入数据形状: {X.shape}")
    print(f"分类标签形状: {y_class.shape}")
    print(f"回归标签形状: {y_reg.shape}")
    
    # 实例化模型
    model = MultiTaskModel(
        in_channels=in_channels,
        embed_dim=64,  # 降低维度以便快速测试
        num_heads=4,
        num_layers=2,
        ffn_dim=128,
        window_size=8,
        dropout=0.1,
        num_classes=num_classes,
        num_regression=future_steps,
        kernel_sizes=[3, 5, 7],
        period=100  # 假设周期约为100个时间步
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 前向传播
    try:
        class_logits, reg_out = model(X)
        print(f"分类输出形状: {class_logits.shape}")
        print(f"回归输出形状: {reg_out.shape}")
        
        # 计算简单损失以测试反向传播
        criterion_cls = nn.CrossEntropyLoss()
        criterion_reg = nn.MSELoss()
        
        loss_cls = criterion_cls(class_logits, y_class)
        loss_reg = criterion_reg(reg_out, y_reg)
        
        print(f"分类损失: {loss_cls.item():.4f}")
        print(f"回归损失: {loss_reg.item():.4f}")
        
        # 测试不确定性加权
        total_loss, precision_cls, precision_reg = model.uncertainty_weighter(loss_cls, loss_reg)
        print(f"加权总损失: {total_loss.item():.4f}")
        print(f"分类任务权重: {precision_cls:.4f}")
        print(f"回归任务权重: {precision_reg:.4f}")
        
        # 反向传播测试
        total_loss.backward()
        print("反向传播成功!")
        
        print("\n所有测试通过，模型能够正常工作! ✓")
    except Exception as e:
        print(f"测试失败: {str(e)}")
