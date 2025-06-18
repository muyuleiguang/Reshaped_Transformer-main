# main.py
import os
import torch
import torch.optim as optim 
import argparse
import logging
from torch.utils.data import DataLoader
from models.model import MultiTaskModel
from training.train import train
from training.evaluate import evaluate
from bearing_dataset import BearingDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task Bearing Fault Diagnosis")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="运行模式：'train' 执行训练流程；'eval' 仅加载最优模型并评估")
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--data_dir', type=str, default='.', help='.joblib 数据文件所在目录')
    parser.add_argument('--log_dir', type=str, default='logs/', help='日志及模型存储目录')
    
    # 模型相关参数
    parser.add_argument('--embed_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数量')
    parser.add_argument('--num_layers', type=int, default=3, help='Transformer层数')
    parser.add_argument('--ffn_dim', type=int, default=256, help='前馈网络维度')
    parser.add_argument('--window_size', type=int, default=16, help='局部注意力窗口大小')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比例')
    parser.add_argument('--kernel_sizes', type=str, default='3,5,7,9', help='多尺度卷积核大小，逗号分隔')
    parser.add_argument('--period', type=int, default=None, help='周期先验值，如果知道信号周期')
    
    args = parser.parse_args()

    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'main.log'), mode='w'),
            logging.StreamHandler()
        ]
    )

    # 解析卷积核大小列表
    kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    
    # 准备 DataLoader：
    train_loader = DataLoader(
        BearingDataset(data_dir=args.data_dir, split='train'),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        BearingDataset(data_dir=args.data_dir, split='val'),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        BearingDataset(data_dir=args.data_dir, split='test'),
        batch_size=args.batch_size, shuffle=False
    )
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化模型
    model = MultiTaskModel(
        in_channels=1,  # 假设输入是单通道振动信号
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        window_size=args.window_size,
        dropout=args.dropout,
        num_classes=10,  # 根据实际故障类别数调整
        num_regression=1024,  # 回归任务输出长度
        kernel_sizes=kernel_sizes,
        period=args.period
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型总参数量: {total_params:,}")
    logging.info(f"可训练参数量: {trainable_params:,}")

    if args.mode == 'train':
        # ----------------- 训练 + 测试评估 -----------------
        logging.info("开始训练 …")
        logging.info(f"模型配置: embed_dim={args.embed_dim}, heads={args.num_heads}, layers={args.num_layers}, "
                    f"window_size={args.window_size}, kernel_sizes={kernel_sizes}")
        
        best_model = train(
            model, train_loader, val_loader,
            optimizer=optim.Adam(model.parameters(), lr=args.learning_rate),
            # TensorBoard 日志目录
            log_dir=args.log_dir,
            epochs=args.epochs,
            patience=20
        )
        logging.info("训练完毕，开始测试集评估 …")
        test_metrics = evaluate(best_model, test_loader)
        logging.info(f"Test结果：{test_metrics}")
        # 最优模型已在 train() 中存储于 logs/best_model.pth
    else:
        # ----------------- 仅评估 -----------------
        ckpt = os.path.join(args.log_dir, "best_model.pth")
        logging.info(f"加载最优模型权重：{ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        logging.info("开始在测试集上评估 …")
        test_metrics = evaluate(model, test_loader)
        logging.info(f"Test结果：{test_metrics}")

    logging.info("流程结束。")
