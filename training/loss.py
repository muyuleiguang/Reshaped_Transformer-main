# 损失函数模块：定义分类和回归任务的损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义分类和回归的基础损失
classification_criterion = nn.CrossEntropyLoss(reduction='none')  # 使用'none'以便应用样本权重
regression_criterion = nn.MSELoss(reduction='none')  # 使用'none'以便应用样本权重

def multi_task_loss(class_logits, class_labels, reg_output, reg_labels, model=None):
    """
    计算分类和回归的联合损失，使用不确定性加权。
    参数:
        class_logits: 分类任务的预测logits张量 [batch_size, num_classes]
        class_labels: 分类任务的真实标签张量 [batch_size]
        reg_output: 回归任务的预测输出张量 [batch_size, num_regression]
        reg_labels: 回归任务的真实值张量 [batch_size, num_regression]
        model: 包含uncertainty_weighter的模型实例
    返回:
        total_loss: 加权总损失（标量）
        class_loss: 分类损失值（标量）
        reg_loss: 回归损失值（标量）
        precision_cls: 分类任务的精度权重
        precision_reg: 回归任务的精度权重
    """
    # 计算分类损失
    class_loss_per_sample = classification_criterion(class_logits, class_labels)
    class_loss = class_loss_per_sample.mean()
    
    # 计算回归损失
    reg_loss_per_sample = regression_criterion(reg_output, reg_labels)
    reg_loss = reg_loss_per_sample.mean()
    
    # 使用不确定性加权
    if model is not None and hasattr(model, 'uncertainty_weighter'):
        total_loss, precision_cls, precision_reg = model.uncertainty_weighter(class_loss, reg_loss)
        return total_loss, class_loss, reg_loss, precision_cls, precision_reg
    else:
        # 传统的固定权重方式
        alpha = 1.0  # 如果未提供模型或模型没有不确定性加权器，使用固定权重
        total_loss = class_loss + alpha * reg_loss
        return total_loss, class_loss, reg_loss, 1.0, alpha

def compute_loss(outputs, targets, model=None):
    """
    对外统一接口：给定模型输出和标签，返回总损失、分类损失、回归损失
    参数:
        outputs: 模型的输出元组 (class_logits, reg_output)
        targets: 目标标签元组 (class_labels, reg_labels)
        model: 模型实例，用于获取不确定性加权器
    返回:
        total_loss: 总损失
        class_loss: 分类损失
        reg_loss: 回归损失
        precision_cls: 分类任务的精度权重
        precision_reg: 回归任务的精度权重
    """
    class_logits, reg_output = outputs
    class_labels, reg_labels = targets
    return multi_task_loss(class_logits, class_labels, reg_output, reg_labels, model) 