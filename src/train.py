import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data.preprocess import TextDataset, load_data
from models.bert_classifier import BertClassifier
from src.utils.log_utils import LoggerManager

logger = LoggerManager.get_logger()

# 创建保存模型和图表的目录
os.makedirs('outputs', exist_ok=True)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    # AdamW优化器 - 增加权重衰减（L2正则化）
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,                # 学习率，决定每次参数更新的步长。 太大：训练不稳定，可能无法收敛；太小：训练速度慢，可能陷入局部最优。建议范围：1e-5 到 5e-5
        weight_decay=0.01,      # L2正则化系数，控制模型复杂度，防止过拟合。太大：模型欠拟合；太小：模型过拟合。建议范围：0.01 到 0.1
        betas=(0.9, 0.999),     # 动量参数，帮助优化器跳出局部最优。beta1：控制一阶矩估计；beta2：控制二阶矩估计。
        eps=1e-8                # 数值稳定性参数
    )

    # 学习率调度器
    total_steps = len(train_loader) * num_epochs # 总训练步数 = 每个 epoch 中的 batch数 * 训练轮数
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps, # 前10%的步数用于预热，帮助模型在开始时稳定训练。太大：浪费训练时间；太小：模型不稳定。建议范围：总步数的 5%-10%
        num_training_steps=total_steps
    )

    # 交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 记录训练过程
    train_losses = [] # 每一个元素 记录一轮epoch的 batch平均loss
    train_accs = [] # 每一个元素 记录一轮epoch的 所有case的准确率
    val_losses = [] # 每一个元素 记录一轮epoch的 batch平均loss
    val_accs = [] # 每一个元素 记录一轮epoch的 所有case的准确率

    # 早停机制
    best_val_loss = float('inf') #初始值为正无穷大
    best_model_state = None # 最佳模型状态
    patience = 3 # 早停轮数。决定模型在验证集性能不再提升时继续训练的轮数，太大：训练时间过长；太小：可能过早停止。建议范围：3-10
    counter = 0 # 计数器

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0 # 每一个batch的总损失
        all_preds = [] # 预测结果
        all_labels = [] # 真实标签

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            # 将数据移动到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播阶段
            optimizer.zero_grad() # 清空之前的梯度
            outputs = model(input_ids, attention_mask) # 自动调用模型的 forward 方法进行前向传播
            loss = criterion(outputs, labels) # 计算预测结果和真实标签之间的损失

            # 反向传播阶段
            loss.backward() # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸。太大：可能梯度爆炸；太小：可能梯度消失。建议范围：0.5-1.0
            optimizer.step() # 根据梯度更新模型参数
            scheduler.step() # 更新学习率

            # 损失统计阶段
            total_loss += loss.item() # 累加每个batch的损失

            # 收集预测结果阶段
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果张量
            all_preds.extend(predicted.cpu().numpy()) # 将预测结果添加到列表中， cpu().numpy()作用将张量转换为numpy数组
            all_labels.extend(labels.cpu().numpy()) # 将真实标签添加到列表中

        # 计算在这一个epoch中平均损失和准确率
        avg_loss = total_loss / len(train_loader) # 计算在这一个epoch中，所有batch的平均损失
        train_accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels)) # 计算在这一个epoch中，每个所有case的准确率
        train_losses.append(avg_loss)
        train_accs.append(train_accuracy)
        logger.info(f'Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

        # 验证阶段
        model.eval()
        val_loss = 0 # 每一个batch的总损失
        all_preds = [] # 预测结果
        all_labels = [] # 真实标签
        
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移动到设备上
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # 前向传播阶段，不需要反向传播
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                # 损失统计阶段
                val_loss += loss.item()

                # 收集预测结果阶段
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算在这一个epoch中平均损失和准确率
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)
        logger.info(f'Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # 更详细的评估指标
        report = classification_report(all_labels, all_preds, digits=4)
        logger.info("\n分类报告:")
        logger.info(report)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            # 如果验证结果更好，则保存当前的最佳模型
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
            torch.save(model.state_dict(), 'outputs/best_model.pth')
            logger.info("保存当前最佳模型!")
        else:
            # 如果验证结果更差，则累加counter，当counter次数达到阈值，跳出epoch循环
            counter += 1
            logger.info(f"验证损失未改善。当前耐心: {counter}/{patience}")
            
        if counter >= patience:
            logger.info("早停! 验证损失已连续3轮未改善")
            break

    # 加载最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    # 绘制损失曲线和准确率曲线
    plot_curves(train_losses, val_losses, train_accs, val_accs)
    
    return model

def plot_curves(train_losses, val_losses, train_accs, val_accs):
    """
    绘制训练和验证的损失曲线和准确率曲线
    
    参数:
    train_losses: 训练损失列表
    val_losses: 验证损失列表
    train_accs: 训练准确率列表
    val_accs: 验证准确率列表
    """
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    epochs = np.arange(1, len(train_losses) + 1)  # 使用实际训练的epoch数
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/loss_curve.png')

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/accuracy_curve.png')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info("加载数据...")
    texts, labels = load_data('../data/Data.csv')

    # 划分训练集和验证集，train_texts是训练集的输入，train_labels是训练集的标签，val_texts是验证集的输入，val_labels是验证集的标签
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info(f"训练集大小：{len(train_texts)}，验证集大小：{len(val_texts)}")

    # 初始化模型 和 tokenizer
    logger.info("初始化BERT模型 和 tokenizer...")
    model = BertClassifier('google-bert/bert-base-chinese', dropout_rate=0.3)
    model.to(device)

    # 创建数据集
    # max_length： 句子的截断长度。过小：丢失下文，速度快；过大：保留完整信息，但速度慢
    train_dataset = TextDataset(train_texts, train_labels, model.tokenizer, max_length=128)
    val_dataset = TextDataset(val_texts, val_labels, model.tokenizer, max_length=128)

    # 创建数据加载器，
    # Epoch：所有训练样本都已输入到模型中，称为一个Epoch
    # iteration：一批样本输入到模型中，称之为一个iteration
    # batch_size：批大小，决定一个Epoch有多少个iteration。太大：内存消耗大，可能影响泛化能力；太小：训练不稳定，速度慢。建议范围：16-64
    # 样本总数：80，batch_size=8时， 那么 1 Epoch = 10 iteration
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)


    # 训练模型
    logger.info("开始训练...")
    # num_epochs训练轮数：太大：可能过拟合；太小：可能欠拟合
    train_model(model, train_loader, val_loader, device, num_epochs=10)

    # 保存最终模型
    torch.save(model.state_dict(), 'outputs/final_model.pth')
    logger.info("模型训练完成！")

if __name__ == '__main__':
    main()                                   
    