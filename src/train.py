import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from data.preprocess import TextDataset, load_data
from models.bert_classifier import BertClassifier

# 创建保存模型和图表的目录
os.makedirs('outputs', exist_ok=True)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    # 优化器 - 增加权重衰减（L2正则化）
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # 学习率调度器
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )

    criterion = torch.nn.CrossEntropyLoss()

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 早停机制
    best_val_loss = float('inf')
    best_model_state = None
    patience = 3
    counter = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # 收集预测结果
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))

        train_losses.append(avg_loss)
        train_accs.append(train_accuracy)

        print(f'Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))

        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)

        print(f'Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # 更详细的评估指标
        if epoch == num_epochs - 1 or epoch % 2 == 0:
            report = classification_report(all_labels, all_preds, digits=4)
            print("\n分类报告:")
            print(report)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
            
            # 保存当前最佳模型
            torch.save(model.state_dict(), 'outputs/best_model.pth')
            print("保存当前最佳模型!")
        else:
            counter += 1
            print(f"验证损失未改善。当前耐心: {counter}/{patience}")
            
        if counter >= patience:
            print("早停! 验证损失已连续3轮未改善")
            break

    # 恢复最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    epochs = np.arange(1, num_epochs + 1)
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
    
    return model

def main():
    # 检测并使用MPS加速 (Apple Silicon Mac) 或 回退到CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用MPS加速训练 (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("MPS不可用, 使用CPU训练")

    # 加载数据
    print("加载数据...")
    texts, labels = load_data('data/Data.csv')

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"训练集大小：{len(train_texts)}，验证集大小：{len(val_texts)}")

    # 初始化tokenizer
    print("初始化BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 创建数据集
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=128)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 初始化模型 - 从models/bert_classifier.py文件中导入的BertClassifier类
    print("初始化BERT模型...")
    model = BertClassifier('bert-base-chinese', dropout_rate=0.3)
    model.to(device)

    # 训练模型
    print("开始训练...")
    train_model(model, train_loader, val_loader, device, num_epochs=10)

    # 保存最终模型
    torch.save(model.state_dict(), 'outputs/final_model.pth')
    print("模型训练完成！")

if __name__ == '__main__':
    main()                                   
    