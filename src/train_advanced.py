import json
import os
from typing import List, Dict, Any, Tuple

import optuna
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from data.preprocess import TextDataset, load_data
from models.bert_classifier import BertClassifier
from src.utils.log_utils import LoggerManager

logger = LoggerManager.get_logger(__name__)

class ModelEnsemble:
    def __init__(self):
        self.models = []  # 存储模型列表
        self.val_f1_scores = []  # 存储对应的验证F1分数
        self.best_model_index = None
        self.best_val_f1 = 0.0  # 最佳验证F1分数
        self.avg_val_f1 = 0.0  # 平均验证F1分数

    def add_model(self, model: BertClassifier, val_f1: float):
        """
        添加模型和其验证F1分数
        
        Args:
            model: 训练好的模型
            val_f1: 模型在验证集上的F1分数
        """
        self.models.append(model)
        self.val_f1_scores.append(val_f1)
        self.best_val_f1 = max(self.best_val_f1, val_f1)
        self.avg_val_f1 = sum(self.val_f1_scores) / len(self.val_f1_scores)
        self.best_model_index = self.val_f1_scores.index(self.best_val_f1)

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                predictions.append(outputs)
        # 对多个模型的预测结果进行平均
        ensemble_output = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_output

    def get_model_info(self) -> List[Dict[str, Any]]:
        """
        获取所有模型的信息
        
        Returns:
            List[Dict[str, Any]]: 包含每个模型的索引和验证F1分数的列表
        """
        return [
            {
                "model_index": i,
                "val_f1": score
            }
            for i, score in enumerate(self.val_f1_scores)
        ]


def train_fold(
        model: BertClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
) -> Tuple[BertClassifier, float]:
    """训练单个折的模型"""
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    criterion = torch.nn.CrossEntropyLoss()
    best_val_f1 = 0
    best_model_state = None

    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()

            total_loss += loss.item()

        # 验证阶段
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_f1


def cross_validation(
        texts: List[str],
        labels: List[int],
        device: torch.device,
        config: Dict[str, Any]
) -> ModelEnsemble:
    """K折交叉验证训练"""
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    ensemble = ModelEnsemble()  # 创建空的集成模型
    # 生成一个唯一值
    unique_value = str(hash(tuple(config.items())))

    logger.info(f"开始新一轮的K折交叉验证训练, 索引为: {unique_value}, 具体参数为: {config}")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        logger.info(f"索引{unique_value} 开始训练第 {fold} 折, 共 {config['n_folds']} 折")

        # 初始化模型和tokenizer
        model = BertClassifier(
            bert_model_name=config['bert_model_name'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
        model.to(device)

        # 创建数据加载器
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        dataset = TextDataset(texts, labels, model.tokenizer, max_length=config['max_length'])
        train_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler
        )
        val_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler
        )

        # 训练模型
        model, val_f1 = train_fold(model, train_loader, val_loader, device, config)
        logger.info(f"索引{unique_value} 完成训练第 {fold} 折，F1分数: {val_f1:.4f}")

        # 将模型和验证F1分数添加到集成模型中
        ensemble.add_model(model, val_f1)

    # 打印所有模型的信息
    logger.info(
        f"此轮的K折交叉验证训练完成, 索引为: {unique_value}, 平均F1分数: {ensemble.avg_val_f1:.4f}, 最佳F1分数: {ensemble.best_val_f1:.4f}, 最佳模型下标(折数): {ensemble.best_model_index}, 以下是所有模型信息:")
    for info in ensemble.get_model_info():
        logger.info(f"\t 模型下标(折数) {info['model_index']}: F1分数 = {info['val_f1']:.4f}")

    return ensemble


def objective(trial: optuna.Trial, texts: List[str], labels: List[int], device: torch.device) -> float:
    """Optuna 超参数优化目标函数， suggest_categorical时离散型超参数，suggest_float时连续型超参数"""
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.1),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        'max_length': trial.suggest_categorical('max_length', [64, 128, 256]),
        'num_epochs': 3,  # 为了快速搜索，减少训练轮数
        'n_folds': 3,  # 为了快速搜索，减少折数, 每一折会进行 num_epochs 轮训练
        'max_grad_norm': 1.0,
        'bert_model_name': 'google-bert/bert-base-chinese',
        'num_classes': 2
    }

    ensemble = cross_validation(texts, labels, device, config)
    return ensemble.best_val_f1


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info("加载数据...")
    texts, labels = load_data('../data/Data.csv')

    if not os.path.exists('outputs/best_config.json'):
        # 超参数优化
        logger.info("开始超参数优化...")
        study = optuna.create_study(direction='maximize')  # 目标函数最值化，可以用["minimize", "maximize"]
        study.optimize(lambda trial: objective(trial, texts, labels, device),
                       n_trials=10)  # 超参数尝试多少组不同的超参数组合
        logger.info("获取最佳超参数成功，具体数值为:")
        for key, value in study.best_params.items():
            logger.info(f"\t {key}: {value}")

        # 使用最佳超参数进行最终训练
        logger.info("使用最佳超参数进行最终训练...")
        best_config = {
            **study.best_params,
            'num_epochs': 10,
            'n_folds': 5,
            'max_grad_norm': 1.0,
            'bert_model_name': 'google-bert/bert-base-chinese',
            'num_classes': 2
        }
        with open('outputs/best_config.json', 'w') as f:
            json.dump(best_config, f, indent=4)
    else:
        # 读取最佳超参数
        with open('outputs/best_config.json', 'r') as f:
            best_config = json.load(f)

    # 使用最佳配置进行交叉验证训练
    final_ensemble = cross_validation(texts, labels, device, best_config)

    # 保存集成模型
    logger.info(
        f"最终训练结束，平均F1分数: {final_ensemble.avg_val_f1:.4f}, 最佳F1分数: {final_ensemble.best_val_f1:.4f}, 最佳模型下标(折数): {final_ensemble.best_model_index}, 以下是所有模型信息:")
    for info in final_ensemble.get_model_info():
        logger.info(f"\t 最终训练模型下标(折数) {info['model_index']}: F1分数 = {info['val_f1']:.4f}")
    os.makedirs('outputs/ensemble', exist_ok=True)
    for i, model in enumerate(final_ensemble.models):
        torch.save(model.state_dict(), f'outputs/ensemble/model_fold_{i}.pth')

    logger.info("模型保存成功，训练完成！")


if __name__ == '__main__':
    main()
