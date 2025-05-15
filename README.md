# 基于BERT的中文评论情感分析

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0+-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.5.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

基于PyTorch和BERT实现的中文评论情感分析系统，可以准确识别文本评论的情感极性（正面/负面）。本项目利用预训练的BERT模型结合优化策略，实现了高效、高精度的文本情感分类。

## 目录

- [项目特性](#项目特性)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [模型架构](#模型架构)
- [数据集说明](#数据集说明)
- [训练过程](#训练过程)
- [模型评估](#模型评估)
- [预测与使用](#预测与使用)
- [未来工作](#未来工作)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 项目特性

- 💡 使用预训练的BERT-base-Chinese模型实现高精度情感分类
- 🚀 采用多种优化策略防止过拟合，提高模型泛化能力
- 📊 详细的训练过程可视化与评估报告
- ⚡ 在Apple Silicon Mac上支持MPS加速训练
- 📝 支持自定义数据集训练
- 📱 简单易用的预测接口，可集成到实际应用中

## 项目结构

```
text_sentiment_classification/
├── README.md               # 项目说明文档
├── requirements.txt        # 项目依赖
├── .gitignore              # Git忽略文件
├── data/                   # 数据目录
│   ├── Data.csv            # 主数据集
│   ├── train_data.csv      # 训练数据
│   ├── test_data.csv       # 测试数据
│   └── dev_data.csv        # 验证数据
├── src/                    # 源代码
│   ├── data/               # 数据处理模块
│   │   ├── __init__.py
│   │   └── preprocess.py   # 数据预处理
│   ├── models/             # 模型定义
│   │   ├── __init__.py
│   │   └── bert_classifier.py  # BERT分类器
│   ├── utils/              # 工具模块
│   └── train.py            # 训练脚本
├── notebooks/              # Jupyter笔记本
└── outputs/                # 输出目录
    ├── best_model.pth      # 最佳模型权重
    ├── final_model.pth     # 最终模型权重
    ├── loss_curve.png      # 损失曲线图
    └── accuracy_curve.png  # 准确率曲线图
```

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- Transformers 4.5.0+
- pandas 1.2.0+
- numpy 1.19.0+
- scikit-learn 0.24.0+
- tqdm 4.60.0+
- matplotlib 3.3.0+
- seaborn 0.11.0+

## 快速开始

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/Altocumuli/text_sentiment_classification.git
cd text_sentiment_classification
```

2. 创建虚拟环境(可选)：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python src/train.py
```

训练过程会自动:
- 加载并预处理数据
- 训练BERT情感分类模型
- 保存最佳模型和训练曲线
- 显示训练进度和评估结果

### GPU加速

本项目支持在Apple Silicon Mac上使用MPS加速训练，在其他平台上会自动检测并使用GPU（如有）。

## 模型架构

本项目使用预训练的中文BERT模型（bert-base-chinese）作为基础，通过迁移学习实现情感分类：

1. **BERT编码器层**：将中文文本转换为768维的向量表示
2. **Dropout层**：防止过拟合，丢弃率设为0.3
3. **分类层**：全连接层，将768维向量映射到2个类别（正面/负面）

### 优化策略

- **早停机制**：验证损失连续3轮未改善时停止训练
- **学习率调度**：线性预热后递减的学习率调度
- **权重衰减**：L2正则化减少过拟合
- **梯度裁剪**：防止梯度爆炸
- **分层学习率**：BERT层使用较小学习率，分类层使用较大学习率

## 数据集说明

数据集包含中文商品/服务评论及其情感标签：
- 标签1：正面情感
- 标签0：负面情感

数据集统计信息：
- 总样本数：约12000条评论
- 训练集占比：80%
- 验证集占比：20%
- 最大序列长度：128个token

## 训练过程

- **优化器**：AdamW，学习率2e-5，权重衰减0.01
- **批次大小**：训练16，验证32
- **训练轮数**：最多10轮，通常3-5轮会达到最佳效果
- **评估指标**：准确率、精确率、召回率、F1分数

## 模型评估

模型在验证集上的表现：
- 准确率：约92%
- F1分数：约91%
- 训练损失和验证损失曲线表明模型收敛良好，没有严重过拟合

详细的评估结果和可视化请查看`outputs/`目录中的图表。

## 预测与使用

您可以使用训练好的模型进行情感预测。以下是简单的使用示例：

```python
from transformers import BertTokenizer
import torch
from src.models.bert_classifier import BertClassifier

# 加载模型
model = BertClassifier('bert-base-chinese')
model.load_state_dict(torch.load('outputs/best_model.pth'))
model.eval()

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 预测函数
def predict_sentiment(text):
    # 预处理文本
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 预测
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
        _, predicted = torch.max(outputs, 1)
    
    return "正面评价" if predicted.item() == 1 else "负面评价"

# 示例
text = "服务态度很好，送餐很快，味道也不错！"
result = predict_sentiment(text)
print(f"评论: {text}")
print(f"情感分析结果: {result}")
```

## 未来工作

- [ ] 添加多分类支持（例如：正面、中性、负面）
- [ ] 实现模型量化以减小模型体积
- [ ] 开发Web演示界面
- [ ] 支持更多预训练模型（如RoBERTa、ERNIE等）
- [ ] 添加更多中文评论数据集

## 贡献指南

欢迎贡献代码、报告问题或提出建议！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。详情请参阅[LICENSE](LICENSE)文件。