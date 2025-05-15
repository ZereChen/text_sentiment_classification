# 基于BERT的中文文本情感分类

这个项目使用BERT模型实现中文文本情感分类任务。

## 项目结构

text_sentiment_classification/
├── README.md
├── requirements.txt
├── data/
│ └── Data.csv
└── src/
├── data/
│ ├── init.py
│ └── preprocess.py
├── models/
│ ├── init.py
│ └── bert_classifier.py
└── train.py

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- Transformers 4.5.0+
- 其他依赖见requirements.txt

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行训练：
```bash
python src/train.py
```

## 模型说明

本项目使用预训练的中文BERT模型（bert-base-chinese）作为基础模型，在其上添加分类层进行情感分类。模型结构如下：

1. BERT编码器：将输入文本转换为向量表示
2. Dropout层：防止过拟合
3. 分类层：将BERT输出映射到情感类别

## 数据集

数据集包含评论文本和对应的情感标签（1表示正面情感）。

## 训练过程

- 使用AdamW优化器
- 学习率：2e-5
- 批次大小：16
- 训练轮数：3
- 最大序列长度：128

## 评估指标

- 准确率（Accuracy）
- 损失值（Loss）