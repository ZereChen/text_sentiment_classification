import torch.nn as nn
from src.utils.model_loader import ModelLoader

class BertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes=2, dropout_rate=0.1):
        super(BertClassifier, self).__init__()
        bert, tokenizer = ModelLoader.load_pretrained(model_name = bert_model_name, is_modelscope=True)
        self.bert = bert
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(dropout_rate) # 随机丢弃一些神经元，防止过拟合
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes) # 线性层，将BERT的输出映射到 num_classes 个类别

    def forward(self, input_ids, attention_mask):
        # [CLS] 我 爱 自然 语言 处理 [SEP]
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ) #返回一个元组，包含：所有token的隐藏状态（批次大小、序列长度、隐藏层维度），[CLS]标记的池化输出（批次大小、隐藏层维度）
        pooled_output = outputs[1]  # [CLS] token output
        pooled_output = self.dropout(pooled_output) # 随机丢弃一些神经元，防止过拟合，在训练时生效，评估时自动关闭
        logits = self.classifier(pooled_output) # 将pooled_output映射到 num_classes 个类别
        return logits