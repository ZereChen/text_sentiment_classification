import torch
from transformers import BertTokenizer

from src.models.bert_classifier import BertClassifier

# 加载模型
model = BertClassifier('bert-base-chinese')
model.load_state_dict(torch.load('outputs/best_model.pth'))
# 从训练模式 model.train() 切换为评估模式 model.eval()
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
        _, predicted = torch.max(outputs, 1)  # 返回 最大值（logits 值）和 最大值所在的索引（即预测的类别）

    return "正面评价" if predicted.item() == 1 else "负面评价"


# 示例
text = "真的太喜欢了，耐用"
result = predict_sentiment(text)
print(f"评论: {text} 情感分析结果: {result}")
