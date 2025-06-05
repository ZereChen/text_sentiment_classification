import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        :param texts: 文本
        :param labels: 标签
        :param tokenizer: 分词器
        :param max_length: 文本的最大长度，默认为128个token
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        text = self.augment_text(text)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # 添加特殊token
            max_length=self.max_length,  # 最大长度
            padding='max_length',  # 将短文本填充到最大长度
            truncation=True,  # 将长文本截断到最大长度
            return_tensors='pt',  # 返回PyTorch张量
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),  # 将input_ids词汇id展平， "我喜欢这个产品" 可能被转换为 [101, 2769, 3221, 102, 0, 0, ...]，其中101是[CLS]标记，102是[SEP]标记，0是填充标记
            'attention_mask': encoding['attention_mask'].flatten(),  # 将attention_mask展平, 1表示实际token，0表示填充token，例如：[1, 1, 1, 1, 0, 0, ...]
            'labels': torch.tensor(label, dtype=torch.long)  # 将真实的label转换为PyTorch long张量
        }

    def augment_text(self, text):
        # 实现数据增强方法
        # 1. 同义词替换
        # 2. 随机删除
        # 3. 随机插入
        # 4. 回译
        return text

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['review'].values, df['label'].values
