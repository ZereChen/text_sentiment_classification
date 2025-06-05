import logging
from pathlib import Path
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.bert_classifier import BertClassifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="情感分析API",
    description="使用BERT模型进行中文文本情感分析",
    version="1.0.0"
)

# 定义请求模型
class TextInput(BaseModel):
    text: str
    max_length: Optional[int] = 128

class BatchTextInput(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 128

# 定义响应模型
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

# 全局变量
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    """加载模型和tokenizer"""
    global model, tokenizer
    try:
        # 检查模型文件是否存在
        model_path = Path('outputs/best_model.pth')
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型
        model = BertClassifier('bert-base-chinese')
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 加载tokenizer
        tokenizer = model.tokenizer
        logger.info("模型和tokenizer加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

def predict_sentiment(text: str, max_length: int = 128) -> tuple:
    """
    预测单条文本的情感
    
    Args:
        text: 输入文本
        max_length: 最大序列长度
        
    Returns:
        tuple: (情感标签, 置信度)
    """
    try:
        # 预处理文本
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 预测
        with torch.no_grad():
            outputs = model(encoding['input_ids'], encoding['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1) # 返回 最大值（logits 值）和 最大值所在的索引（即预测的类别）

        sentiment = "正面评价" if predicted.item() == 1 else "负面评价"
        return sentiment, confidence.item()

    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/predict", response_model=SentimentResponse)
async def predict(input_data: TextInput):
    """
    预测单条文本的情感
    
    Args:
        input_data: 包含文本和最大长度的输入数据
        
    Returns:
        SentimentResponse: 包含预测结果的响应
    """
    try:
        sentiment, confidence = predict_sentiment(input_data.text, input_data.max_length)
        return SentimentResponse(
            text=input_data.text,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchSentimentResponse)
async def predict_batch(input_data: BatchTextInput):
    """
    批量预测文本情感
    
    Args:
        input_data: 包含文本列表和最大长度的输入数据
        
    Returns:
        BatchSentimentResponse: 包含批量预测结果的响应
    """
    try:
        results = []
        for text in input_data.texts:
            sentiment, confidence = predict_sentiment(text, input_data.max_length)
            results.append(SentimentResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence
            ))
        return BatchSentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
