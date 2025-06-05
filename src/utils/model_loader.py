from transformers import AutoTokenizer, AutoModel
import logging
import logging
import os
import shutil
from typing import Optional, Tuple, Any

from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """模型加载器类"""

    # 类级别的缓存
    _model_cache = {}
    _tokenizer_cache = {}
    _cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

    @classmethod
    def init(cls, cache_dir: Optional[str] = None):
        """
        初始化模型加载器
        
        Args:
            cache_dir: 模型缓存目录，如果为None则使用默认目录
        """
        cls._cache_dir = cache_dir or cls._cache_dir
        # 确保缓存目录存在
        os.makedirs(cls._cache_dir, exist_ok=True)
        logger.info(f"模型缓存目录: {cls._cache_dir}")

    @classmethod
    def load_pretrained(
            cls,
            model_name: str,
            use_cache: bool = True,
            **kwargs
    ) -> Tuple[Any, Any]:
        """
        加载预训练模型和tokenizer
        
        Args:
            model_name: 模型名称或路径
            use_cache: 是否使用内存缓存
            **kwargs: 其他参数
            
        Returns:
            Tuple[Any, Any]: (model, tokenizer)元组
        """
        try:
            # 尝试从内存缓存加载
            if use_cache:
                model = cls._model_cache.get(model_name)
                tokenizer = cls._tokenizer_cache.get(model_name)
                if model is not None and tokenizer is not None:
                    logger.info(f"从内存缓存加载模型和tokenizer: {model_name}")
                    return model, tokenizer

            # 尝试从本地缓存目录加载
            if cls._cache_dir and os.path.exists(os.path.join(cls._cache_dir, model_name)):
                logger.info(f"从本地缓存加载模型和tokenizer: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.join(cls._cache_dir, model_name),
                    **kwargs
                )
                model = AutoModel.from_pretrained(
                    os.path.join(cls._cache_dir, model_name),
                    **kwargs
                )
            else:
                # 从远程加载
                logger.info(f"从远程加载模型和tokenizer: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cls._cache_dir,
                    **kwargs
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cls._cache_dir,
                    **kwargs
                )

            # 存入内存缓存
            if use_cache:
                cls._model_cache[model_name] = model
                cls._tokenizer_cache[model_name] = tokenizer

            return model, tokenizer

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    @classmethod
    def clear_cache(cls):
        """清除模型和tokenizer缓存"""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        logger.info("模型缓存已清除")

    @classmethod
    def clear_disk_cache(cls):
        """清除磁盘上的模型缓存"""
        if cls._cache_dir and os.path.exists(cls._cache_dir):
            shutil.rmtree(cls._cache_dir)
            os.makedirs(cls._cache_dir)
            logger.info("磁盘缓存已清除")


# 初始化默认缓存目录
ModelLoader.init()
