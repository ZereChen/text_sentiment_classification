import logging
import os
import shutil
from typing import Optional, Tuple, Any

from transformers import AutoTokenizer, AutoModel

from src.utils.log_utils import LoggerManager

logger = LoggerManager.get_logger(__name__)


class ModelLoader:
    """模型加载器类，支持HuggingFace和ModelScope模型"""

    # 类级别的缓存
    _model_cache = {}
    _tokenizer_cache = {}
    _hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    _ms_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models")

    @classmethod
    def init(cls, hf_cache_dir: Optional[str] = None, ms_cache_dir: Optional[str] = None):
        """
        初始化模型加载器
        
        Args:
            hf_cache_dir: HuggingFace模型缓存目录，如果为None则使用默认目录
            ms_cache_dir: ModelScope模型缓存目录，如果为None则使用默认目录
        """
        cls._hf_cache_dir = hf_cache_dir or cls._hf_cache_dir
        cls._ms_cache_dir = ms_cache_dir or cls._ms_cache_dir
        
        # 确保缓存目录存在
        os.makedirs(cls._hf_cache_dir, exist_ok=True)
        os.makedirs(cls._ms_cache_dir, exist_ok=True)
        
        logger.info(f"HuggingFace模型缓存目录: {cls._hf_cache_dir}")
        logger.info(f"ModelScope模型缓存目录: {cls._ms_cache_dir}")

    @classmethod
    def _is_modelscope_model(cls, model_name: str) -> bool:
        """
        判断是否为ModelScope模型
        
        Args:
            model_name: 模型名称或路径
            
        Returns:
            bool: 是否为ModelScope模型
        """
        return model_name.startswith("damo/") or model_name.startswith("modelscope/")

    @classmethod
    def load_pretrained(
            cls,
            model_name: str,
            use_cache: bool = True,
            is_modelscope: bool = False,
            **kwargs
    ) -> Tuple[Any, Any]:
        """
        加载预训练模型和tokenizer，支持HuggingFace和ModelScope模型
        
        Args:
            model_name: 模型名称或路径
            use_cache: 是否使用内存缓存
            is_modelscope: 是否为ModelScope模型
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
            
            # 选择缓存目录
            cache_dir = cls._ms_cache_dir if is_modelscope else cls._hf_cache_dir
            
            if is_modelscope:
                try:
                    from modelscope import AutoModel as MSModel
                    from modelscope import AutoTokenizer as MSTokenizer
                    
                    # 尝试从本地缓存目录加载
                    if os.path.exists(os.path.join(cache_dir, model_name)):
                        logger.info(f"从本地缓存目录加载ModelScope模型和tokenizer: {model_name}")
                        tokenizer = MSTokenizer.from_pretrained(
                            os.path.join(cache_dir, model_name),
                            **kwargs
                        )
                        model = MSModel.from_pretrained(
                            os.path.join(cache_dir, model_name),
                            **kwargs
                        )
                    else:
                        # 从远程加载
                        logger.info(f"从远程服务器加载ModelScope模型和tokenizer: {model_name}")
                        tokenizer = MSTokenizer.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            **kwargs
                        )
                        model = MSModel.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            **kwargs
                        )
                except ImportError:
                    logger.warning("未安装modelscope，尝试使用transformers加载")

            if not is_modelscope:
                # 使用transformers加载
                if os.path.exists(os.path.join(cache_dir, model_name)):
                    logger.info(f"从本地缓存目录加载HuggingFace模型和tokenizer: {model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        os.path.join(cache_dir, model_name),
                        **kwargs
                    )
                    model = AutoModel.from_pretrained(
                        os.path.join(cache_dir, model_name),
                        **kwargs
                    )
                else:
                    # 从远程加载
                    logger.info(f"从远程服务器加载HuggingFace模型和tokenizer: {model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        **kwargs
                    )
                    model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
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
    def clear_disk_cache(cls, is_modelscope: bool = False):
        """
        清除磁盘上的模型缓存
        
        Args:
            is_modelscope: 是否清除ModelScope缓存，默认为False（清除HuggingFace缓存）
        """
        cache_dir = cls._ms_cache_dir if is_modelscope else cls._hf_cache_dir
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            logger.info(f"{'ModelScope' if is_modelscope else 'HuggingFace'}磁盘缓存已清除")


# 初始化默认缓存目录
ModelLoader.init()
