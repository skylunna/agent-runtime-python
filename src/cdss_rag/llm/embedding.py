import logging
import threading
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


from ..core.config import settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding 模型封装
        - 单例：模型只加载一次
        - 线程安全：sentence-transformers 模型本身是线程安全的
        - 同步接口：模型推理是 CPU/GPU 密集任务，async 包一层即可
    """

    _instance = None
    _lock = threading.Lock()


    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 防止重复初始化
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model_name = settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded, dim={self.dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        批量编码，返回归一化后的向量
        归一化是为了用余弦距离（与HNSW的 vector_consine_ops 配合）
        """
        if not texts:
            return np.array([])
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    

    def encode_one(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode([text])[0]
    

# 模块级单例
embedding_service = EmbeddingService()