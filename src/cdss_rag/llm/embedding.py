import logging
import threading
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI 


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

        logger.info(f"Initializing cloud embedding client for model: {settings.embedding_model}")

        # 初始化 OpenAI 兼容客户端
        self.client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )
        self.model_name = settings.embedding_model
        # 云端 API 无法动态获取维度，直接从配置读取
        self.dim = settings.embedding_dim

        logger.info(f"Cloud embedding client initialized, dim={self.dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        批量编码，返回归一化后的向量
        归一化是为了用余弦距离（与HNSW的 vector_consine_ops 配合）
        """
        if not texts:
            return np.array([])

        # 1. 调用云端API
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        # 2. 提取向量并转换为 numpy array
        embeddings = [item.embedding for item in response.data]
        vectors = np.array(embeddings, dtype=np.float32)

        # 3. 手动进行 L2 归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1 # 防止 除以 0
        vectors = vectors / norms

        return vectors


    def encode_one(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode([text])[0]
    

# 模块级单例
embedding_service = EmbeddingService()