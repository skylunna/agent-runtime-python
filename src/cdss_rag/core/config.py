from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用
    app_env: str = "dev"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # 数据库
    pg_dsn: str
    pg_dsn_sa: str
    pg_pool_min: int = 2
    pg_pool_max: int = 10

    # LLM
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    

    # Embedding
    embedding_api_key: str
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1" 
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 512


settings = Settings()