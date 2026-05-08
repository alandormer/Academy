from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Database ---
    database_url: str = "postgresql+asyncpg://contextengine:contextengine_dev@localhost:5432/academy_knowledge"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # --- LLM (GGUF via llama-cpp-python) ---
    gguf_model_path: str = "/Users/alandormer/Projects/contextengine/models/mistral_models/mistral-7b-instruct-v0.3.Q6_K.gguf"
    gguf_n_gpu_layers: int = -1   # -1 = all layers on Metal GPU
    gguf_n_ctx: int = 4096

    # --- Chunking ---
    chunk_size: int = 400
    chunk_overlap: int = 60

    # --- Retrieval ---
    retrieval_top_k: int = 6

    # --- MinIO ---
    minio_endpoint: str = "localhost:9100"
    minio_access_key: str = "contextengine"
    minio_secret_key: str = "contextengine123"
    minio_secure: bool = False
    minio_bucket_raw: str = "academy-raw"
    minio_bucket_audio: str = "academy-audio"
    minio_bucket_exports: str = "academy-exports"

    # --- Application ---
    log_level: str = "INFO"
    environment: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
