"""Central configuration loaded from .env and environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM — tiered
    groq_api_key: str = ""
    groq_model_fast: str = "openai/gpt-oss-20b"       # extraction agents
    groq_model_reason: str = "openai/gpt-oss-120b"    # reasoning agents
    groq_model_alt: str = "qwen/qwen3.6-27b"          # debate diversity
    
    # Optional HF classifier for sentiment
    hf_api_key: str = ""
    hf_sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Paths
    data_raw_path: Path = Path("./data/raw")
    data_processed_path: Path = Path("./data/processed")
    chroma_db_path: Path = Path("./data/chroma")
    output_path: Path = Path("./output")
    inferred_events_file: Path = Path("./output/inferred-events.jsonl")

    # Logging
    log_level: str = "INFO"

    # Memory decay
    memory_decay_lambda: float = 0.01
    memory_base_weight: float = 1.0

    def ensure_dirs(self) -> None:
        for p in [
            self.data_raw_path,
            self.data_processed_path,
            self.chroma_db_path,
            self.output_path,
            self.inferred_events_file.parent,
        ]:
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()