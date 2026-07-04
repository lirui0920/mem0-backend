from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="production", validation_alias="APP_ENV")
    app_name: str = Field(default="memory-chat-service", validation_alias="APP_NAME")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    service_api_key: str | None = Field(default=None, validation_alias="SERVICE_API_KEY")

    llm_api_key: str = Field(validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"))
    llm_base_url: str = Field(default="https://api.deepseek.com", validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"))
    llm_chat_model: str = Field(default="deepseek-v4-flash", validation_alias=AliasChoices("LLM_CHAT_MODEL", "OPENAI_CHAT_MODEL"))
    llm_temperature: float = Field(default=0.3, validation_alias=AliasChoices("LLM_TEMPERATURE", "OPENAI_TEMPERATURE"))
    llm_max_tokens: int = Field(default=800, validation_alias=AliasChoices("LLM_MAX_TOKENS", "OPENAI_MAX_TOKENS"))

    mem0_collection: str = Field(default="chat_memories", validation_alias="MEM0_COLLECTION")
    mem0_llm_provider: str = Field(default="openai", validation_alias="MEM0_LLM_PROVIDER")
    mem0_llm_model: str = Field(default="deepseek-v4-flash", validation_alias="MEM0_LLM_MODEL")
    mem0_embedder_provider: str = Field(default="huggingface", validation_alias="MEM0_EMBEDDER_PROVIDER")
    mem0_embedder_model: str = Field(default="/root/bge-model", validation_alias="MEM0_EMBEDDER_MODEL")
    mem0_embedder_dims: int = Field(default=384, validation_alias="MEM0_EMBEDDER_DIMS")
    qdrant_path: Path = Field(default=Path("./storage/qdrant"), validation_alias="QDRANT_PATH")
    mem0_history_db_path: Path = Field(default=Path("./storage/mem0_history.db"), validation_alias="MEM0_HISTORY_DB_PATH")
    memory_debug_log_path: Path = Field(default=Path("./storage/memory_debug_logs.jsonl"), validation_alias="MEMORY_DEBUG_LOG_PATH")
    memory_profile_path: Path = Field(default=Path("./storage/personality_profiles.json"), validation_alias="MEMORY_PROFILE_PATH")
    summary_interval_seconds: int = Field(default=86400, validation_alias="SUMMARY_INTERVAL_SECONDS")
    summary_memory_batch_size: int = Field(default=100, validation_alias="SUMMARY_MEMORY_BATCH_SIZE")
    decay_medium_penalty: float = Field(default=0.35, validation_alias="DECAY_MEDIUM_PENALTY")
    decay_strong_penalty: float = Field(default=0.75, validation_alias="DECAY_STRONG_PENALTY")
    summary_retention_boost: float = Field(default=0.6, validation_alias="SUMMARY_RETENTION_BOOST")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    settings.mem0_history_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.memory_debug_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.memory_profile_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
