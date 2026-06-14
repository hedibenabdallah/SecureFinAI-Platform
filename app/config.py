from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    environment: str = "development"
    log_level: str = "INFO"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    database_url: str = (
        "postgresql+asyncpg://securefinai:securefinai@localhost:5432/securefinai"
    )
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"
    cache_similarity_threshold: float = 0.85


settings = Settings()


def get_settings() -> Settings:
    return settings
