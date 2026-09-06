from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    PERSIST_DIRECTORY: str
    AI_PROVIDER: str
    # Retrieval remains available even when no chat LLM is configured.
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    API_KEY: str
    FILE_ALLOWED_EXTENSIONS: list[str]
    FILE_MAX_SIZE: int
    DATAFRAME_RAW_LOCATION: str
    DATAFRAME_PROCESSED_LOCATION: str
    LLM_MODEL: str = ""
    LLM_BASE_URL: str | None = None
    LLM_TEMPERATURE: float = 0.0
    DATAFRAME_CLEANED_LOCATION: str
    DATAFRAME_CLEANED_RAW_LOCATION: str
    


    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def get_settings() -> Settings:
    return Settings()
