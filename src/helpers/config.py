from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    PERSIST_DIRECTORY: str
    AI_PROVIDER: str
    MODEL_NAME: str
    API_KEY: str
    FILE_ALLOWED_EXTENSIONS: list[str]
    FILE_MAX_SIZE: int
    DATAFRAME_RAW_LOCATION: str
    DATAFRAME_PROCESSED_LOCATION: str


    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def get_settings() -> Settings:
    return Settings()