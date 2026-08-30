from pydantic-settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME:str
    class config:
        env_file = ".env"
