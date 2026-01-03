from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "podcast-automate"
    VERSION: str = "0.1.0"
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
