from dataclasses import dataclass

@dataclass
class Settings:
    random_state: int = 42

def get_settings():
    settings = Settings()
    return settings