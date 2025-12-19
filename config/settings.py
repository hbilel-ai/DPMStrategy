# config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if it exists)
load_dotenv() 

class Settings:
    # --- Database Settings ---
    # Prioritize TRADING_DB_URL environment variable.
    # Provide a safe default for development (e.g., SQLite file)
    DATABASE_URL: str = os.getenv(
        "TRADING_DB_URL",
        "sqlite:///./live_trading.db"  # Default SQLite file for local dev/testing
    )

    # --- Live Engine Settings ---
    LIVE_TICKER: str = os.getenv("LIVE_TICKER", "BNP.PA")
    
    # --- API Settings ---
    API_PORT: int = int(os.getenv("API_PORT", 8000))

# Instantiate the settings object
settings = Settings()

# Example usage in other files: from config.settings import settings
# db_url = settings.DATABASE_URL
