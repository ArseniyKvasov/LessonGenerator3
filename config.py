import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_timeout_seconds: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
    groq_temperature: float = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
    groq_max_tokens: int = int(os.getenv("GROQ_MAX_TOKENS", "1200"))
    pollinations_api_key: str = os.getenv("POLLINATIONS_API_KEY", "")
    pollinations_base_url: str = os.getenv("POLLINATIONS_BASE_URL", "https://gen.pollinations.ai")
    pollinations_timeout_seconds: float = float(os.getenv("POLLINATIONS_TIMEOUT_SECONDS", "60"))
    pollinations_image_model: str = os.getenv("POLLINATIONS_IMAGE_MODEL", "gptimage")
    pollinations_image_size: str = os.getenv("POLLINATIONS_IMAGE_SIZE", "512x512")
    pollinations_image_quality: str = os.getenv("POLLINATIONS_IMAGE_QUALITY", "low")
    pollinations_audio_model: str = os.getenv("POLLINATIONS_AUDIO_MODEL", "qwen-tts")
    pollinations_audio_voice: str = os.getenv("POLLINATIONS_AUDIO_VOICE", "alloy")
    pollinations_audio_format: str = os.getenv("POLLINATIONS_AUDIO_FORMAT", "mp3")


settings = Settings()
