import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from app.providers import ProviderConfig, resolve_providers

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BACKEND_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BACKEND_DIR / "chroma_db"
DOCUMENTS_DIR = Path(__file__).resolve().parent / "data" / "documents"
ORDERS_PATH = Path(__file__).resolve().parent / "data" / "orders.json"
USERS_PATH = Path(__file__).resolve().parent / "data" / "users.json"

DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
LEGACY_COLLECTION_NAME = "nimbusmart_support_kb"  # indexes built before providers were pluggable

@dataclass(frozen=True)
class Settings:
    providers: ProviderConfig
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

@lru_cache()
def get_settings() -> Settings:
    origins = os.environ.get("CORS_ORIGINS", "").strip()
    return Settings(
        providers=resolve_providers(),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()) or DEFAULT_CORS_ORIGINS,
    )

def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:40] or "default"

@lru_cache()
def collection_name() -> str:
    """Vectors are namespaced per embedding model.

    Providers emit different vector dimensions, so switching from (say) OpenAI
    to Ollama builds its own collection instead of colliding with an index that
    was created by the previous model.
    """
    embedding = get_settings().providers.embedding
    return f"nimbusmart_kb__{_slug(embedding.provider)}__{_slug(embedding.model)}"
