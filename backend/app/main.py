import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import LEGACY_COLLECTION_NAME, collection_name, get_settings
from app.providers import describe
from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.observability import setup_observability

app = FastAPI(title="Nimbusmart Support Agent")
setup_observability(app)

settings = get_settings()
logging.getLogger("uvicorn.error").info(
    "LLM provider: %s (%s) · embeddings: %s (%s)",
    settings.providers.chat.label,
    settings.providers.chat.model,
    settings.providers.embedding.label,
    settings.providers.embedding.model,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

@app.get("/api/health")
def health() -> dict:
    from app.config import CHROMA_DIR
    import chromadb

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        names = {c.name for c in client.list_collections()}
        indexed = bool(names & {collection_name(), LEGACY_COLLECTION_NAME})
    except Exception:
        indexed = False
    return {"status": "ok", "knowleadge_base_indexed": indexed, **describe(get_settings().providers)}

@app.get("/api/provider")
def provider() -> dict:
    """Which vendor is answering right now — handy when swapping keys."""
    return describe(get_settings().providers)
