from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import COLLECTION_NAME, get_settings
from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.observability import setup_observability

app = FastAPI(title="Nimbusmart Support Agent")
setup_observability(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
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
        name = {c.name for c in client.list_collections()}
        indexed = COLLECTION_NAME in name
    except Exception:
        indexed = False
    return {"status": "ok", "knowleadge_base_indexed": indexed}