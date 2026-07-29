from functools import lru_cache
import chromadb
from rank_bm25 import BM25Okapi
from app.config import CHROMA_DIR, LEGACY_COLLECTION_NAME, collection_name
from opentelemetry import trace
from app.kernel_setup import get_chat_service
from app.rag.embeddings import get_embedder
from app.rag.reranker import llm_rerank
from app.rag.retriever_types import Candidate

_tracer = trace.get_tracer("nimbus.rag")

RRF_K = 60
VECTOR_TOP_N = 10
BM25_TOP_N = 10
CANDIDATES_BEFORE_RERANK = 8
FINAL_TOP_K = 4

def _tokenize(text: str) -> list[str]:
    return text.lower().split()

class HybridRetriever:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        name = collection_name()
        existing = {c.name for c in self._client.list_collections()}
        # Fall back to the pre-multi-provider collection so older local indexes keep working.
        if name not in existing and LEGACY_COLLECTION_NAME in existing:
            name = LEGACY_COLLECTION_NAME
        try:
            self._collection = self._client.get_collection(name)
        except Exception as exc:
            raise RuntimeError(
                f"Collection '{collection_name()}' not found in ChromaDB at {CHROMA_DIR}. "
                "Build it for the current embedding model with: uv run python -m app.rag.ingest"
            ) from exc
        self._embedder = get_embedder()
        self._chat = get_chat_service()
        self._load_corpus()

    def _load_corpus(self) -> None:
        data = self._collection.get(include=["documents", "metadatas"])
        self._ids: list[str] = data["ids"]
        self._documents: list[str] = data["documents"] or []
        self._metadatas: list[dict] = [dict(m) for m in data["metadatas"] or []]
        self._bm25 = BM25Okapi([_tokenize(d) for d in self._documents])
        self._id_to_index = {id_: i for i, id_ in enumerate(self._ids)}

    async def search(self, query: str, top_k: int = FINAL_TOP_K) -> list[Candidate]:
        vector_ranked = await self._vector_search(query)
        bm25_ranked = self._bm25_search(query)
        fused_ids = self._reciprocal_rank_fusion([vector_ranked, bm25_ranked])
        candidates = [self._to_candidate(id_) for id_ in fused_ids[:CANDIDATES_BEFORE_RERANK]]
        return await llm_rerank(query, candidates, self._chat, top_k=top_k)
    
    async def _vector_search(self, query: str) -> list[str]:
        vector = (await self._embedder.embed([query]))[0]
        results = self._collection.query(query_embeddings=[vector], n_results=VECTOR_TOP_N)
        return results["ids"][0]
    
    def _bm25_search(self, query: str) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_TOP_N]
        return [self._ids[i] for i in ranked_indices]
    
    @staticmethod
    def _reciprocal_rank_fusion(rankings: list[list[str]]) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, id_ in enumerate(ranking):
                scores[id_] = scores.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(scores, key=lambda id_: scores[id_], reverse=True)
        
    def _to_candidate(self, id_: str) -> Candidate:
        i = self._id_to_index[id_]
        meta = self._metadatas[i]
        return Candidate(id=id_, text=self._documents[i], source=meta["source"], section=meta["section"])

@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever()