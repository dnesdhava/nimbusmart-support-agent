"""Embeddings that work the same whichever provider is configured.

Every backend below returns plain `list[list[float]]`, so ingestion and
retrieval never need to know who produced the vectors.
"""

import asyncio
from functools import lru_cache
from typing import Protocol
from app.config import get_settings
from app.providers import ProviderConfigError, ServiceConfig

class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class RemoteEmbedder:
    """Azure / OpenAI / Gemini / Ollama — anything with an embeddings endpoint."""

    def __init__(self, config: ServiceConfig) -> None:
        self.model = config.model
        self._service = _build_service(config)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = await self._service.generate_embeddings(texts)
        return [v.tolist() for v in vectors]

class LocalEmbedder:
    """ChromaDB's bundled MiniLM ONNX model — no key, no network call.

    This is what keeps RAG working for providers with no embeddings API of
    their own (Anthropic, OpenRouter, Groq, Cerebras).
    """

    def __init__(self, config: ServiceConfig) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self.model = config.model
        self._fn = ONNXMiniLM_L6_V2()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # The ONNX model is synchronous; keep the event loop free while it runs.
        vectors = await asyncio.to_thread(self._fn, texts)
        return [list(map(float, v)) for v in vectors]

def _build_service(config: ServiceConfig):
    from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding, OpenAITextEmbedding

    if config.spec.kind == "azure":
        return AzureTextEmbedding(
            deployment_name=config.model,
            endpoint=config.base_url,
            api_key=config.api_key,
            api_version=config.api_version,
        )
    if config.spec.kind == "openai_compatible":
        from app.kernel_setup import openai_compatible_client

        return OpenAITextEmbedding(
            ai_model_id=config.model,
            async_client=openai_compatible_client(config),
        )
    raise ProviderConfigError(f"{config.label} cannot generate embeddings.")

@lru_cache
def get_embedder() -> Embedder:
    config = get_settings().providers.embedding
    return LocalEmbedder(config) if config.spec.kind == "local" else RemoteEmbedder(config)
