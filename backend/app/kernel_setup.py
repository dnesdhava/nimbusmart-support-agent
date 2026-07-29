"""Builds the chat service for whichever provider is configured."""

from functools import lru_cache
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from app.config import get_settings
from app.providers import ProviderConfigError, ServiceConfig

def openai_compatible_client(config: ServiceConfig) -> AsyncOpenAI:
    """OpenAI, Gemini, OpenRouter, Groq, Cerebras and Ollama all speak this dialect."""
    return AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

def chat_service(service_id: str = "chat") -> ChatCompletionClientBase:
    config = get_settings().providers.chat

    if config.spec.kind == "azure":
        return AzureChatCompletion(
            service_id=service_id,
            deployment_name=config.model,
            endpoint=config.base_url,
            api_key=config.api_key,
            api_version=config.api_version,
        )

    if config.spec.kind == "anthropic":
        try:
            from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
        except ImportError as exc:  # the Anthropic SDK ships as an optional extra
            raise ProviderConfigError(
                "Anthropic support needs the 'anthropic' package — run `uv sync` in backend/."
            ) from exc
        return AnthropicChatCompletion(
            service_id=service_id,
            ai_model_id=config.model,
            api_key=config.api_key,
        )

    if config.spec.kind == "openai_compatible":
        return OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id=config.model,
            async_client=openai_compatible_client(config),
        )

    raise ProviderConfigError(f"{config.label} cannot serve chat completions.")

@lru_cache
def get_chat_service() -> ChatCompletionClientBase:
    """Shared instance for callers outside the agent (e.g. the RAG reranker)."""
    return chat_service("rerank")

def build_kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_service(chat_service())
    return kernel
