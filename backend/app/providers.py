"""Provider registry — one project, any LLM vendor.

NimbusSupport talks to whichever provider you have an API key for. Most vendors
speak the OpenAI wire format, so they only differ by base URL and model name;
Azure and Anthropic get their own dedicated connectors.

Set `LLM_PROVIDER` explicitly, or just drop a key in `.env` and let
`resolve_providers()` auto-detect which vendor you meant.
"""

import os
from dataclasses import dataclass

# --- provider catalogue ------------------------------------------------------

# kind drives which Semantic Kernel connector is built in kernel_setup.py:
#   "openai_compatible" -> OpenAIChatCompletion pointed at a custom base_url
#   "azure"             -> AzureChatCompletion (deployment + endpoint + api-version)
#   "anthropic"         -> AnthropicChatCompletion (native Messages API)
#   "local"             -> no network call; embeddings computed on this machine


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    label: str
    kind: str
    base_url: str | None
    default_chat_model: str | None
    default_embedding_model: str | None  # None => this provider cannot embed
    key_env: tuple[str, ...] = ()  # env var names holding this vendor's API key
    model_env: tuple[str, ...] = ()  # env var names holding the chat model
    embedding_model_env: tuple[str, ...] = ()
    base_url_env: tuple[str, ...] = ()
    needs_key: bool = True
    docs: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    "azure": ProviderSpec(
        key="azure",
        label="Microsoft Foundry / Azure OpenAI",
        kind="azure",
        base_url=None,  # comes from AZURE_OPENAI_ENDPOINT
        default_chat_model="gpt-4o-mini",
        default_embedding_model="text-embedding-3-small",
        key_env=("AZURE_OPENAI_API_KEY",),
        model_env=("AZURE_OPENAI_MODEL", "AZURE_OPENAI_DEPLOYMENT"),
        embedding_model_env=("AZURE_OPENAI_EMBEDDING_MODEL",),
        base_url_env=("AZURE_OPENAI_ENDPOINT",),
        docs="https://ai.azure.com/",
    ),
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI",
        kind="openai_compatible",
        base_url="https://api.openai.com/v1",
        default_chat_model="gpt-4o-mini",
        default_embedding_model="text-embedding-3-small",
        key_env=("OPENAI_API_KEY",),
        model_env=("OPENAI_MODEL",),
        embedding_model_env=("OPENAI_EMBEDDING_MODEL",),
        base_url_env=("OPENAI_BASE_URL",),
        docs="https://platform.openai.com/api-keys",
    ),
    "anthropic": ProviderSpec(
        key="anthropic",
        label="Anthropic",
        kind="anthropic",
        base_url=None,
        default_chat_model="claude-sonnet-5",
        default_embedding_model=None,  # Anthropic has no embeddings API
        key_env=("ANTHROPIC_API_KEY",),
        model_env=("ANTHROPIC_MODEL",),
        docs="https://console.anthropic.com/settings/keys",
    ),
    "gemini": ProviderSpec(
        key="gemini",
        label="Google Gemini",
        kind="openai_compatible",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_chat_model="gemini-2.5-flash",
        default_embedding_model="gemini-embedding-001",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        model_env=("GEMINI_MODEL",),
        embedding_model_env=("GEMINI_EMBEDDING_MODEL",),
        docs="https://aistudio.google.com/apikey",
    ),
    "openrouter": ProviderSpec(
        key="openrouter",
        label="OpenRouter",
        kind="openai_compatible",
        base_url="https://openrouter.ai/api/v1",
        default_chat_model="openai/gpt-4o-mini",
        default_embedding_model=None,  # OpenRouter routes chat only
        key_env=("OPENROUTER_API_KEY",),
        model_env=("OPENROUTER_MODEL",),
        docs="https://openrouter.ai/keys",
    ),
    "groq": ProviderSpec(
        key="groq",
        label="Groq",
        kind="openai_compatible",
        base_url="https://api.groq.com/openai/v1",
        default_chat_model="llama-3.3-70b-versatile",
        default_embedding_model=None,
        key_env=("GROQ_API_KEY",),
        model_env=("GROQ_MODEL",),
        docs="https://console.groq.com/keys",
    ),
    "cerebras": ProviderSpec(
        key="cerebras",
        label="Cerebras",
        kind="openai_compatible",
        base_url="https://api.cerebras.ai/v1",
        default_chat_model="llama-3.3-70b",
        default_embedding_model=None,
        key_env=("CEREBRAS_API_KEY",),
        model_env=("CEREBRAS_MODEL",),
        docs="https://cloud.cerebras.ai/",
    ),
    "ollama": ProviderSpec(
        key="ollama",
        label="Ollama (local)",
        kind="openai_compatible",
        base_url="http://localhost:11434/v1",
        default_chat_model="llama3.1",
        default_embedding_model="nomic-embed-text",
        key_env=("OLLAMA_API_KEY",),
        model_env=("OLLAMA_MODEL",),
        embedding_model_env=("OLLAMA_EMBEDDING_MODEL",),
        base_url_env=("OLLAMA_BASE_URL",),
        needs_key=False,  # a local server needs no credentials
        docs="https://ollama.com/download",
    ),
    "local": ProviderSpec(
        key="local",
        label="Local ONNX (MiniLM-L6-v2)",
        kind="local",
        base_url=None,
        default_chat_model=None,  # embeddings only — cannot answer questions
        default_embedding_model="all-MiniLM-L6-v2",
        needs_key=False,
        docs="bundled with ChromaDB — no key, no network",
    ),
}

# Providers that can serve the agent, in the order auto-detection tries them.
CHAT_PROVIDERS: tuple[str, ...] = (
    "azure",
    "openai",
    "anthropic",
    "gemini",
    "openrouter",
    "groq",
    "cerebras",
    "ollama",
)


class ProviderConfigError(RuntimeError):
    """Raised when the .env does not describe a usable provider."""


# --- resolved configuration --------------------------------------------------


@dataclass(frozen=True)
class ServiceConfig:
    """Everything needed to construct one connector (chat or embeddings)."""

    provider: str
    model: str
    api_key: str
    base_url: str | None
    api_version: str  # Azure only

    @property
    def spec(self) -> ProviderSpec:
        return PROVIDERS[self.provider]

    @property
    def label(self) -> str:
        return self.spec.label


@dataclass(frozen=True)
class ProviderConfig:
    chat: ServiceConfig
    embedding: ServiceConfig


# --- env helpers -------------------------------------------------------------


def _env(*names: str) -> str:
    """First non-empty value among `names`, else empty string."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _normalise(name: str) -> str:
    key = name.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "azure-openai": "azure",
        "microsoft": "azure",
        "microsoft-foundry": "azure",
        "foundry": "azure",
        "ai-foundry": "azure",
        "google": "gemini",
        "google-gemini": "gemini",
        "googleai": "gemini",
        "open-router": "openrouter",
        "claude": "anthropic",
        "open-ai": "openai",
    }
    return aliases.get(key, key)


def _known(name: str, allowed: tuple[str, ...]) -> str:
    key = _normalise(name)
    if key not in allowed:
        raise ProviderConfigError(
            f"Unknown provider '{name}'. Supported: {', '.join(allowed)}."
        )
    return key


def _detect_chat_provider() -> str:
    """Pick the provider whose API key is actually present in the environment."""
    for key in CHAT_PROVIDERS:
        spec = PROVIDERS[key]
        if spec.needs_key and _env(*spec.key_env):
            return key
    # Nothing keyed: Ollama is the only provider that can run without one.
    if _env("OLLAMA_BASE_URL", "OLLAMA_MODEL"):
        return "ollama"
    vendor_keys = ", ".join(
        PROVIDERS[k].key_env[0] for k in CHAT_PROVIDERS if PROVIDERS[k].needs_key
    )
    raise ProviderConfigError(
        "No LLM provider configured. Set LLM_PROVIDER and LLM_API_KEY in backend/.env "
        f"(or a vendor key such as {vendor_keys}), or run Ollama locally and set LLM_PROVIDER=ollama. "
        "See backend/.env.example."
    )


def _resolve_chat() -> ServiceConfig:
    requested = _env("LLM_PROVIDER", "CHAT_PROVIDER")
    provider = _known(requested, CHAT_PROVIDERS) if requested else _detect_chat_provider()
    spec = PROVIDERS[provider]

    api_key = _env("LLM_API_KEY", *spec.key_env)
    if spec.needs_key and not api_key:
        raise ProviderConfigError(
            f"{spec.label} selected but no API key found. "
            f"Set LLM_API_KEY or {spec.key_env[0]} in backend/.env — get one at {spec.docs}."
        )

    base_url = _env("LLM_BASE_URL", *spec.base_url_env) or spec.base_url
    if spec.kind == "azure" and not base_url:
        raise ProviderConfigError(
            "Microsoft Foundry / Azure OpenAI needs AZURE_OPENAI_ENDPOINT "
            "(e.g. https://your-resource.openai.azure.com/)."
        )

    return ServiceConfig(
        provider=provider,
        model=_env("LLM_MODEL", *spec.model_env) or (spec.default_chat_model or ""),
        api_key=api_key or "not-needed",
        base_url=base_url,
        api_version=_env("AZURE_OPENAI_API_VERSION") or "2024-10-21",
    )


def _resolve_embedding(chat: ServiceConfig) -> ServiceConfig:
    requested = _env("EMBEDDING_PROVIDER")
    if requested:
        provider = _known(requested, tuple(PROVIDERS))
    elif chat.spec.default_embedding_model:
        provider = chat.provider  # same vendor handles both
    else:
        # Anthropic / OpenRouter / Groq / Cerebras have no embeddings API —
        # fall back to the ONNX model bundled with ChromaDB so RAG still works.
        provider = "local"

    spec = PROVIDERS[provider]
    if not spec.default_embedding_model:
        raise ProviderConfigError(
            f"{spec.label} does not provide embeddings. "
            "Set EMBEDDING_PROVIDER to one of: azure, openai, gemini, ollama, local."
        )

    api_key = _env("EMBEDDING_API_KEY", *spec.key_env)
    if not api_key and provider == chat.provider:
        api_key = chat.api_key  # reuse the chat credential
    if spec.needs_key and not api_key:
        raise ProviderConfigError(
            f"Embeddings via {spec.label} need a key. Set EMBEDDING_API_KEY or {spec.key_env[0]}."
        )

    base_url = _env("EMBEDDING_BASE_URL", *spec.base_url_env) or spec.base_url
    if provider == chat.provider and chat.base_url:
        base_url = _env("EMBEDDING_BASE_URL") or chat.base_url

    return ServiceConfig(
        provider=provider,
        model=_env("EMBEDDING_MODEL", *spec.embedding_model_env) or spec.default_embedding_model,
        api_key=api_key or "not-needed",
        base_url=base_url,
        api_version=chat.api_version,
    )


def resolve_providers() -> ProviderConfig:
    chat = _resolve_chat()
    return ProviderConfig(chat=chat, embedding=_resolve_embedding(chat))


def describe(config: ProviderConfig) -> dict:
    """Safe-to-log summary — never includes the API key."""
    return {
        "chat": {
            "provider": config.chat.provider,
            "label": config.chat.label,
            "model": config.chat.model,
        },
        "embedding": {
            "provider": config.embedding.provider,
            "label": config.embedding.label,
            "model": config.embedding.model,
        },
    }
