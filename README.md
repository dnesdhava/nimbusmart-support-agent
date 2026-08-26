<div align="center">

# NimbusMart Support Agent

### A retrieval-augmented customer support agent built with Semantic Kernel, ChromaDB and FastAPI

**[HERE AND NOW AI](https://hereandnowai.com)** — *AI is Good*

</div>

---

## Overview

NimbusMart Support Agent is a role-aware customer support assistant for a fictional e-commerce
store. It answers questions from a curated knowledge base of policies, FAQs and product
documentation, and grounds every answer in retrieved source material rather than free-form
generation.

The retrieval stack combines **hybrid search** — dense embeddings via ChromaDB alongside BM25
lexical matching — with an **LLM reranker** that reorders candidates by relevance before they
reach the answering model.

It is **provider-agnostic**: the agent, the guardrail filter and the retriever all run unchanged
on whichever API key you happen to have. See [Choosing a provider](#choosing-a-provider).

📐 **[Read the visual architecture walkthrough →](https://hereandnowai.github.io/nimbusmart-support-agent/)**

## Architecture

```
architecture.html  Visual architecture walkthrough (deployed to GitHub Pages)
frontend/          React 18 + TypeScript + Vite chat UI
backend/
  app/
    config.py        Environment + path configuration
    providers.py     Provider registry — Azure, OpenAI, Anthropic, Gemini, …
    kernel_setup.py  Builds the chat connector for the selected provider
    auth.py          Token-based login, customer vs. staff roles
    session.py       Per-conversation state, citations, tool calls
    filters.py       Guardrail filter (RBAC, card-data block, audit)
    rag/
      chunking.py       Markdown-header-aware splitting (700 chars, 100 overlap)
      embeddings.py     Provider embeddings, or a local ONNX fallback
      ingest.py         Builds the ChromaDB collection
      retriever.py      Hybrid search — vector + BM25 + RRF
      reranker.py       LLM-based candidate reranking
      retriever_types.py
    data/
      documents/     Knowledge base (policies, FAQs, catalog)
      orders.json    Demo order records
      users.json     Demo accounts
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | Semantic Kernel |
| LLM & embeddings | Any of 8 providers (see below) |
| Vector store | ChromaDB |
| Lexical search | BM25 (`rank-bm25`) |
| Chunking | LangChain text splitters |
| API | FastAPI + Uvicorn |
| Frontend | React 18, TypeScript, Vite |
| Package manager | `uv` (Python), `npm` (Node) |

## Getting Started

### Prerequisites

- Python 3.13+ and [`uv`](https://docs.astral.sh/uv/)
- Node.js 18+
- An API key from **any one** of the supported providers — or Ollama running locally, which
  needs no key at all

### Backend

```bash
cd backend
cp .env.example .env      # then set LLM_PROVIDER + LLM_API_KEY
uv sync
uv run python -m app.rag.ingest    # build the vector index
uv run uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Or start both at once from the project root with `./run.sh`.

### Tests and checks

This repository does not yet include an automated test suite. Run the available backend and
frontend checks before making changes:

```bash
cd backend
uv run python -m compileall app

cd ../frontend
npm install
npm run build
```

## Choosing a provider

Set two variables in `backend/.env` and everything else follows:

```bash
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
```

If `LLM_PROVIDER` is omitted, the backend auto-detects it from whichever vendor key is present
(`GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, …).

| # | Provider | `LLM_PROVIDER` | Default model | Embeddings |
|---|---|---|---|---|
| 1 | Microsoft Foundry / Azure OpenAI | `azure` | your deployment name | ✅ own |
| 2 | OpenAI | `openai` | `gpt-4o-mini` | ✅ own |
| 3 | Anthropic | `anthropic` | `claude-sonnet-5` | ↩︎ local fallback |
| 4 | Google Gemini | `gemini` | `gemini-2.5-flash` | ✅ own |
| 5 | OpenRouter | `openrouter` | `openai/gpt-4o-mini` | ↩︎ local fallback |
| 6 | Groq | `groq` | `llama-3.3-70b-versatile` | ↩︎ local fallback |
| 7 | Cerebras | `cerebras` | `llama-3.3-70b` | ↩︎ local fallback |
| 8 | Ollama (local, no key) | `ollama` | `llama3.1` | ✅ own |

**Local embedding fallback.** Anthropic, OpenRouter, Groq and Cerebras serve chat only. When one
of them is selected, embeddings fall back to the MiniLM ONNX model bundled with ChromaDB — no key
and no network call — so hybrid retrieval works with a single chat key.

Azure additionally needs `AZURE_OPENAI_ENDPOINT` (and optionally `AZURE_OPENAI_API_VERSION`).
`LLM_MODEL` and `LLM_BASE_URL` override the defaults for any provider; the agent relies on
function calling, so pick a tool-capable model.

### Other configuration

| Variable | Description |
|---|---|
| `EMBEDDING_PROVIDER` | Embed with a different vendor than chat — `azure`, `openai`, `gemini`, `ollama` or `local` |
| `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` | Per-service overrides |
| `CORS_ORIGINS` | Comma-separated origins allowed to call the API |
| `OBSERVABILITY_ENABLED`, `LANGFUSE_*` | Langfuse tracing over OpenTelemetry |

Each embedding model gets its own ChromaDB collection, so switching providers never collides with
an index built at a different vector size — **re-run the ingest command after changing it**.

`GET /api/provider` reports which vendor and model are live (never the key).

## Demo Accounts

`backend/app/data/users.json` ships with **hardcoded demo credentials** so the role-based access
logic can be exercised locally. They are sample fixtures for a fictional store — not real
accounts — and this file should be replaced with a proper user store before any real deployment.

| Username | Role |
|---|---|
| `riya`, `arjun`, `kavya` | customer |
| `admin`, `manager` | staff |

## Roadmap

- [x] Wire up `app/main.py` with FastAPI chat, login and health routes
- [x] Connect the React frontend to the live backend
- [x] Persist the ChromaDB index and add an ingestion command
- [x] Run on any LLM provider from a single environment variable
- [ ] Add tests for chunking, reranking and provider resolution
- [ ] Show the active provider in the chat UI header

## License

Released under the [MIT License](LICENSE).

---

<div align="center">

**HERE AND NOW AI**

[Website](https://hereandnowai.com) · *AI is Good*

</div>
