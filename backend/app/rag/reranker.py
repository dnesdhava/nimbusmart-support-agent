import re
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents import ChatHistory
from app.rag.retriever_types import Candidate

_PROMPT = """You are ranking search results for a customer support question.

Question: {query}

Candidates:
{listing}

Reply with ONLY a comma-separated list of candidate IDs, ordered from MOST to LEAST relevant
to answering the question. Include every ID exactly once. No other text.

"""

def _format_listing(candidates: list[Candidate]) -> str:
    return "\n".join(f"[{c.id}] {c.text[:280].replace(chr(10), ' ')}" for c in candidates)

def _parse_order(raw: str, valid_ids: set[str]) -> list[str]:
    found = re.findall(r"[\w./-]+::\d+", raw)
    ordered = [i for i in found if i in valid_ids]
    seen = set(ordered)
    ordered.extend(i for i in valid_ids if i not in seen)
    return ordered

async def llm_rerank(
        query: str, candidates: list[Candidate], chat_service: ChatCompletionClientBase, top_k: int = 4
) -> list[Candidate]:
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    history = ChatHistory()
    history.add_user_message(_PROMPT.format(query=query, listing=_format_listing(candidates)))
    # Each provider has its own settings class — ask the service for the right one.
    settings = chat_service.instantiate_prompt_execution_settings()

    reply = await chat_service.get_chat_message_content(history, settings)
    by_id = {c.id: c for c in candidates}
    order = _parse_order(str(reply) if reply else "", set(by_id))
    return [by_id[i] for i in order][:top_k]