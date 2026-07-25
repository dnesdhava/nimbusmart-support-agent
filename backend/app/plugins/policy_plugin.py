from semantic_kernel.functions import kernel_function
from app.rag.retriever import get_retriever
from app.session import ChatSession

class PolicyPlugin:
    def __init__(self, session: ChatSession) -> None:
        self._session = session

    @kernel_function(
        description=(
            "Search NimbusMart Policies, FAQ, product specs, and troubleshooting guides."
            "guides for information releveant to the customer's question."
            )
            )

    async def search(self, query: str) -> str:
        hits = await get_retriever().search(query)
        if not hits:
            return "NO_MATCH"
        for hit in hits:
            self._session.citations.append({"source": hit.source, "section": hit.section})
        return "\n\n".join(f"{hit.source} {hit.text}" for hit in hits)