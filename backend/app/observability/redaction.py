import re
from opentelemetry.sdk.trace import SpanProcessor

_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

class RedactingSpanProcessor(SpanProcessor):
    def on_start(self, span, parent_context=None) -> None:
        pass

    def on_end(self, span) -> None:
        try:
            attrs = getattr(span, "_attributes", None)
            if not attrs:
                return
            for key, value in list(attrs.items()):
                if isinstance(value, str) and _CARD.search(value):
                    attrs[key] = _CARD.sub("[REDACTED]", value)
        except Exception:
            pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True