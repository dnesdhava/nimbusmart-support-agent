import base64
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from app.observability.redaction import RedactingSpanProcessor

_INITIALISED = False

def _enbaled() -> bool:
    return os.getenv("OBSERVABILITY_ENABLED", "false").lower() in ("1", "true", "yes")

def setup_observability(app) -> None:
    """Called once from main.py. No-op unless enabled and Langfuse keys exist."""
    global _INITIALISED
    if _INITIALISED or not _enbaled():
        return

    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not (public and secret):
        return

    host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com").rstrip("/")
    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()

    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "nimbussupport-backend"),
            "service.version": "0.1.0"
        }
    )

    provider = TracerProvider(resource=resource, sampler=ALWAYS_ON)
    exporter = OTLPSpanExporter(
        endpoint=f"{host}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}"},
    )
    provider.add_span_processor(RedactingSpanProcessor())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(
        app, excluded_urls="api/health", exclude_spans=["send", "receive"]
    )
    HTTPXClientInstrumentor().instrument()

    _INITIALISED = True
