"""Observability - OpenTelemetry Tracing."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from graphrag.core.config import get_settings


def setup_tracing() -> None:
    """Configure OpenTelemetry tracing."""
    settings = get_settings()

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: settings.otel_service_name,
        "deployment.environment": settings.app_env,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint,
        insecure=not settings.is_production,
    )
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set global tracer provider
    trace.set_tracer_provider(provider)


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance.

    Args:
        name: Tracer name (usually module name).

    Returns:
        OpenTelemetry tracer.
    """
    return trace.get_tracer(name)


def instrument_fastapi(app) -> None:
    """Instrument FastAPI with OpenTelemetry.

    Args:
        app: FastAPI application.
    """
    FastAPIInstrumentor.instrument_app(app)


# Convenience decorator for tracing functions
def traced(name: str | None = None):
    """Decorator to trace a function.

    Args:
        name: Optional span name (defaults to function name).

    Returns:
        Decorated function.
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer(__name__)
            span_name = name or func.__name__
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise

        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer(__name__)
            span_name = name or func.__name__
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
