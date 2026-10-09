"""
OpenTelemetry Distributed Tracing Module for EcoSort.
Supports both:
1. Arize Phoenix Cloud (SaaS dashboard like Langfuse, enabled when PHOENIX_API_KEY is present)
2. Local Grafana Tempo (OTLP gRPC receiver at tempo:4317)
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logger = logging.getLogger("ecosort-tracing")

_TRACER: Optional[trace.Tracer] = None
_IS_INITIALIZED = False


def setup_tracing(service_name: str = "ecosort-app", service_version: str = "1.2.0") -> trace.Tracer:
    """
    Initialize OpenTelemetry TracerProvider.
    Automatically prioritizes Arize Phoenix Cloud if PHOENIX_API_KEY is set;
    otherwise falls back to Grafana Tempo OTLP gRPC.
    """
    global _TRACER, _IS_INITIALIZED
    if _IS_INITIALIZED and _TRACER is not None:
        return _TRACER

    phoenix_api_key = os.getenv("PHOENIX_API_KEY")
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    phoenix_project = os.getenv("PHOENIX_PROJECT_NAME", "ecosort-trash-classifier")

    # 1. Check for Arize Phoenix Cloud
    if phoenix_api_key:
        try:
            from phoenix.otel import register as phoenix_register
            provider = phoenix_register(
                project_name=phoenix_project,
                endpoint=phoenix_endpoint,
                api_key=phoenix_api_key,
                batch=False,
                set_global_tracer_provider=True,
            )
            _TRACER = trace.get_tracer(service_name, service_version)
            _IS_INITIALIZED = True
            logger.info(f"OpenTelemetry successfully registered with Arize Phoenix Cloud [Project: {phoenix_project}]")
            return _TRACER
        except Exception as exc:
            logger.warning(f"Could not connect to Arize Phoenix Cloud: {exc}. Trying fallback OTLP exporter...")

    # 2. Fallback to Local Grafana Tempo
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "tempo:4317")
    if otel_endpoint.startswith("http://"):
        otel_endpoint = otel_endpoint[7:]
    elif otel_endpoint.startswith("https://"):
        otel_endpoint = otel_endpoint[8:]

    insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() in ("true", "1", "yes")
    environment = os.getenv("ENVIRONMENT", "production")

    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        DEPLOYMENT_ENVIRONMENT: environment,
    })

    provider = TracerProvider(resource=resource)

    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otel_endpoint,
            insecure=insecure,
            timeout=3,
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=2048,
                schedule_delay_millis=1000,
            )
        )
        logger.info(f"OpenTelemetry OTLP exporter initialized -> {otel_endpoint}")
    except Exception as exc:
        logger.warning(f"Could not connect OTLP Span Exporter to {otel_endpoint}: {exc}. Using in-memory tracer.")

    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer(service_name, service_version)
    _IS_INITIALIZED = True
    return _TRACER


def instrument_fastapi_app(app):
    """
    Instrument FastAPI application with OpenTelemetry middleware.
    Excludes high-frequency health check and prometheus metrics scraping endpoints.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="metrics,health",
        )
        logger.info("FastAPI successfully instrumented with OpenTelemetry.")
    except Exception as exc:
        logger.warning(f"Failed to instrument FastAPI with OpenTelemetry: {exc}")


def get_current_trace_and_span_ids() -> Tuple[Optional[str], Optional[str]]:
    """
    Extract hex-encoded trace_id and span_id from the active OpenTelemetry context.
    Returns (None, None) if no active span is found.
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        ctx = span.get_span_context()
        return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"
    return None, None


@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager to easily trace internal ML stages (preprocessing, model inference, GradCAM).
    Example:
        with trace_span("model.inference", {"model_name": "resnet50"}) as span:
            output = model.predict(data)
    """
    global _TRACER
    if _TRACER is None:
        _TRACER = trace.get_tracer("ecosort-app")

    with _TRACER.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
            raise


def flush_tracing(timeout_millis: int = 3000):
    """Force flush active spans to collector"""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        try:
            provider.force_flush(timeout_millis)
        except Exception:
            pass
