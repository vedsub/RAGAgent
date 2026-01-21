import os
import sys
import logging
import structlog
import time
import uuid
from typing import Optional, Dict, Any, Callable
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.semantic_conventions.ai import SpanAttributes


class ObservabilityConfig:
    """Central configuration for observability setup"""

    def __init__(self):
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "rag-evaluation")
        self.service_version = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
        self.otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.trace_sample_rate = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))

        # LLM Cost tracking (approximate pricing)
        self.llm_costs = {
            "llama-3.3-70b-versatile": {
                "input_token_cost": 0.00059,  # per 1K tokens
                "output_token_cost": 0.00079,  # per 1K tokens
            }
        }


class ObservabilityManager:
    """Manages all observability setup and provides utility functions"""

    def __init__(self, config: ObservabilityConfig = None):
        self.config = config or ObservabilityConfig()
        self.tracer = None
        self.meter = None
        self.logger = None
        self._setup_logging()
        self._setup_telemetry()

    def _setup_logging(self):
        """Setup structured logging with correlation IDs"""
        # Configure structlog for structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, self.config.log_level),
        )

        self.logger = structlog.get_logger()
        self.logger.info(
            "Logging initialized",
            service=self.config.service_name,
            level=self.config.log_level,
        )

    def _setup_telemetry(self):
        """Setup OpenTelemetry tracing and metrics"""
        resource = Resource.create(
            {
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.environment,
            }
        )

        if self.config.enable_tracing:
            # Setup Tracing
            trace_provider = TracerProvider(resource=resource)
            otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace_provider.add_span_processor(span_processor)
            trace.set_tracer_provider(trace_provider)

            # Auto-instrument LangChain and requests
            LangChainInstrumentor().instrument()
            RequestsInstrumentor().instrument()

            self.tracer = trace.get_tracer(__name__)
            self.logger.info("Tracing initialized", endpoint=self.config.otlp_endpoint)

        if self.config.enable_metrics:
            # Setup Metrics
            metric_reader = PeriodicExportingMetricReader(
                exporter=OTLPMetricExporter(endpoint=self.config.otlp_endpoint),
                export_interval_millis=30000,  # 30 seconds
            )
            meter_provider = MeterProvider(
                resource=resource, metric_readers=[metric_reader]
            )
            metrics.set_meter_provider(meter_provider)

            self.meter = metrics.get_meter(__name__)
            self._setup_instruments()
            self.logger.info("Metrics initialized", endpoint=self.config.otlp_endpoint)

    def _setup_instruments(self):
        """Create metric instruments for monitoring"""
        # LLM Metrics
        self.llm_request_counter = self.meter.create_counter(
            "llm.requests.total", description="Total number of LLM requests"
        )
        self.llm_latency_histogram = self.meter.create_histogram(
            "llm.request.duration",
            description="LLM request latency in seconds",
            unit="s",
        )
        self.llm_token_usage_counter = self.meter.create_counter(
            "llm.tokens.used", description="Total number of LLM tokens used"
        )
        self.llm_cost_counter = self.meter.create_counter(
            "llm.cost.total", description="Total LLM cost in USD", unit="USD"
        )

        # RAG Metrics
        self.rag_retrieval_counter = self.meter.create_counter(
            "rag.retrievals.total", description="Total number of RAG retrievals"
        )
        self.rag_retrieval_latency = self.meter.create_histogram(
            "rag.retrieval.duration",
            description="RAG retrieval latency in seconds",
            unit="s",
        )
        self.rag_relevance_score_histogram = self.meter.create_histogram(
            "rag.relevance.score", description="RAG context relevance scores"
        )

        # Vector Store Metrics
        self.vector_store_query_counter = self.meter.create_counter(
            "vector_store.queries.total",
            description="Total number of vector store queries",
        )
        self.vector_store_query_latency = self.meter.create_histogram(
            "vector_store.query.duration",
            description="Vector store query latency in seconds",
            unit="s",
        )

    def record_llm_request(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency: float,
        error: bool = False,
    ):
        """Record metrics for an LLM request"""
        attributes = {"model.name": model_name, "error": str(error)}

        self.llm_request_counter.add(1, attributes)
        self.llm_latency_histogram.record(latency, attributes)
        self.llm_token_usage_counter.add(
            input_tokens, {"model.name": model_name, "token_type": "input"}
        )
        self.llm_token_usage_counter.add(
            output_tokens, {"model.name": model_name, "token_type": "output"}
        )

        # Calculate cost
        cost = self._calculate_llm_cost(model_name, input_tokens, output_tokens)
        self.llm_cost_counter.add(cost, {"model.name": model_name})

    def record_rag_retrieval(self, num_results: int, latency: float):
        """Record metrics for RAG retrieval"""
        self.rag_retrieval_counter.add(1)
        self.rag_retrieval_latency.record(latency)
        self.logger.info(
            "RAG retrieval completed", num_results=num_results, latency_seconds=latency
        )

    def record_relevance_score(self, score: float):
        """Record relevance score from evaluation"""
        self.rag_relevance_score_histogram.record(score)

    def record_vector_store_query(self, latency: float, num_results: int = None):
        """Record metrics for vector store query"""
        self.vector_store_query_counter.add(1)
        self.vector_store_query_latency.record(latency)
        if num_results:
            self.logger.debug(
                "Vector store query completed",
                num_results=num_results,
                latency_seconds=latency,
            )

    def _calculate_llm_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate approximate cost for LLM request"""
        if model_name not in self.config.llm_costs:
            return 0.0

        costs = self.config.llm_costs[model_name]
        input_cost = (input_tokens / 1000) * costs["input_token_cost"]
        output_cost = (output_tokens / 1000) * costs["output_token_cost"]
        return input_cost + output_cost


# Global observability instance
_obs_manager = None


def get_observability_manager() -> ObservabilityManager:
    """Get the global observability manager instance"""
    global _obs_manager
    if _obs_manager is None:
        _obs_manager = ObservabilityManager()
    return _obs_manager


def get_logger():
    """Get the structured logger"""
    return get_observability_manager().logger


def get_tracer():
    """Get the OpenTelemetry tracer"""
    return get_observability_manager().tracer


def get_meter():
    """Get the OpenTelemetry meter"""
    return get_observability_manager().meter
