import time
import uuid
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from .observability import get_observability_manager, get_logger


class LLMObservabilityWrapper:
    """Wrapper for adding observability to LLM calls"""

    def __init__(self):
        self.obs_manager = get_observability_manager()
        self.logger = get_logger()

    def observe_llm_call(
        self, operation_name: str, model_name: str, **additional_attributes
    ):
        """Decorator to observe LLM calls with tracing and metrics"""

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate unique request ID for correlation
                request_id = str(uuid.uuid4())

                # Start tracing span
                if self.obs_manager.tracer:
                    span = self.obs_manager.tracer.start_span(f"llm.{operation_name}")
                    span.set_attribute("request_id", request_id)
                    span.set_attribute("model.name", model_name)

                    # Add additional attributes
                    for key, value in additional_attributes.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(f"llm.{key}", value)

                # Log request start
                self.logger.info(
                    "LLM request started",
                    request_id=request_id,
                    operation=operation_name,
                    model=model_name,
                    **additional_attributes,
                )

                start_time = time.time()
                error_occurred = False
                result = None

                try:
                    # Execute the original function
                    result = func(*args, **kwargs)

                    # Extract metrics from result
                    input_tokens, output_tokens = self._extract_token_usage(
                        result, model_name
                    )
                    latency = time.time() - start_time

                    # Log success
                    self.logger.info(
                        "LLM request completed",
                        request_id=request_id,
                        operation=operation_name,
                        model=model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_seconds=latency,
                    )

                    # Record metrics
                    self.obs_manager.record_llm_request(
                        model_name=model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency=latency,
                        error=False,
                    )

                    # Add span attributes
                    if self.obs_manager.tracer and span:
                        span.set_attribute("input_tokens", input_tokens)
                        span.set_attribute("output_tokens", output_tokens)
                        span.set_attribute("latency_seconds", latency)
                        span.set_attribute("success", True)

                    return result

                except Exception as e:
                    error_occurred = True
                    latency = time.time() - start_time

                    # Log error
                    self.logger.error(
                        "LLM request failed",
                        request_id=request_id,
                        operation=operation_name,
                        model=model_name,
                        error=str(e),
                        latency_seconds=latency,
                        exc_info=True,
                    )

                    # Record error metrics
                    self.obs_manager.record_llm_request(
                        model_name=model_name,
                        input_tokens=0,
                        output_tokens=0,
                        latency=latency,
                        error=True,
                    )

                    # Add span attributes for error
                    if self.obs_manager.tracer and span:
                        span.set_attribute("latency_seconds", latency)
                        span.set_attribute("success", False)
                        span.set_attribute("error.message", str(e))
                        span.set_status(
                            status=trace.Status(
                                trace.StatusCode.ERROR, description=str(e)
                            )
                        )

                    raise

                finally:
                    # End span
                    if self.obs_manager.tracer and span:
                        span.end()

            return wrapper

        return decorator

    def _extract_token_usage(self, result: Any, model_name: str) -> tuple[int, int]:
        """Extract token usage from LLM result"""
        input_tokens = 0
        output_tokens = 0

        # Handle LangChain LLMResult
        if isinstance(result, LLMResult):
            if result.llm_output and "token_usage" in result.llm_output:
                token_usage = result.llm_output["token_usage"]
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)

        # Handle string results
        elif isinstance(result, str):
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            output_tokens = max(1, len(result) // 4)

        # Handle other types
        elif hasattr(result, "content") and isinstance(result.content, str):
            output_tokens = max(1, len(result.content) // 4)

        return input_tokens, output_tokens

    @contextmanager
    def observe_rag_pipeline(self, pipeline_name: str, query: str):
        """Context manager for observing RAG pipeline execution"""
        request_id = str(uuid.uuid4())

        # Start tracing span
        if self.obs_manager.tracer:
            span = self.obs_manager.tracer.start_span(f"rag.{pipeline_name}")
            span.set_attribute("request_id", request_id)
            span.set_attribute("query", query[:500])  # Truncate for span attribute
            span.set_attribute("pipeline.name", pipeline_name)

        self.logger.info(
            "RAG pipeline started",
            request_id=request_id,
            pipeline=pipeline_name,
            query_length=len(query),
        )

        start_time = time.time()

        try:
            yield {
                "request_id": request_id,
                "span": span if self.obs_manager.tracer else None,
            }

        finally:
            latency = time.time() - start_time

            self.logger.info(
                "RAG pipeline completed",
                request_id=request_id,
                pipeline=pipeline_name,
                latency_seconds=latency,
            )

            if self.obs_manager.tracer and span:
                span.set_attribute("latency_seconds", latency)
                span.end()

    @contextmanager
    def observe_vector_store_operation(self, operation: str, **attributes):
        """Context manager for observing vector store operations"""
        request_id = str(uuid.uuid4())

        # Start tracing span
        if self.obs_manager.tracer:
            span = self.obs_manager.tracer.start_span(f"vector_store.{operation}")
            span.set_attribute("request_id", request_id)
            span.set_attribute("operation", operation)

            for key, value in attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"vector_store.{key}", value)

        self.logger.debug(
            "Vector store operation started",
            request_id=request_id,
            operation=operation,
            **attributes,
        )

        start_time = time.time()

        try:
            yield {
                "request_id": request_id,
                "span": span if self.obs_manager.tracer else None,
            }

        finally:
            latency = time.time() - start_time

            self.logger.debug(
                "Vector store operation completed",
                request_id=request_id,
                operation=operation,
                latency_seconds=latency,
            )

            # Record metrics
            self.obs_manager.record_vector_store_query(latency)

            if self.obs_manager.tracer and span:
                span.set_attribute("latency_seconds", latency)
                span.end()


# Global wrapper instance
_llm_wrapper = None


def get_llm_wrapper() -> LLMObservabilityWrapper:
    """Get the global LLM observability wrapper"""
    global _llm_wrapper
    if _llm_wrapper is None:
        _llm_wrapper = LLMObservabilityWrapper()
    return _llm_wrapper


# Convenience functions for common patterns
def observe_llm_generation(model_name: str):
    """Decorator for LLM generation calls"""
    wrapper = get_llm_wrapper()
    return wrapper.observe_llm_call(operation_name="generation", model_name=model_name)


def observe_llm_evaluation(model_name: str):
    """Decorator for LLM evaluation calls"""
    wrapper = get_llm_wrapper()
    return wrapper.observe_llm_call(operation_name="evaluation", model_name=model_name)


def observe_llm_synthetic_generation(model_name: str):
    """Decorator for synthetic data generation calls"""
    wrapper = get_llm_wrapper()
    return wrapper.observe_llm_call(
        operation_name="synthetic_generation",
        model_name=model_name,
        purpose="dataset_generation",
    )
