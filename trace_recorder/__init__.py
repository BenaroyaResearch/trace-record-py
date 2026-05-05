"""
trace-recorder: Lightweight, self-hosted LangChain agent tracer.

Accumulates tool calls, LLM call counts, token usage, route steps, and
internal timing into a structured dict that the caller can persist wherever
they like.

Domain-specific property tracking is supported via an optional property_extractor
callable. All domain-specific logic lives in the application layer, not here.
"""

from .trace_logger import (
    PropertyExtractor,
    TraceCallbackHandler,
    TraceRecorder,
    clear_active_trace,
    get_active_trace,
    set_active_trace,
)

__all__ = [
    "PropertyExtractor",
    "TraceCallbackHandler",
    "TraceRecorder",
    "clear_active_trace",
    "get_active_trace",
    "set_active_trace",
]

__version__ = "0.1.0"
