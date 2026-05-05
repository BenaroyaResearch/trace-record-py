from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Dict, Optional

from langchain_core.callbacks.base import AsyncCallbackHandler

"""
Lightweight, self-hosted LangChain agent tracer. Accumulates tool calls,
LLM call counts, token usage, route steps, and internal timing into a
structured dict that the caller can persist wherever they like.

Domain-specific property tracking is supported via an optional
property_extractor callable passed at construction time — see TraceRecorder.

All domain-specific logic lives in the application layer, not here.
"""

# Type alias for the optional extractor hook.
# Receives the tool input dict from an on_tool_start event.
# Returns a dict mapping property-bucket names to sets of string values.
# Example:  lambda inp: {"study_groups_used": set(inp.get("study_groups", []))}
PropertyExtractor = Callable[[Dict[str, Any]], Dict[str, set]]

_ACTIVE_TRACE: ContextVar["TraceRecorder | None"] = ContextVar("active_trace", default=None)


def set_active_trace(trace: "TraceRecorder") -> Token:
    """Set the active trace recorder in the current context."""
    return _ACTIVE_TRACE.set(trace)


def clear_active_trace(token: Token) -> None:
    """Clear the active trace recorder from the current context."""
    _ACTIVE_TRACE.reset(token)


def get_active_trace() -> "TraceRecorder | None":
    """Get the currently active trace recorder, if any."""
    return _ACTIVE_TRACE.get()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get_usage_dict(payload: Any) -> Any:
    """Extract a raw usage mapping from either a dict or object, trying three attribute paths."""
    if isinstance(payload, dict):
        candidates = [
            payload.get("usage_metadata"),
            payload.get("token_usage"),
            (payload.get("response_metadata") or {}).get("token_usage"),
        ]
    else:
        response_metadata = getattr(payload, "response_metadata", None)
        candidates = [
            getattr(payload, "usage_metadata", None),
            getattr(payload, "token_usage", None),
            response_metadata.get("token_usage") if isinstance(response_metadata, dict) else None,
        ]
    return next((c for c in candidates if c is not None), None)


def _extract_token_usage_from_payload(payload: Any) -> Optional[Dict[str, int]]:
    """Best-effort extraction of token usage from model event payloads."""
    if payload is None:
        return None

    usage = _get_usage_dict(payload)
    if not isinstance(usage, dict):
        return None

    prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion = usage.get("output_tokens", usage.get("completion_tokens"))
    total = usage.get("total_tokens")

    if prompt is None and completion is None and total is None:
        return None

    prompt_val = int(prompt) if prompt is not None else 0
    completion_val = int(completion) if completion is not None else 0
    total_val = int(total) if total is not None else prompt_val + completion_val

    return {
        "prompt": prompt_val,
        "completion": completion_val,
        "total": total_val,
    }


@dataclass
class _ToolCallRecord:
    name: str
    latency_ms: float
    status: str
    error: str | None = None


class TraceCallbackHandler(AsyncCallbackHandler):
    """Bridge LangChain callback events into the TraceRecorder schema."""

    def __init__(self, trace: "TraceRecorder"):
        self.trace = trace

    @staticmethod
    def _serialized_name(serialized: dict[str, Any] | None, default: str) -> str:
        if not isinstance(serialized, dict):
            return default

        name = serialized.get("name")
        if isinstance(name, str) and name:
            return name

        serialized_id = serialized.get("id")
        if isinstance(serialized_id, list) and serialized_id:
            tail = serialized_id[-1]
            if isinstance(tail, str) and tail:
                return tail
        if isinstance(serialized_id, str) and serialized_id:
            return serialized_id
        return default

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.trace.record_event(
            {
                "event": "on_tool_start",
                "name": self._serialized_name(serialized, default="tool"),
                "run_id": str(run_id),
                "data": {"input": inputs if isinstance(inputs, dict) else None},
            }
        )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        **kwargs: Any,
    ) -> None:
        self.trace.record_event({"event": "on_tool_end", "run_id": str(run_id)})

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        **kwargs: Any,
    ) -> None:
        self.trace.record_event({
            "event": "on_tool_error",
            "run_id": str(run_id),
            "error_message": str(error),
        })

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs: Any,
    ) -> None:
        self.trace.record_event({"event": "on_chat_model_start", "run_id": str(run_id)})

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        **kwargs: Any,
    ) -> None:
        self.trace.record_event(
            {
                "event": "on_chat_model_end",
                "run_id": str(run_id),
                "data": {"output": getattr(response, "llm_output", None)},
            }
        )


class TraceRecorder:
    """Thin request trace accumulator for agent runs.

    Records tool calls, LLM invocations, token usage, route steps, and
    arbitrary internal timing steps. Call finalize() at the end of a run
    to get a plain dict you can persist however you like.

    Args:
        trace_id:           Unique ID for this trace.
        session_id:         ID of the enclosing session (optional).
        question:           The user's input text (optional).
        username:           Identifier of the requesting user (optional).
        property_extractor: Optional hook for domain-specific property
                            tracking. Called on every on_tool_start event
                            with the tool's input dict. Should return a
                            dict mapping bucket names to sets of strings.
                            Values are accumulated across all tool calls
                            and appear in finalize() output under
                            'extracted_properties'.

                            Example::

                                def my_extractor(tool_input):
                                    groups = tool_input.get("study_groups", [])
                                    return {
                                        "study_groups_used": set(
                                            g for g in groups
                                            if isinstance(g, str) and g.strip()
                                        )
                                    }
    """

    def __init__(
        self,
        trace_id: str,
        session_id: Optional[str] = None,
        question: Optional[str] = None,
        username: Optional[str] = None,
        property_extractor: Optional[PropertyExtractor] = None,
    ):
        self.trace_id = trace_id
        self.session_id = session_id
        self.question = question
        self.username = username
        self.created_at = _utcnow()
        self._started = perf_counter()
        self._property_extractor = property_extractor

        self._tool_starts: dict[str, tuple[str, float]] = {}
        self._tool_calls: list[_ToolCallRecord] = []
        self._tool_path: list[str] = []
        self._internal_steps: list[dict[str, Any]] = []
        self._route_path: list[str] = []
        self._manual_llm_call_labels: dict[str, int] = {}
        # Accumulated sets keyed by bucket name, populated by property_extractor.
        self._extracted_properties: dict[str, set] = {}

        self.llm_calls = 0
        self._llm_calls_breakdown = {"langchain": 0, "manual": 0}
        self._token_prompt = 0
        self._token_completion = 0
        self._token_total = 0

    def _tool_run_id(self, event: dict[str, Any]) -> str:
        run_id = event.get("run_id")
        if run_id:
            return str(run_id)
        return f"{event.get('name', 'tool')}:{len(self._tool_starts) + 1}"

    def _tool_name(self, event: dict[str, Any]) -> str:
        return str(event.get("name", "tool"))

    def _accumulate_token_usage(self, usage: Optional[Dict[str, int]]) -> None:
        if not usage:
            return
        self._token_prompt += int(usage.get("prompt", 0) or 0)
        self._token_completion += int(usage.get("completion", 0) or 0)
        self._token_total += int(usage.get("total", 0) or 0)

    def _run_property_extractor(self, tool_input: Any) -> None:
        """Call the property extractor (if set) and merge results into buckets."""
        if self._property_extractor is None or not isinstance(tool_input, dict):
            return
        try:
            extracted = self._property_extractor(tool_input)
        except Exception:
            # A buggy extractor must never take down the trace.
            return
        if not isinstance(extracted, dict):
            return
        for bucket, values in extracted.items():
            if not isinstance(values, set):
                continue
            if bucket not in self._extracted_properties:
                self._extracted_properties[bucket] = set()
            self._extracted_properties[bucket].update(values)

    def record_route_step(self, node_name: str) -> None:
        """Record a node/state transition in the agent's routing logic."""
        if node_name:
            self._route_path.append(str(node_name))

    def record_internal_step(
        self,
        name: str,
        status: str = "success",
        latency_ms: float | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an arbitrary internal processing step with optional timing."""
        self._internal_steps.append(
            {
                "name": name,
                "status": status,
                "latency_ms": float(latency_ms) if latency_ms is not None else None,
                "metadata": metadata or {},
            }
        )

    def record_manual_llm_call(
        self,
        label: str,
        usage: Optional[Dict[str, int]] = None,
        latency_ms: float | None = None,
        status: str = "success",
    ) -> None:
        """Record an LLM call made outside of LangChain's callback system."""
        self.llm_calls += 1
        self._llm_calls_breakdown["manual"] += 1
        label_key = str(label or "manual")
        self._manual_llm_call_labels[label_key] = self._manual_llm_call_labels.get(label_key, 0) + 1
        self._accumulate_token_usage(usage)
        self.record_internal_step(
            name=f"llm:{label_key}",
            status=status,
            latency_ms=latency_ms,
            metadata={"source": "manual", "label": label_key},
        )

    def record_event(self, event: dict[str, Any]) -> None:
        """Process a LangChain callback event. Typically called by TraceCallbackHandler."""
        event_type = event.get("event")

        if event_type == "on_tool_start":
            tool_name = self._tool_name(event)
            run_id = self._tool_run_id(event)
            self._tool_starts[run_id] = (tool_name, perf_counter())
            # Run the domain extractor against this tool's input.
            tool_input = (event.get("data") or {}).get("input")
            self._run_property_extractor(tool_input)
            return

        if event_type == "on_tool_end":
            run_id = self._tool_run_id(event)
            tool_name, started = self._tool_starts.pop(
                run_id,
                (self._tool_name(event), perf_counter()),
            )
            latency_ms = (perf_counter() - started) * 1000.0
            self._tool_path.append(tool_name)
            self._tool_calls.append(_ToolCallRecord(tool_name, latency_ms, "success"))
            return

        if event_type == "on_tool_error":
            run_id = self._tool_run_id(event)
            tool_name, started = self._tool_starts.pop(
                run_id,
                (self._tool_name(event), perf_counter()),
            )
            latency_ms = (perf_counter() - started) * 1000.0
            error_msg = event.get("error_message")
            self._tool_path.append(tool_name)
            self._tool_calls.append(_ToolCallRecord(tool_name, latency_ms, "error", error_msg))
            return

        if event_type == "on_chat_model_start":
            self.llm_calls += 1
            self._llm_calls_breakdown["langchain"] += 1
            return

        if event_type == "on_chat_model_end":
            data = event.get("data", {})
            usage = _extract_token_usage_from_payload(data.get("output"))
            if usage is None:
                usage = _extract_token_usage_from_payload(data.get("chunk"))
            self._accumulate_token_usage(usage)

    def finalize(
        self,
        status: str,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a plain dict summarising the completed run.

        Args:
            status:   'success' or 'error'.
            error:    Error message string, if status is 'error'.
            metadata: Arbitrary caller-supplied dict stored under 'metadata'.
                      Use for domain-specific fields (e.g. group mappings).
        """
        latency_ms = (perf_counter() - self._started) * 1000.0
        token_usage = None
        if self._token_total > 0 or self._token_prompt > 0 or self._token_completion > 0:
            token_usage = {
                "prompt": self._token_prompt,
                "completion": self._token_completion,
                "total": self._token_total,
            }

        # Serialise accumulated sets to sorted lists for JSON-safety.
        extracted_properties = {
            bucket: sorted(values)
            for bucket, values in self._extracted_properties.items()
            if values
        }

        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "username": self.username,
            "question": self.question,
            "created_at": self.created_at,
            "latency_ms": latency_ms,
            "status": status,
            "tool_path": self._tool_path,
            "tool_calls": [
                {
                    "name": t.name,
                    "latency_ms": t.latency_ms,
                    "status": t.status,
                    **({"error": t.error} if t.error else {}),
                }
                for t in self._tool_calls
            ],
            "llm_calls": self.llm_calls,
            "llm_calls_breakdown": self._llm_calls_breakdown,
            "manual_llm_call_labels": self._manual_llm_call_labels,
            "token_usage": token_usage,
            "route_path": self._route_path,
            "internal_steps": self._internal_steps,
            "extracted_properties": extracted_properties,
            "metadata": metadata or {},
            "error": error,
        }
