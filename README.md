# trace-recorder

**Lightweight, self-hosted LangChain agent tracer.** Capture tool calls, LLM usage, token counts, and timing without external dependencies or API keys.

## Why trace-recorder?

LangSmith is great, but sometimes you need observability that:
- **Stays local** — no data egress, no external API keys
- **Persists where you want** — MongoDB, Postgres, S3, wherever
- **Costs nothing** — no metered pricing
- **Works in restricted environments** — HIPAA, air-gapped, institutional compliance

`trace-recorder` bridges LangChain's callback system into a structured dict you can persist however you like.

## Installation

```bash
uv pip install trace-recorder
```

Or install in development mode:

```bash
uv pip install -e /path/to/trace-recorder
```

Alternatively, using pip:

```bash
pip install trace-recorder
# or for development:
pip install -e /path/to/trace-recorder
```

## Quick Start

```python
from trace_recorder import TraceRecorder, TraceCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Create a trace recorder
trace = TraceRecorder(
    trace_id="unique-trace-id",
    session_id="session-123",
    question="What is the capital of France?",
    username="alice",
)

# Create the callback handler
callback = TraceCallbackHandler(trace)

# Use it with your LangChain components
llm = ChatOpenAI(callbacks=[callback])
response = llm.invoke([HumanMessage(content="What is the capital of France?")])

# Finalize and get the trace summary
summary = trace.finalize(status="success")

# summary is a plain dict with:
# - trace_id, session_id, username, question
# - created_at, latency_ms, status
# - tool_path, tool_calls (with timing)
# - llm_calls, token_usage
# - route_path, internal_steps
# - extracted_properties (from property_extractor)
# - metadata (custom fields)

# Persist it however you want
print(summary)
# or: await db.traces.insert_one(summary)
# or: save_to_postgres(summary)
# or: s3.put_object(Body=json.dumps(summary), ...)
```

## Features

### 🎯 Tool Call Tracking

Automatically captures every tool invocation with timing:

```python
summary["tool_path"]  # ["search_database", "calculate_results", "format_output"]
summary["tool_calls"]
# [
#   {"name": "search_database", "latency_ms": 123.4, "status": "success"},
#   {"name": "calculate_results", "latency_ms": 45.6, "status": "success"},
#   {"name": "format_output", "latency_ms": 12.3, "status": "error", "error": "..."}
# ]
```

### 💰 Token Usage Tracking

Best-effort extraction from OpenAI, Anthropic, and other providers:

```python
summary["token_usage"]
# {"prompt": 150, "completion": 75, "total": 225}

summary["llm_calls"]  # 3
summary["llm_calls_breakdown"]  # {"langchain": 2, "manual": 1}
```

### 🔌 Domain-Specific Property Extraction

Add a custom `property_extractor` to pull domain-specific data from tool inputs:

```python
def extract_datasets(tool_input):
    """Extract dataset names from tool calls."""
    datasets = tool_input.get("datasets", [])
    return {
        "datasets_used": set(d for d in datasets if isinstance(d, str))
    }

trace = TraceRecorder(
    trace_id="trace-123",
    property_extractor=extract_datasets,
)

# After tools run with {"datasets": ["study1", "study2"]}
summary["extracted_properties"]
# {"datasets_used": ["study1", "study2"]}  # sorted list
```

### 📍 Route & Internal Step Tracking

Record agent state transitions and custom processing steps:

```python
trace.record_route_step("start")
trace.record_route_step("analyze")
trace.record_route_step("respond")

trace.record_internal_step(
    name="data_validation",
    status="success",
    latency_ms=12.3,
    metadata={"rows": 1000}
)

summary["route_path"]  # ["start", "analyze", "respond"]
summary["internal_steps"]
# [{"name": "data_validation", "status": "success", "latency_ms": 12.3, "metadata": {...}}]
```

### 🎛️ Manual LLM Call Recording

For LLM calls outside of LangChain's callback system:

```python
trace.record_manual_llm_call(
    label="custom_completion",
    usage={"prompt": 50, "completion": 100, "total": 150},
    latency_ms=250.5,
)
```

### 🌍 Context Variable Support

Use `set_active_trace()` to make a trace available in the current async context:

```python
from trace_recorder import set_active_trace, get_active_trace, clear_active_trace

trace = TraceRecorder(trace_id="ctx-trace")
token = set_active_trace(trace)

# Later, in any nested function...
active = get_active_trace()
if active:
    active.record_internal_step("nested_operation", status="success")

# Clean up when done
clear_active_trace(token)
```

## API Reference

### `TraceRecorder`

Main accumulator class.

**Constructor:**
```python
TraceRecorder(
    trace_id: str,               # Required: unique ID for this trace
    session_id: str | None = None,
    question: str | None = None,
    username: str | None = None,
    property_extractor: PropertyExtractor | None = None,
)
```

**Methods:**
- `record_event(event: dict)` — Process LangChain callback event (usually via `TraceCallbackHandler`)
- `record_route_step(node_name: str)` — Record a state/node transition
- `record_internal_step(name, status="success", latency_ms=None, metadata=None)` — Record custom step
- `record_manual_llm_call(label, usage=None, latency_ms=None, status="success")` — Record non-LangChain LLM call
- `finalize(status: str, error: str | None = None, metadata: dict | None = None) -> dict` — Return summary dict

### `TraceCallbackHandler`

LangChain async callback handler.

```python
from langchain_core.callbacks import AsyncCallbackHandler

callback = TraceCallbackHandler(trace)
# Pass to LangChain components via callbacks=[callback]
```

### `PropertyExtractor`

Type alias for the property extractor hook:

```python
PropertyExtractor = Callable[[Dict[str, Any]], Dict[str, set]]
```

Example:
```python
def my_extractor(tool_input: dict) -> dict[str, set]:
    return {
        "datasets_used": set(tool_input.get("datasets", []))
    }
```

## Persistence Examples

### MongoDB (Motor)

```python
from motor.motor_asyncio import AsyncIOMotorClient

client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client.my_database

async def save_trace(trace: TraceRecorder, status: str):
    summary = trace.finalize(status=status)
    await db.analysis_traces.insert_one(summary)
```

### PostgreSQL (asyncpg)

```python
import asyncpg
import json

async def save_trace_pg(trace: TraceRecorder, status: str):
    summary = trace.finalize(status=status)
    conn = await asyncpg.connect('postgresql://localhost/mydb')
    await conn.execute(
        "INSERT INTO traces (trace_id, data) VALUES ($1, $2)",
        summary["trace_id"],
        json.dumps(summary),
    )
    await conn.close()
```

### S3 / Object Storage

```python
import boto3
import json
from datetime import datetime

s3 = boto3.client('s3')

def save_trace_s3(trace: TraceRecorder, status: str):
    summary = trace.finalize(status=status)
    key = f"traces/{datetime.utcnow().date()}/{summary['trace_id']}.json"
    s3.put_object(
        Bucket='my-traces-bucket',
        Key=key,
        Body=json.dumps(summary),
    )
```

## Testing

```bash
uv run pytest
```

Or with coverage:

```bash
uv run pytest --cov=trace_recorder --cov-report=term-missing
```

## License

MIT

## Contributing

Contributions welcome! This is a small, focused library. Please keep PRs scoped and well-tested.

## Roadmap

- [ ] Streaming trace output (for long-running agents)
- [ ] Built-in formatters (JSON, MessagePack, Parquet)
- [ ] Optional structured output validation (Pydantic models)
- [ ] Example integrations (FastAPI, LangGraph, LangServe)

## Related Projects

- **LangSmith** — Hosted tracing from LangChain (SaaS)
- **Phoenix** — Open-source LLM observability by Arize
- **Helicone** — Hosted LLM observability

`trace-recorder` is for when you need something simpler, local, and with zero external dependencies.
