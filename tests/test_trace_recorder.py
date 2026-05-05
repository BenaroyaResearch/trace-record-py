import time

from trace_recorder import TraceRecorder


def test_trace_recorder_tracks_tool_path_and_tokens():
    """Test that TraceRecorder correctly tracks tool execution path and token usage."""
    trace = TraceRecorder(
        trace_id="trace-1",
        session_id="session-1",
        question="Q1",
    )

    trace.record_event({"event": "on_chat_model_start"})
    trace.record_event(
        {
            "event": "on_chat_model_end",
            "data": {
                "output": {
                    "usage_metadata": {
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "total_tokens": 18,
                    }
                }
            },
        }
    )

    trace.record_event({"event": "on_tool_start", "name": "search_database", "run_id": "a"})
    time.sleep(0.001)
    trace.record_event({"event": "on_tool_end", "name": "search_database", "run_id": "a"})

    trace.record_event(
        {"event": "on_tool_start", "name": "calculate_results", "run_id": "b"}
    )
    time.sleep(0.001)
    trace.record_event(
        {"event": "on_tool_end", "name": "calculate_results", "run_id": "b"}
    )

    summary = trace.finalize(status="success")

    assert summary["trace_id"] == "trace-1"
    assert summary["session_id"] == "session-1"
    assert summary["status"] == "success"
    assert summary["llm_calls"] == 1
    assert summary["token_usage"] == {"prompt": 11, "completion": 7, "total": 18}
    assert summary["tool_path"] == [
        "search_database",
        "calculate_results",
    ]
    assert len(summary["tool_calls"]) == 2
    assert summary["tool_calls"][0]["latency_ms"] >= 0
    assert summary["latency_ms"] >= 0


def test_trace_recorder_handles_tool_error_event():
    """Test that TraceRecorder correctly handles tool errors."""
    trace = TraceRecorder(
        trace_id="trace-2",
        session_id="session-2",
        question="Q2",
    )

    trace.record_event({"event": "on_tool_start", "name": "search_database", "run_id": "x"})
    trace.record_event({
        "event": "on_tool_error",
        "name": "search_database",
        "run_id": "x",
        "error_message": "Database connection failed"
    })

    summary = trace.finalize(status="error", error="boom")

    assert summary["status"] == "error"
    assert summary["error"] == "boom"
    assert summary["tool_path"] == ["search_database"]
    assert summary["tool_calls"][0]["status"] == "error"
    assert summary["tool_calls"][0]["error"] == "Database connection failed"


def test_trace_recorder_with_property_extractor():
    """Test that property extractor correctly accumulates domain-specific properties."""
    def my_extractor(tool_input):
        """Example extractor that pulls out dataset names."""
        datasets = tool_input.get("datasets", [])
        return {
            "datasets_used": set(
                d for d in datasets
                if isinstance(d, str) and d.strip()
            )
        }

    trace = TraceRecorder(
        trace_id="trace-3",
        session_id="session-3",
        question="Q3",
        property_extractor=my_extractor,
    )

    # First tool call with some datasets
    trace.record_event({
        "event": "on_tool_start",
        "name": "query_data",
        "run_id": "a",
        "data": {
            "input": {"datasets": ["dataset1", "dataset2"]}
        }
    })
    trace.record_event({"event": "on_tool_end", "run_id": "a"})

    # Second tool call with overlapping and new datasets
    trace.record_event({
        "event": "on_tool_start",
        "name": "analyze_data",
        "run_id": "b",
        "data": {
            "input": {"datasets": ["dataset2", "dataset3"]}
        }
    })
    trace.record_event({"event": "on_tool_end", "run_id": "b"})

    summary = trace.finalize(status="success")

    assert "extracted_properties" in summary
    assert "datasets_used" in summary["extracted_properties"]
    # Should have all unique datasets, sorted
    assert summary["extracted_properties"]["datasets_used"] == ["dataset1", "dataset2", "dataset3"]


def test_trace_recorder_minimal_initialization():
    """Test that TraceRecorder works with minimal required arguments."""
    trace = TraceRecorder(trace_id="minimal-trace")

    trace.record_event({"event": "on_chat_model_start"})
    summary = trace.finalize(status="success")

    assert summary["trace_id"] == "minimal-trace"
    assert summary["session_id"] is None
    assert summary["question"] is None
    assert summary["username"] is None
    assert summary["llm_calls"] == 1
    assert summary["status"] == "success"


def test_trace_recorder_manual_llm_call():
    """Test recording manual LLM calls outside of LangChain."""
    trace = TraceRecorder(trace_id="manual-llm-trace")

    trace.record_manual_llm_call(
        label="custom_completion",
        usage={"prompt": 50, "completion": 100, "total": 150},
        latency_ms=250.5,
    )

    summary = trace.finalize(status="success")

    assert summary["llm_calls"] == 1
    assert summary["llm_calls_breakdown"]["manual"] == 1
    assert summary["llm_calls_breakdown"]["langchain"] == 0
    assert summary["manual_llm_call_labels"]["custom_completion"] == 1
    assert summary["token_usage"]["prompt"] == 50
    assert summary["token_usage"]["completion"] == 100
    assert summary["token_usage"]["total"] == 150


def test_trace_recorder_route_steps():
    """Test recording route/state transitions."""
    trace = TraceRecorder(trace_id="route-trace")

    trace.record_route_step("start")
    trace.record_route_step("analyze")
    trace.record_route_step("format_response")
    trace.record_route_step("end")

    summary = trace.finalize(status="success")

    assert summary["route_path"] == ["start", "analyze", "format_response", "end"]


def test_trace_recorder_internal_steps():
    """Test recording arbitrary internal processing steps."""
    trace = TraceRecorder(trace_id="internal-trace")

    trace.record_internal_step(
        name="data_validation",
        status="success",
        latency_ms=12.3,
        metadata={"rows_validated": 1000}
    )

    trace.record_internal_step(
        name="preprocessing",
        status="success",
        latency_ms=45.6,
        metadata={"operation": "normalization"}
    )

    summary = trace.finalize(status="success")

    assert len(summary["internal_steps"]) == 2
    assert summary["internal_steps"][0]["name"] == "data_validation"
    assert summary["internal_steps"][0]["latency_ms"] == 12.3
    assert summary["internal_steps"][0]["metadata"]["rows_validated"] == 1000
    assert summary["internal_steps"][1]["name"] == "preprocessing"
    assert summary["internal_steps"][1]["latency_ms"] == 45.6
