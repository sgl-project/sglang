# Production Request Tracing

SGLang exports request trace data based on the OpenTelemetry Collector. You can enable tracing by adding the `--enable-trace` and configure the OpenTelemetry Collector endpoint using `--otlp-traces-endpoint` when launching the server.

You can find example screenshots of the visualization in https://github.com/sgl-project/sglang/issues/8965.

## Setup Guide
This section explains how to configure the request tracing and export the trace data.
1. Install the required packages and tools
    * install Docker and Docker Compose
    * install the dependencies
    ```bash
    # enter the SGLang root directory
    pip install -e "python[tracing]"

    # or manually install the dependencies using pip
    pip install opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp opentelemetry-exporter-otlp-proto-grpc
    ```

2. Launch OpenTelemetry collector and Jaeger
    ```bash
    docker compose -f examples/monitoring/tracing_compose.yaml up -d
    ```

3. Start your SGLang server with tracing enabled
    ```bash
    # set env variables
    export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500
    export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64
    # start the prefill and decode server
    python -m sglang.launch_server --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <other option>
    # start the model-gate-way
    python -m sglang_router.launch_router --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <other option>
    ```

    Replace `0.0.0.0:4317` with the actual endpoint of the OpenTelemetry collector. If you launched the openTelemetry collector with tracing_compose.yaml, the default receiving port is 4317.

    To use the HTTP/protobuf span exporter, set the following environment variable and point to an HTTP endpoint, for example, `http://0.0.0.0:4318/v1/traces`.
    ```bash
    export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
    ```


4. Raise some requests
5. Observe whether trace data is being exported
    * Access port 16686 of Jaeger using a web browser to visualize the request traces.
    * The OpenTelemetry Collector also exports trace data in JSON format to /tmp/otel_trace.json. In a follow-up patch, we will provide a tool to convert this data into a Perfetto-compatible format, enabling visualization of requests in the Perfetto UI.

6. Dynamically adjust trace level
    The trace level accepts configurable values from `0` to `3`. The meanings of different trace level values are as follows:
    ```
    0: disable tracing
    1: Trace important slices
    2: Trace all slices except nested ones
    3: Trace all slices
    ```
    The trace level can be dynamically set via HTTP API, for example:
    ```bash
    curl http://0.0.0.0:30000/set_trace_level?level=2
    ```
    Replace `0.0.0.0:30000` with your actual server address, and replace `level=2` with the level you want to set.

    **Note**: You must set the parameter `--enable-trace`; otherwise, the trace capability will not be enabled regardless of any dynamic adjustments to the trace level.

## How to add Tracing for slices you're interested in?(API introduction)
We have already inserted instrumentation points in the tokenizer and scheduler main threads. If you wish to trace additional request execution segments or perform finer-grained tracing, please use the APIs from the tracing package as described below.

**All of the following implementations are done in python/sglang/srt/observability/req_time_stats.py. If you want to add another slice, please do it here.**

1. Initialization

    Every process involved in tracing during the initialization phase should execute:
    ```python
    process_tracing_init(otlp_traces_endpoint, server_name)
    ```
    The otlp_traces_endpoint is obtained from the arguments, and you can set server_name freely, but it should remain consistent across all processes.

    Every thread involved in tracing during the initialization phase should execute:
    ```python
    trace_set_thread_info("thread label", tp_rank, dp_rank)
    ```
    The "thread label" can be regarded as the name of the thread, used to distinguish different threads in the visualization view.

2. Create a trace context for a request
    Each request needs to call `TraceReqContext()` to initialize a request context, which is used to generate slice spans and record request stage info. You can either store it within the request object or maintain it as a global variable.

3. Mark the beginning and end of a request
    ```
    trace_ctx.trace_req_start().
    trace_ctx.trace_req_finish()
    ```
    trace_req_start() and trace_req_finish() must be called within the same process, for example, in the tokenizer.

4. Add tracing for a slice

    * Add slice tracing normally:
        ```python
        trace_ctx.trace_slice_start(RequestStage.TOKENIZER.stage_name)
        trace_ctx.trace_slice_end(RequestStage.TOKENIZER.stage_name)

        or
        trace_ctx.trace_slice(slice: TraceSliceContext)
        ```

    - The end of the last slice in a thread must be marked with thread_finish_flag=True, or explicitly call trace_ctx.abort(); otherwise, the thread's span will not be properly generated.
        ```python
        trace_ctx.slice_end(RequestStage.D.stage_name, thread_finish_flag = True)
        trace_ctx.abort()
        ```

5. When the request execution flow transfers to another thread, the thread context needs to be explicitly rebuilt.
    - receiver: Execute the following code after receiving the request via ZMQ
        ```python
        trace_ctx.rebuild_thread_context()
        ```

## How to Extend the Tracing Framework to Support Complex Tracing Scenarios

The currently provided tracing package still has potential for further development. If you wish to build more advanced features upon it, you must first understand its existing design principles.

The core of the tracing framework's implementation lies in the design of the span structure and the trace context. To aggregate scattered slices and enable concurrent tracking of multiple requests, we have designed a three-level trace context structure or span structure: `TraceReqContext`, `TraceThreadContext` and `TraceSliceContext`. Their relationship is as follows:
```
TraceReqContext (req_id="req-123")
├── TraceThreadContext(thread_label="scheduler", tp_rank=0)
|     └── TraceSliceContext(slice_name="prefill")
|
└── TraceThreadContext(thread_label="scheduler", tp_rank=1)
      └── TraceSliceContext(slice_name="prefill")
```

Each traced request maintains a global `TraceReqContext` and creates a corresponding request span. For every thread that processes the request, a `TraceThreadContext` is recorded and a thread span is created. The `TraceThreadContext` is nested within the `TraceReqContext`, and each currently traced code slice—potentially nested—is stored in its associated `TraceThreadContext`.

In addition to the above hierarchy, each slice also records its previous slice via Span.add_link(), which can be used to trace the execution flow.
