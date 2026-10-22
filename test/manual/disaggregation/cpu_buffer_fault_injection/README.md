# Cross-node CPU-buffer fault injection

Use isolated prefill/decode test servers with the same model and TP layout.
This helper instruments the Python CPU-buffer allocator and NIXL submission
boundary; it does not modify production source or inject network packet loss.

On both hosts, prepend this directory to `PYTHONPATH` before launching the
server and choose a writable directory for the worker event logs:

```bash
export PYTHONPATH="$PWD/test/manual/disaggregation/cpu_buffer_fault_injection:$PWD/python:$PWD"
export SGLANG_CPU_FAULT_DIR=/tmp/sglang-cpu-buffer-fault
mkdir -p "$SGLANG_CPU_FAULT_DIR"
rm -f "$SGLANG_CPU_FAULT_DIR/arm"
```

Launch the servers with `--nixl-use-cpu-buffer` and checksum verification.
After readiness, run the following cases separately:

| Host to arm | File content | Injected failure |
| --- | --- | --- |
| Decode | `allocate` | Receive-buffer allocation raises `BufferError` |
| Prefill | `allocate` | Send-buffer allocation raises `BufferError` |
| Prefill | `post` | `transfer()` submits a real NIXL write, then raises |

For example, on the selected host:

```bash
printf '%s' allocate > "$SGLANG_CPU_FAULT_DIR/arm"
```

Submit the same request directly to both `/generate` endpoints, with a shared
`rid`, `bootstrap_host`, `bootstrap_port`, and unique `bootstrap_room`. Bypassing
the router keeps retries from hiding the injected request failure. Check that
the decode response reports an abort/error and that the worker JSONL logs record
an `injected` event. Remove `arm` before issuing recovery requests:

```bash
rm "$SGLANG_CPU_FAULT_DIR/arm"
```

Run four more paired requests with fresh request and room IDs. Require all four
to succeed, no checksum mismatch, empty disaggregation queues, and zero pool
`allocations`, `bytes`, and `pending` counts on every rank after cleanup. For
`post`, the `send_exception` event must show retained allocations and pending
handles before cleanup. This checks ownership after a posted write, rather than
only failure before submission.

Each process injects each fault kind once per lifetime. Restart the isolated
servers to repeat a kind. A subsequent server launch without this directory on
`PYTHONPATH` and without `SGLANG_CPU_FAULT_DIR` runs without instrumentation.
