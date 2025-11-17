# Observability

## Production Metrics
SGLang exposes the following metrics via Prometheus. You can enable them by adding `--enable-metrics` when launching the server.
You can query them by:
```
curl http://localhost:30000/metrics
```

See [Production Metrics](../references/production_metrics.md) for more details.

## Logging

By default, SGLang does not log any request contents. You can log them by using `--log-requests`.
You can control the verbosity by using `--log-request-level`.
See [Logging](server_arguments.md#logging) for more details.

## Request Dump and Replay

You can dump all requests and replay them later for benchmarking or other purposes.

To start dumping, use the following command to send a request to a server:
```
python3 -m sglang.srt.managers.configure_logging --url http://localhost:30000 --dump-requests-folder /tmp/sglang_request_dump --dump-requests-threshold 100
```
The server will dump the requests into a pickle file for every 100 requests.

To replay the request dump, use `scripts/playground/replay_request_dump.py`.

## Crash Dump and Replay
Sometimes the server might crash, and you may want to debug the cause of the crash.
SGLang supports crash dumping, which will dump all requests from the 5 minutes before the crash, allowing you to replay the requests and debug the reason later.

To enable crash dumping, use `--crash-dump-folder /tmp/crash_dump`.
To replay the crash dump, use `scripts/playground/replay_request_dump.py`.
