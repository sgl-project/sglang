# Code Structures

- `lang`: The frontend language.
- `srt`: The backend engine for running local models. (SRT = SGLang Runtime).
- `test`: The test utilities.
- `api.py`: The public APIs.
- `bench_latency.py`: Benchmark the latency of running a single static batch.
- `bench_server_latency.py`: Benchmark the latency of serving a single batch with a real server.
- `bench_serving.py`: Benchmark online serving with dynamic requests.
- `global_config.py`: The global configs and constants.
- `launch_server.py`: The entry point for launching the local server.
- `utils.py`: Common utilities.
