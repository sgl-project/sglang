# Code Structures

- `eval`: The evaluation utilities.
- `lang`: The frontend language.
- `srt`: The backend engine for running local models. (SRT = SGLang Runtime).
- `test`: The test utilities.
- `api.py`: The public APIs.
- `bench_offline_throughput.py`: Benchmark the throughput in the offline mode.
- `bench_one_batch.py`: Benchmark the latency of running a single static batch without a server.
- `bench_one_batch_server.py`: Benchmark the latency of running a single batch with a server.
- `bench_serving.py`: Benchmark online serving with dynamic requests.
- `check_env.py`: Check the environment variables and dependencies.
- `global_config.py`: The global configs and constants.
- `launch_server.py`: The entry point for launching the local server.
- `utils.py`: Common utilities.
- `version.py`: Version info.
