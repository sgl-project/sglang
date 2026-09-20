# Code Structure

## Folders

- `benchmark/`: Benchmark implementations and dataset utilities.
- `cli/`: Command-line interface commands and entrypoints.
- `kernels/`: Kernel interfaces, implementations, selection, and debugging utilities shared by the runtimes.
- `lang/`: Deprecated language frontend that is no longer actively maintained.
- `multimodal_gen/`: Core runtime for image, video, and audio generation models, most of which are diffusion models.
- `srt/`: Core runtime for autoregressive language models. (SRT = SGLang Runtime.)
- `test/`: Shared test and evaluation utilities.

## Files

- `README.md`: This package structure overview.
- `__init__.py`: Package initialization and public Python APIs.
- `bench_offline_throughput.py`: Deprecated wrapper for the offline throughput benchmark.
- `bench_one_batch.py`: Deprecated wrapper for the one-batch benchmark.
- `bench_one_batch_server.py`: Deprecated wrapper for the server-based one-batch benchmark.
- `bench_serving.py`: Deprecated wrapper for the online serving benchmark.
- `check_env.py`: Environment and dependency diagnostics.
- `compile_deep_gemm.py`: DeepGEMM kernel precompilation entrypoint.
- `launch_server.py`: Compatibility entrypoint for launching an inference server.
- `profiler.py`: Client entrypoint for collecting server profiling traces.
- `utils.py`: Common package utilities.
- `version.py`: Public package version resolution.
