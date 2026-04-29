# Source Map

Use these upstream files when the workflow or behavior needs to be justified from SGLang source.

## Profiler entrypoints

- `python/sglang/profiler.py`
  - live profiler CLI
  - writes `server_args.json`
  - forwards `num_steps`, `profile_by_stage`, `merge_profiles`, and `profile_prefix`

- `python/sglang/test/send_one.py`
  - minimal request path that can trigger profiling from a single command

- `python/sglang/bench_serving.py`
  - profile-capable serving benchmark path
  - forwards `profile_activities`, `profile_by_stage`, `profile_stages`, and `profile_prefix`

## Scheduler-side trace writing

- `python/sglang/srt/managers/scheduler_profiler_mixin.py`
  - actual trace start/stop behavior
  - filename pattern for `TP/DP/PP/EP` and optional stage suffixes
  - `CUDA_PROFILER` and torch profiler handling

- `python/sglang/srt/utils/profile_merger.py`
  - merged distributed trace behavior
  - why merged traces should be treated differently from rank-local traces

- `python/sglang/srt/utils/profile_utils.py`
  - newer profile v2 manager path used for stage-based traces

## Documentation and tests

- `docs/developer_guide/benchmark_and_profiling.md`
  - canonical profiling docs

- `test/registered/profiling/test_start_profile.py`
  - validates `/start_profile` behavior, including `CUDA_PROFILER`

- `test/registered/profiling/test_profile_v2.py`
  - validates stage-scoped trace outputs under `SGLANG_PROFILE_V2`
