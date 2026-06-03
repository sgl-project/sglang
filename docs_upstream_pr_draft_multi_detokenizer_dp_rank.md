# Upstream PR Draft

**From:** `JunlinW113/sglang` → `sgl-project/sglang`  
**Branch:** `feature/multi-detokenizer-dp-routing` → `main`  
**Open:** https://github.com/sgl-project/sglang/compare/main...JunlinW113:sglang:feature/multi-detokenizer-dp-routing

---

## Title

```
perf: shard MultiDetokenizerRouter by DP rank (#24944 follow-up)
```

---

## Body (copy everything below into the PR description)

## Motivation

#24944 added parallel detokenizer workers via `--detokenizer-worker-num` and `MultiDetokenizerRouter`. With the default layout, a **single** router still receives output from **all** DP schedulers. Under high request rate or long output sequences, that router serializes work on one event loop (ZMQ receive, `_handle_output_by_index` for batches, and fan-out to workers), which can become a throughput bottleneck.

This PR introduces `--detokenizer-router-sharding dp_rank` to launch **one router per DP rank**, binding each scheduler to its own router IPC. The default remains `single`, preserving #24944 behavior.

Related: #24944

## Modifications

- Add CLI flag `--detokenizer-router-sharding {single,dp_rank}` (default: `single`).
- In `dp_rank` mode (when `detokenizer_worker_num > 1` and `dp_size > 1`):
  - Allocate `detokenizer_router_ipc_names` in `PortArgs.init_new`.
  - Route each DP scheduler through `PortArgs.scheduler_detokenizer_ipc(dp_rank)`.
  - Launch `detokenizer_router_{i}` subprocesses in `Engine._launch_detokenizer_subprocesses`.
- Update DP controller wiring (`data_parallel_controller.py`, `ray/data_parallel_controller.py`).
- No changes to model forward, sampling, or KV cache (IPC routing only).

## Accuracy Tests

Not applicable. This change does not affect model outputs.

## Speed Tests and Profiling

**Issue:** A single `MultiDetokenizerRouter` centralizes scheduler-to-detokenizer traffic.

**Change:** `dp_rank` sharding splits router load across `dp_size` processes so each scheduler talks only to its paired router. Detokenizer workers are still selected by `http_worker_ipc` hashing as in #24944.

| Configuration | Test | Result |
|---------------|------|--------|
| `single`, 4 detokenizer workers | `TestMultiDetokenizer.test_multi_detokenizer_ttft` (100 prompts, 4096/2048 tokens) | 100/100 requests succeeded |
| `dp_rank`, `dp_size=3` | `TestMultiDetokenizerDpRankSharding` (generation + TTFT benchmark) | 32/32 requests succeeded |

Unit tests: `test/registered/tokenizer/test_multi_detokenizer_router_unit.py` (24 cases, CPU).

```bash
python3 test/registered/tokenizer/test_multi_detokenizer_router_unit.py -v
python3 test/registered/tokenizer/test_multi_detokenizer.py::TestMultiDetokenizer -v
python3 test/registered/tokenizer/test_multi_detokenizer.py::TestMultiDetokenizerDpRankSharding -v  # requires >= 3 GPUs
```

Example launch:

```bash
python3 -m sglang.launch_server \
  --model-path <model> \
  --dp-size 3 \
  --tokenizer-worker-num 4 \
  --detokenizer-worker-num 4 \
  --detokenizer-router-sharding dp_rank
```

Formal throughput comparison (`single` vs `dp_rank`) on project CI hardware can be added if maintainers request it.

## Checklist

- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Default `detokenizer_router_sharding=single` to avoid changing existing deployments.
- [x] Format changed Python files with black and isort locally.
- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit) (full hook run pending network access to fetch hooks).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations) (`docs_new/docs/advanced_features/server_arguments.mdx`).
- [ ] Provide formal speed benchmark results according to [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed) (optional follow-up).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

/cc @ShangmingCai @yhyang201 — reviewers of #24944; feedback welcome on the sharding design.

Ready for CI when appropriate (e.g. `/tag-and-rerun-ci`).
