# Upstream PR Draft (short / perf-focused)

**From:** `JunlinW113/sglang` branch `feature/multi-detokenizer-dp-routing` → `sgl-project/sglang` `main`  
**Open:** https://github.com/sgl-project/sglang/compare/main...JunlinW113:sglang:feature/multi-detokenizer-dp-routing

---

## Title

```
perf: shard MultiDetokenizerRouter by DP rank (#24944 follow-up)
```

---

## Body (copy below)

## Motivation

#24944 added parallel detokenizer workers, but a **single** `MultiDetokenizerRouter` still handles **all** DP schedulers. Under high QPS or long generations, that process becomes a bottleneck: every output batch goes through one ZMQ recv loop, `_handle_output_by_index`, and fan-out.

This PR adds `--detokenizer-router-sharding dp_rank` to run **one router per DP rank** (scheduler ↔ router 1:1). Default stays `single` (#24944 behavior).

Related: #24944

## Modifications

- New flag: `--detokenizer-router-sharding {single,dp_rank}` (default: `single`)
- `dp_rank`: `dp_size` routers; each scheduler uses `PortArgs.scheduler_detokenizer_ipc(dp_rank)`
- Launch path in `engine._launch_detokenizer_subprocesses`; IPC wiring in DP controllers (mp + Ray)
- No model/math changes — IPC routing only

## Accuracy Tests

N/A

## Speed Tests and Profiling

**Problem:** centralized router serializes scheduler→detokenizer handoff at scale.

**Mitigation:** partition router traffic by DP rank so each scheduler only contends on its own router.

| Config | Integration test | Result |
|--------|------------------|--------|
| `single` + 4 detokenizer workers | `test_multi_detokenizer_ttft` (100× 4096/2048) | 100/100 OK |
| `dp_rank`, `dp_size=3` | `test_dp_rank_router_ttft` (32× 512/128) | 32/32 OK |

Unit: `test_multi_detokenizer_router_unit.py` — 24/24 PASS (CPU).

```bash
python3 test/registered/tokenizer/test_multi_detokenizer_router_unit.py -v
python3 test/registered/tokenizer/test_multi_detokenizer.py::TestMultiDetokenizerDpRankSharding -v  # >=3 GPUs
```

Example:
```bash
--dp-size 3 --detokenizer-worker-num 4 --detokenizer-router-sharding dp_rank
```

Throughput A/B (`single` vs `dp_rank`) on maintainer hardware — happy to add if requested.

## Checklist

- [x] Unit + integration tests
- [x] Default `single` (no perf regression for existing users)
- [x] black + isort on changed files (local)
- [ ] Full `pre-commit` (hooks fetch blocked offline; ruff not installed locally)
- [ ] Docs for new flag

/cc @ShangmingCai @yhyang201 (#24944)
