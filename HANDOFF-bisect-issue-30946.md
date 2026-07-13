# Handoff: devbox bisect for TestDisaggregationHybridAttentionMamba GSM8K regression

Tracking issue: https://github.com/sgl-project/sglang/issues/30946
Failing CI job (2026-07-12): https://github.com/sgl-project/sglang/actions/runs/29190816712/job/86645077403

## Failure signature

- Test: `test/registered/disaggregation/test_disaggregation_hybrid_attention.py::TestDisaggregationHybridAttentionMamba::test_gsm8k`
- Model: `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (hybrid mamba), PD disaggregation:
  prefill TP4 + decode TP4 (`--base-gpu-id 4`), mooncake transfer backend, RDMA devices.
- Error: `AssertionError: 0.605 not greater than 0.87` (GSM8K score collapse; scores vary
  per attempt: 0.630 / 0.605 / 0.63 / 0.565 → nondeterministic output corruption).
- Control observations from the same failing CI jobs:
  - `TestDisaggregationHybridAttentionGDNExtraBuffer` (Qwen3-Next, same PD setup): 0.96 PASS
  - `TestDisaggregationHybridAttentionMambaExtraBuffer` (same model, `--mamba-scheduler-strategy extra_buffer`): PASS
  - Non-disagg Nemotron-Super on same runner: 0.955–0.975 PASS
  - => corruption is specific to the DEFAULT mamba scheduler strategy under PD disaggregation.

## Pass/fail boundary (scheduled `pr-test.yml` runs on main, job `extra-b-test-8-gpu-h200 (0)`)

| Date (UTC)   | Run                    | SHA         | Runner          | Result |
|--------------|------------------------|-------------|-----------------|--------|
| 07-10 12:00  | 29091187849            | `7045e0fdf` | h200-gmi-wk03   | PASS |
| 07-10 23:21  | 29130141094            | `e8646701c` | h200-rdxa-51-43 | PASS |
| 07-11 11:23  | 29150892604            | `32cb89d41` | h200-rdxa-51-41 | PASS |
| 07-11 23:19  | 29171938366            | `4884f6fbe` | h200-gmi-wk01   | PASS (0.895 / 0.900) |
| 07-12 11:26  | 29190816712            | `80856aba8` | h200-rdxa-51-3  | FAIL (0.630 → 0.605) |
| 07-12 23:19  | 29213183642 (attempt 1)| `c616d5a55` | h200-gmi-wk01   | FAIL (0.63 → 0.565) |

Failures on two different runners (one of which passed 24h earlier) => NOT runner-specific.
Deps pinned (sgl-kernel from repo pin, mooncake 0.3.11.post1, nixl 1.3.0) => NOT dep drift.

## Tight window: `4884f6fbe..80856aba8` (8 commits, none obviously related)

```
80856ab Make the mxfp8 MoE runner backend list extensible (#30828)
81d273f Handle coredump dirs and cache hit updates (#30897)
f1c247e profile: add vlm prefill profiler ranges (#30871)
bce3fc9 perf: reuse MoonViT FA3 max-seqlen metadata (#30878)
592c043 Update test repository case scripts to the main community (#29939)
a358abd chore: update vlm moe config and tune scripts (#30866)
14bef7c fix: lazy load TileLang MHC kernels (#30580)
af66370 bench: support random image resolutions (#30879)
```

Already exonerated for this recurrence (ancestors of passing sha `4884f6fbe`, i.e. present in
3+ passing runs): `fc2ef35308` (#30802), `2286e25a21` (#30636), `0299393758` (#30626),
`90688366d9` (#30737), `9b4bb415dd` (#30834).

Caveat from issue #30946: the test has one older hard failure on 06-23 (`0460f277b7`,
scores 0.845/0.720) with passes before/after, so treat it as a bimodal flake whose failure
probability jumped at the 07-12 boundary. Bisect verdicts need REPEATED runs per sha.

## Devbox reproduction plan (8x H200 required; needs RDMA/mooncake)

1. Environment: same as CI — `scripts/ci/cuda/ci_install_dependency.sh` at the sha under test,
   or a recent lmsysorg/sglang dev docker image with the sha checked out + `pip install -e "python[all]"`.
   Test needs env `SGLANG_IS_IN_CI=true` unset locally (it gates skips); check
   `test/registered/disaggregation/test_disaggregation_hybrid_attention.py` — the GDN class is
   CI-skipped, others run everywhere.
2. Single-class repro command (~6 min including two server launches):
   ```bash
   cd test
   python3 registered/disaggregation/test_disaggregation_hybrid_attention.py \
       TestDisaggregationHybridAttentionMamba
   ```
   The fixture reads RDMA devices from env (see `python/sglang/test/server_fixtures/disaggregation_fixture.py`;
   CI exports rdma_devices, e.g. `--disaggregation-ib-device mlx5_0,mlx5_6`, and MC_GID_INDEX on RoCE hosts per #30737).
3. Score extraction: the test prints `gsm8k_score=... labels={"model": "nvidia/NVIDIA-Nemotron-Nano-9B-v2"...}`.
   PASS distribution ≈ 0.88–0.93; FAIL distribution ≈ 0.56–0.65. Treat ≤0.80 as FAIL, ≥0.87 as PASS.
4. Protocol (bimodal flake ⇒ repeat): run the class 5x at `4884f6fbe` and 5x at `80856aba8`.
   - If 80856aba8 fails ≥3/5 and 4884f6fbe fails 0/5 → true code regression:
     `git bisect start 80856aba8 4884f6fbe` and run the 5x protocol per step (only 8 commits ⇒ 3 steps).
     Prioritize testing `81d273f` first (only commit touching scheduler/server_args).
   - If both shas show similar failure rates → timing-sensitive race, not in the window;
     instrument mamba state transfer instead: checksum mamba state on prefill send vs decode
     receive (`python/sglang/srt/mem_cache/...mamba...`, disagg mooncake path), and compare
     default strategy vs `--mamba-scheduler-strategy extra_buffer` on decode.
5. Controls if reproducing: TP=1 vs TP=4; `--disable-cuda-graph` on decode; radix cache off.
