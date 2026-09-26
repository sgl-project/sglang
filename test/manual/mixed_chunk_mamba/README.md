# Mixed chunk prefill + mamba radix cache checkpoint probe (#39342)

Manual GPU reproduction for [sgl-project/sglang#39342](https://github.com/sgl-project/sglang/issues/39342):
with `--enable-mixed-chunk` and `--mamba-radix-cache-strategy extra_buffer`, a
prefill that is co-batched with running decodes claimed a ping-pong slot and a
donate depth but never wrote the checkpoint, so later prefix hits restored
stale state. Not part of CI.

## Layout

| file | role |
|---|---|
| `probe_instrumentation.patch` | log-only patch: `[39342] claim / mixed / forward / donate / donate-result / restore` lines with a per-slot fingerprint (five statistics of the first mamba layer's SSM and conv state, read after the forward's result sync). Applies with `git apply -C1` on `242d8a70c` and on the fix. |
| `lifecycle_probe.py` | one deterministic phase: `--phase mixed` keeps K streaming decoders busy while a ~4.8k-token shared-prefix prefill P, a prefix-sharing R and an identical P2 are sent; `--phase ref` sends the same three requests alone. Parses the log into per-request chains. |
| `compare_probe.py` | joins the two phases by checkpoint depth and prints, per depth, whether P's claim was tracked in a forward carrying one-token tails, and whether the donated / restored fingerprints match the reference. |
| `shared_prefix_harness.py` | statistical harness: c1 reference, c1 warm repeat (noise floor), then repeated c32 cold/warm passes with `/flush_cache`; mismatches vs the reference, vs the known facts, pairwise between c32 passes, latency percentiles, prefill-log counters. |
| `server_lib.sh` | sourced by the drivers: `start_server` refuses an occupied port, launches one arm under `setsid`, and takes the server's pid / pgrp / session from a child-side handshake written inside the new session (validated as a fresh session leader that is not the caller's group); `stop_server` signals exactly that group and refuses the caller's; EXIT / INT / TERM traps and the startup timeout call it. |
| `run_probe.sh`, `run_harness.sh` | drive the scripts against one arm; results and logs land under `$WS`. |
| `test_compare_probe.py` | CPU checks that the comparator reports missing or invalid data as `incomplete`, never as equality. |

## Prerequisites

A hybrid GDN model (validated with `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), one GPU, a source checkout of
sglang with the patch applied:

```bash
git apply -C1 test/manual/mixed_chunk_mamba/probe_instrumentation.patch
export MODEL_PATH=/path/to/Qwen3.5-4B WS=/tmp/mixed_chunk_mamba
```

## Arms

The drivers take a free-form label (e.g. `base` / `fix`; the checkout is
whatever the working tree holds), an arm and a graph mode:

| arm | flags |
|---|---|
| A | `--enable-mixed-chunk --mamba-radix-cache-strategy extra_buffer` |
| B | `--mamba-radix-cache-strategy extra_buffer` |
| C | `--enable-mixed-chunk --disable-radix-cache` |

`eager` = `--disable-prefill-cuda-graph --disable-decode-cuda-graph`;
`graph` = `--cuda-graph-backend-prefill breakable` (the only prefill graph
backend accepted with mixed chunk). Common flags pin the Triton GDN and
Triton attention backends, `--sampling-backend pytorch`,
`--mamba-ssm-dtype float32`, `--chunked-prefill-size 2048`.

## Running

```bash
bash test/manual/mixed_chunk_mamba/run_probe.sh base A eager     # mixed phase + reference phase + compare
bash test/manual/mixed_chunk_mamba/run_harness.sh base A graph 200 3 32
```

Depth alignment: a mixed prefill's budget is `chunked_prefill_size -
running_bs`, so with K decoders P chunks at 2040 tokens (checkpoint depths
1984, 4024, 4784 for the default prefix). `run_probe.sh` therefore launches
the reference server with `--chunked-prefill-size 2048-K` (2048 for arm B,
whose budget is never reduced) so the reference checkpoints sit at the same
depths and the fingerprints are comparable.

## What to look for

- `[39342] mixed ... mask=[True, False, ...]` and `[39342] forward ...
  one_token_rows=K ... tracked=[(pool, slot, seqlen)]`: P's chunk was tracked
  inside a forward that also carried the K decode tails (on the pre-fix code
  the mask is `None` and `tracked=None`).
- `compare_probe.py`: `status` must be `complete` (every claimed depth has a
  valid five-value fingerprint on both sides and both restores exist);
  `incomplete` lists the reasons and carries no verdict. Then
  `mixed_equals_ref` per depth and `content_equal` for the R / P2 restores.
  On the pre-fix code the mixed-phase fingerprints are all zero (fresh,
  never-written slots) while the reference is not, so the verdict is
  `mismatch`; with the radix cache off nothing is donated and the result is
  `incomplete` by design.
- Harness: `mismatch_vs_ref` on the c32 passes and the pairwise mismatches.
  The c1 warm repeat and arms B / C give the noise floor of the workload.

The fingerprint is five scalars of one layer's state; equality is strong
evidence that the slot was written by the same computation, not a proof of
tensor equality across all layers.
