# K3 on B300: measure, tune, re-measure

Two K3 kernels resolve tuned tables by `torch.cuda.get_device_name()`, and only
GB300 tables ship. On a B300 the fused SP collective is replaced by NCCL and the
MoE front runs a default heuristic. This is the procedure for quantifying that
gap on a real endpoint and closing it.

## 0. Confirm the gap exists

```bash
python3 benchmark/kernels/kimi_k3/check_tuned_configs.py
```

Reports the raw and normalized device name and, per table, `TUNED` or `MISS`.
Everything below assumes at least one `MISS`. Run it on the GB300 fleet too --
if that reports `MISS`, those tables have never been live and that is the
bigger finding.

## 1. Baseline the endpoint

The tuned tables affect the MoE front and the SP collectives. Their share of
step time bounds the achievable end-to-end gain, so measure the share before
spending node time.

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/k3prof/baseline
curl -s localhost:30000/start_profile -H 'content-type: application/json' \
  -d '{"num_steps": 20, "activities": ["CPU","GPU"], "profile_by_stage": true, "record_shapes": true}'
# drive representative load here; profiling auto-stops after num_steps
```

`num_steps` auto-stops, so no `/stop_profile` is needed. `profile_by_stage`
separates prefill from decode -- necessary, because the MoE front and the
collectives sit in different proportions in each.

Drive load with your own traffic shape. The comparison is only valid if step 4
replays the identical shape, so script it rather than driving by hand.

In the trace, sum self time for:

- MoE front: `route_radix` / front-epilogue kernels
- SP collective: the fused AG/RS kernels, or `nccl*` if falling back

Those two sums over total step time give the ceiling. If they are 4% of step
time, a 1.2x kernel win is under 1% end to end -- worth knowing before, not
after, you book the node.

## 2. Tune

One process per GPU, single node:

```bash
SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB=32768 \
torchrun --nnodes=1 --nproc-per-node=8 \
  benchmark/kernels/kimi_k3/bench_sp_collective.py --tune --output-auto
```

No multinode env vars on B300, but the push size must still be forced: it sizes
the push workspace and is not something the sweep picks. Unforced, every
T >= 512 is skipped ("local shard N B exceeds push slot 786432 B") and you get a
table covering only small batches without any error.

`--output-auto` writes to the exact path the runtime looks up; hand-naming the
file reintroduces the silent-miss bug. The measured device string is
`NVIDIA B300 SXM6 AC`, not `NVIDIA B300`.

**Deployment coupling:** the table records `push_slot_bytes`, but the runtime
does not read it back -- it reads `SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB`.
A server consuming this table must set that env var to at least the recorded
value, or the push strategies the table selects will not fit.

Every candidate is checked against NCCL before timing, so a table that selects
`nccl` for some token bucket is a real result, not a failure.

## 3. Verify the tables now resolve

```bash
python3 benchmark/kernels/kimi_k3/check_tuned_configs.py   # expect TUNED
```

`sp_collective._TABLES` caches per path at first lookup, so the server must be
restarted to pick up a new file. A running process will not see it.

## 4. Re-measure identically

Restart the server, replay the exact step-1 workload, profile again into
`/tmp/k3prof/tuned`, and compare:

- the two kernel-family sums from step 1
- end-to-end TTFT and output token throughput

Report both. The kernel delta is what you tuned; the end-to-end delta is what
anyone cares about, and the ratio between them is the number worth keeping.

## Experiment design

Prefer A/B on **one** box: baseline, drop the table in, restart, re-measure.
Same silicon, same NUMA, same neighbors, same build -- the table file is the
only variable.

Comparing a freshly provisioned box against an existing endpoint confounds the
table with node-to-node variation, driver and image drift, and whatever else
differs between two machines. If the existing endpoint cannot tolerate a
restart, use a second box but pin the image digest and the launch flags, and
baseline the new box untuned first -- so the comparison is still
tuned-vs-untuned on the same host, with the old endpoint only as a sanity check.

## Expected magnitude

The only published numbers are from the GB300 MoE-front sweep (`276e7c91db`),
which reports 1.08x-1.22x on the epilogue kernel across 1 -> 16384 tokens.
That commit's own framing is the relevant part: the kernels had shipped
"with parameters inherited verbatim from route_radix and never swept. Both were
set wrong."

For `sp_collective` there are no published numbers, but the structural point is
stronger: on a miss the fused path does not execute. The comparison is
fused-vs-NCCL, not tuned-vs-untuned.

Neither figure is end to end. Step 1 is what converts them.
