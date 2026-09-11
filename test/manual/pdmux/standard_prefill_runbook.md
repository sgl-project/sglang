# PDMux standard-prefill lane: validation runbook

Companion to `--pdmux-prefill-mode standard`. Everything here is a command to
run on the 8-GPU host; nothing below has been executed in the session that
produced the branch (that checkout had no torch, no GPU and no checkpoint).
Status wording to use in the report: *the main design questions have a static
explanation; correctness, resource attribution, TP8 concurrency and
performance are still to be validated.*

## 0. Checkpoint check (run first; DSpark depends on the answer)

```bash
MODEL=/model/DeepSeek-V4-Flash
python3 - "$MODEL" <<'PY'
import json, sys, pathlib
cfg = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
print("architectures       :", cfg.get("architectures"))
print("num_hidden_layers   :", cfg.get("num_hidden_layers"))
nextn = cfg.get("num_nextn_predict_layers")
print("num_nextn_predict   :", nextn, "-> MTP/NextN head present" if nextn else "-> NO MTP head")
dspark = {k: v for k, v in cfg.items() if k.startswith("dspark_")}
print("dspark_* keys       :", dspark or "NONE -> DSpark needs --speculative-draft-model-path")
print("quant_method        :", (cfg.get("quantization_config") or {}).get("quant_method"))
PY
ls "$MODEL" | grep -c safetensors
```

The public `deepseek-ai/DeepSeek-V4-Flash` config carries
`num_nextn_predict_layers: 1` and no `dspark_*` keys. If the local checkpoint
matches, MTP uses the bundled NextN head and DSpark needs a separate draft
checkpoint. The NextN head is not a DSpark head; do not point DSpark at it.

## 1. CPU unit tests (no GPU)

```bash
python3 -m pytest -q \
  test/registered/unit/multiplex/test_pdmux_standard_prefill.py \
  test/registered/unit/multiplex/test_pdmux_hicache_events.py \
  test/registered/unit/multiplex/test_pdmux_overlap_streams.py \
  test/registered/unit/managers/test_pdmux_scheduler.py \
  test/registered/unit/model_executor/test_pdmux_decode_cuda_graph.py
```

## 2. Launch recipes

Shared environment (from the committed DSV4 PDMux manual tests):

```bash
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
export SGLANG_DSV4_FP4_DEQUANT=1        # pure-TP Triton runner cannot run mxfp4 experts
export SGLANG_TEST_DSV4_FLASH_MODEL_PATH=/model/DeepSeek-V4-Flash
```

Main regression profile: TP8, FP4 dequant, HiCache `write_through` at ratio
1.5, `--mem-fraction-static 0.65`. Both memory values are carried over from
the original launch configuration for this work; `--hicache-ratio` sizes the
host cache relative to the device pool and does not by itself change device
memory. The HiCache-off cells use the 0.85 the committed sanity / concurrent
tests use.

```bash
STANDARD=(--trust-remote-code --tp 8
          --enable-pdmux --disable-overlap-schedule
          --pdmux-prefill-mode standard --cuda-graph-backend-prefill disabled
          --max-running-requests 8
          --enable-metrics)   # the HiCache checks read /metrics; default is off
HICACHE=(--enable-hierarchical-cache --hicache-write-policy write_through
         --hicache-ratio 1.5 --mem-fraction-static 0.65)
NOCACHE=(--mem-fraction-static 0.85)
MTP=(--speculative-algorithm EAGLE --speculative-num-steps 3
     --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
# Only after step 0 confirms the draft; add --speculative-draft-model-path
# <dspark-draft> when the target config has no dspark_* keys.
DSPARK=(--speculative-algorithm DSPARK --speculative-num-draft-tokens 4)
FULL_EXTEND=(--chunked-prefill-size -1)
TOKEN_CHUNK=(--chunked-prefill-size 8192)
```

Two SM layouts. `--sm-group-num` must equal the YAML's `sm_group_num`.

Exclusive partitions (H20, 78 SMs; prefill counts are multiples of 8 on
Hopper). Thresholds spread 1..8 concurrent decodes over the shared groups:

```bash
cat > /tmp/pdmux_exclusive.yaml <<'YAML'
sm_group_num: 8
split_forward_token_budget: 8192
manual_divisions:
  - [56, 22, 1]
  - [48, 30, 2]
  - [40, 38, 3]
  - [32, 46, 4]
  - [24, 54, 6]
  - [16, 62, 8]
YAML
EXCLUSIVE=(--sm-group-num 8 --pdmux-config-path /tmp/pdmux_exclusive.yaml)
```

Overlay (prefill capped, decode on a full-device high-priority stream). The
decode column is ignored and rewritten to the full device; each prefill cap must
sit strictly inside (0, 78); the first threshold is 1 so every non-empty decode
batch selects a group:

```bash
cat > /tmp/pdmux_overlay.yaml <<'YAML'
sm_group_num: 4
split_forward_token_budget: 8192
overlap_decode_full_sm: true
manual_divisions:
  - [24, 0, 1]
  - [40, 0, 8]
YAML
OVERLAY=(--sm-group-num 4 --pdmux-config-path /tmp/pdmux_overlay.yaml)
```

One cell of the matrix, written out. The server runs in the background here
(or launch it in its own terminal); the readback only makes sense once it is
up:

```bash
python3 -m sglang.launch_server --model-path "$SGLANG_TEST_DSV4_FLASH_MODEL_PATH" \
  "${STANDARD[@]}" "${HICACHE[@]}" "${TOKEN_CHUNK[@]}" "${EXCLUSIVE[@]}" \
  --log-level debug > /tmp/pdmux_server.log 2>&1 &
SERVER_PID=$!
until curl -sf localhost:30000/health > /dev/null; do sleep 5; done
curl -s localhost:30000/get_server_info | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("pdmux_prefill_mode :", d["pdmux_prefill_mode"])
print("enable_metrics     :", d["enable_metrics"])
print("hicache_ratio      :", d["hicache_ratio"])
print("mem_fraction_static:", d["mem_fraction_static"])'
```

Always read the effective configuration back: a server that resolved to the
other lane validates nothing about this one, and a server without metrics
makes every counter read as zero. Stop it with `kill $SERVER_PID` afterwards.

Matrix: {plain, MTP, DSpark} x {HICACHE, NOCACHE} x {FULL_EXTEND, TOKEN_CHUNK}
x {EXCLUSIVE, OVERLAY}. Record every cell you did not run as *not validated*.

## 3. Baselines

Every baseline runs the same checkpoint as the cell it is compared with.

- plain, no PDMux: `test/manual/dsv4/test_dsv4_flash_sanity_tp8.py` recipe.
- plain, PDMux layer_split: the same launch with `--pdmux-prefill-mode
  layer_split` (the default). This is the only speculative-free comparison
  layer_split can provide.
- MTP: the standard scheduler (no PDMux) with the `MTP` arguments above and
  real acceptance. `test/manual/dsv4/test_dsv4_flash_mtp_tp8.py` sets
  `SGLANG_SIMULATE_ACC_LEN=3` for a latency case; do not carry that variable
  into either the baseline or the PDMux cell, or the accept lengths compare
  a simulation. Do not use layer_split as the MTP baseline: its split path
  bypasses the draft extend.
- DSpark: the standard scheduler with `DSPARK` and the confirmed draft
  checkpoint.

## 4. What to record per cell

Correctness
- Greedy output identical to the matching baseline on the same prompts.
- Needle recall on ~4k-token prompts (the concurrent test's prompts).
- MTP / DSpark: accept length distribution vs the standard-scheduler baseline,
  from real draft / verify / accept.
- HiCache mechanism: the `*StandardPrefill` classes launch their own server on
  the standard lane and assert the effective mode and metrics from
  `/get_server_info`:

  ```bash
  python3 -m pytest -v test/manual/pdmux/test_dsv4_pdmux_hicache_tp8.py -k StandardPrefill
  ```

  They run at the file's own `--hicache-ratio 2` / `--mem-fraction-static 0.7`
  and cover evicted-prefix reload with identical output, plus a smoke check
  that the backup counter advances while requests are still running. That
  smoke check is not the in-flight proof: request completion includes the
  decode tail and several prompts may share one prefill batch. The 1.5 / 0.65
  profile is covered by the launch recipe in section 2; issue the reload check
  against it as well.
- HiCache in-flight draining (TP8 timing item, standard lane): show that
  backup acks retire while a later prefill work item is running, not only once
  the lane is idle. Launch with `HICACHE` + `TOKEN_CHUNK` + `--log-level
  debug` so every prompt of >= 32k tokens becomes >= 4 chunk work items and
  every finalize logs `PDMux prefill finalized: ...`. Send 4 such prompts with
  `max_new_tokens=1` (completion then coincides with the last chunk's
  finalize) while sampling `sglang:hicache_backup_tokens_total` every 200 ms,
  stamping each sample after the read. Accept when the counter advances at a
  sample whose stamp falls between two consecutive finalize lines of the same
  prompt -- i.e. while its next chunk was in flight. A counter that only moves
  after the last finalize line fails this item.
- hidden states and logprob returns for plain and MTP, where the standard
  worker already supports them; do not widen DSpark beyond what it supports.

Lifecycle
- Abort a request while its prefill is in flight; finish and slot reuse; a
  prefill submitted into an idle server completes.

Performance (measured, never estimated)
- CPU submit time of a prefill and the completion-vote cost: the scheduler
  debug line `PDMux prefill finalized: CPU submit <ms>, <n> completion votes,
  <ms> spent waiting on them`. This is host wall time; the device timer's
  `extend` bucket is GPU time and does not measure it.
- Decode step latency, TTFT, ITL p50/p99, output tokens/s: bench_serving.
- Peak device memory per rank.
- Compare against the layer_split cell (plain only) and the standard-scheduler
  cells (MTP / DSpark). Do not assume parity.

Resource attribution -- judged against the stream group the loop selected at
that moment, which the debug log reports on every switch
(`Adjusting stream groups: <idx>, prefill sm: <n>, decode sm: <n>`):
- With no decode batch the loop selects group 0 and the prefill may use the
  full device; that is correct, not a leak.
- With a decode batch the loop selects a shared group. Prefill-lane kernels
  stay inside that group's prefill cap in both layouts; decode-lane kernels
  stay inside the group's decode partition (exclusive layout) or reach the
  full device (overlay layout).
- Trace the decode-lane CUDA graphs (nsys, or the graph node trace) and check
  the resource context the kernel nodes execute under, not the launch stream:
  a graph captured in one context does not gain another by being replayed on
  a different stream.

## 5. Known limits of this lane

- EAGLE draft CUDA graphs are skipped (eager draft steps / draft extend).
  DSpark's per-group draft and verify graphs stay on, as do the target's
  decode and verify graphs.
- DSV4 and the DSpark draft run their single-stream paths (no model-internal
  helper streams).
- Rejected at launch: a non-disabled prefill CUDA graph, multi-layer EAGLE,
  TBO, unified memory, DP attention, EP, CP, DCP.
