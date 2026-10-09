# DeepSeek-V4.1-Flash layerwise PDMux validation

Use the same local DeepSeek-V4.1-Flash checkpoint, GPU topology, sampling settings
and workload for PDMux and the ordinary scheduler. Set `MODEL_PATH` to that
checkpoint. CPU tests cover state and scheduling; the GPU checks below remain
required before claiming output parity or performance.

For implementation steps, model differences and previous failure modes, see
[the model adaptation guide](model_adaptation_guide.md).

PDMux uses layerwise prefill directly. Token chunking and layer slicing can be
combined. Intermediate slices preserve batch-owned mHC/carry state; only the last
slice computes logits. A prefill with no decode work executes its remaining layers
in one call. With decode work, the slice budget divides by the maximum prefill
token count across attention DP ranks, matching the `layer_split` path of
`feat/pdmux-standard` at `17761860ccbc6cea53b0c3625fa0c89880b48f9e`.
`max_split_forward_layers` defaults to `0`, preserving this token-budget-only
policy. A positive value additionally caps each slice while any DP rank has
decode work; for example, `2` submits at most two layers per slice. It does not
limit a prefill when there is no global decode work.

Example for a 132-SM GPU (adjust both counts to your actual GPU):

```yaml
sm_group_num: 3
manual_divisions:
  - [104, 28, 1]
split_forward_token_budget: 8192
max_split_forward_layers: 0
overlap_decode_full_sm: false
```

Save as `/tmp/dsv41-pdmux.yaml`, then launch plain TP8:

```bash
python -m sglang.launch_server --model-path "$MODEL_PATH" --trust-remote-code \
  --tp 8 --moe-a2a-backend none --enable-pdmux --disable-overlap-schedule --sm-group-num 3 \
  --pdmux-config-path /tmp/dsv41-pdmux.yaml --chunked-prefill-size 8192
```

For full-device decode overlap, set `overlap_decode_full_sm: true`. The prefill
count remains a Green Context cap; the decode column is ignored in that layout.
Test both layouts. The CLI `--sm-group-num` must equal the YAML value.
DSV compressor plans have a 65535-token hard cap (65520 for page size 16);
layer slicing does not reduce a plan's token count. Chunk budgets must fit it.

Compare greedy output with PDMux disabled on TP1 and TP8. Exercise long prompts
while streaming decode, alternating batches, multi-chunk continuations, parked
chunks, prompt-time aborts and a healthy request after the storm. Force small slice
budgets and check that Engram and late-layer tail outputs still match.

The TP8 manual entry points are:

```bash
SGLANG_TEST_DSV4_FLASH_MODEL_PATH="$MODEL_PATH" \
  python test/manual/dsv4/test_dsv4_flash_pdmux_sanity_tp8.py
SGLANG_TEST_DSV4_FLASH_MODEL_PATH="$MODEL_PATH" \
  python test/manual/dsv4/test_dsv4_flash_pdmux_concurrent_tp8.py
```

The concurrency test's manual SM counts are for a 78-SM GPU; adapt them for other
devices. Its serial comparison uses the same PDMux server, so also run the
ordinary-scheduler comparison above. For MXFP4 expert checkpoints select the
appropriate MoE runner/dequantization settings for the hardware.

Record TTFT, ITL p50/p99, output tokens/s and per-rank peak memory under the same
load. No GPU accuracy or throughput result is implied by passing CPU tests.

The core branch supports plain TP. The third layer enables attention DP; the
fourth adds speculative decoding. Earlier branches reject unsupported combinations.

## HiCache/SWA layer

Add `--enable-hierarchical-cache` for cache reload checks. The loop drains acks
before admission and every 16 in-flight/waiting iterations; device frees and
mapping updates publish a dependency to decode. Host-only write-through acks
leave plain-TP prefill/decode overlap intact. Buffer-mode load acks can release
auxiliary device slots and also need that dependency.

```bash
SGLANG_TEST_DSV4_FLASH_MODEL_PATH="$MODEL_PATH" \
  python -m pytest test/manual/pdmux/test_dsv4_pdmux_hicache_tp8.py -v
```

Test both SM layouts with long cached prefixes, pressure-driven FULL/SWA eviction,
load-back and repeated chunk continuations. Check free-list ownership after
tombstone recovery, including a protected prefix that ends inside the tombstone
node. Include in-flight backup acks and post-storm output checks. The manual tests
cover write-through; separately validate write-back and storage/buffer modes before
relying on them in deployment.

## Attention DP layer

Run the same launch with `--enable-dp-attention --attn-dp-size 8` (TP8/DP8), then
`--attn-dp-size 2` (TP8/DP2 with attention TP4). Exercise balanced and uneven prompt
sizes, peer-only prefill/decode and all ranks IDLE. Check matching layer boundaries,
no collective hang, correct greedy output and pad/unpad restoration across slices.
Scheduling retains the default `layer_split` submission order of
`feat/pdmux-standard@17761860cc`, with two DP adaptations from
`feat/glm53-flash-pdmux@a4c3f271fb`. Decode metadata is gathered before selecting
streams; active and IDLE participants use the maximum global decode count. A
locally empty rank remains active for peer-only decode. The prefill scope replaces
full-TP and handles that alias that exact coordinator, including MoE-EP in
TP8/EP8. Independent attention/MoE groups retain their ordinary handles; only the
full-TP duplicate is created. PDMux cross-DP metadata uses CPU/Gloo even with
`--disable-overlap-schedule`, and cannot be skipped by
`SGLANG_SCHEDULER_SKIP_ALL_GATHER`. Ordinary scheduler policies remain unchanged.
Scheduling keeps a separate global token vector even when the model consumes
local MLP counts. Each slice uses the existing DP pad/unpad path without a second
unpadded token-count snapshot.

The loop submits decode, then a prefill slice, then waits for the local decode
stream and processes its result. Prefill completion is polled and merged after
the existing full-TP ready vote. There are no per-phase submission barriers or
extra decode-to-prefill events. The prefill-to-decode formation/HiCache and merge
events remain, as do the drains when changing stream groups. Formation-time
HiCache draining lives in `update_split_prefill_batch` to accommodate this
branch's cache API without changing the reference loop's lane order.

The SM90 V4.1 Torch indexer uses CPU request lengths and slots to build its row
ranges, avoiding CUDA unique/nonzero/item during submission. The FP4 pool uploads
its dequant table at initialization, rather than inside each request. Decode
sequence-length copies use pinned buffers and are published after the completion
fence. Engram extend supplies repeat_interleave's known CPU output size, matching
the standard branch, and eager logits budgets avoid a live CUDA memory query.
The model-specific CPU-mirror retirement runs after the existing lane completion
fence. It adds no collective or stream drain. The rejected GPU-serial and
five-rendezvous protocols have been removed; the restored reference scheduling
still requires GPU liveness and performance validation on the V4.1 workload.

This matrix covers TP/attention DP. EP, CP and DCP need their own validation.

### Uneven-rank hang regression

The communicator-only candidate `34f75d6e6b` failed the original GPU regression:
44 completed requests in 2 minutes 9 seconds, then all eight workers became
unhealthy. DP0/1/3/4 waited in `decode_stream.synchronize()`, DP2/5/6 in
`torch.unique_consecutive`, and DP7 in decode's `new_seq_lens.to("cpu")`.
Only DP2/5/6/7 had cold prefill work. This result supersedes that candidate's
CPU-only validation; communicator isolation did not establish liveness.

The GLM-aligned candidate `439624b4b4` also failed: 234 requests / 6 minutes
14 seconds, seven ranks in decode synchronization and DP5 in prefill's low-ratio
compressor `linear`. The loop calls decode before prefill, so this Python stack
alone does not establish that DP5 omitted decode. The actual process had implicit
ordering unset, PyNCCL 2.30.7, PyTorch NCCL reporting 2.29.7, CUDA 13.0 and driver
580.105.08. GLM's reported success remains a control, not V4.1 acceptance.

Re-run the original SM90 TP8/DP8/EP8 + DSpark + HiCache stress configuration,
including its original MoE backend, graph settings and SM layout. Keep
`chunked_prefill_size: 1024`, `max_split_forward_layers: 2`, cold longcodebench
prompts and concurrency 10. Require at least 30 minutes of continuing request
completion and healthy workers. Preserve token-budget, sampling and cache
settings; increasing the slice cap would change the reproduction. The cap-0
variant already ran for 30 minutes in the reported control and should remain a
separate control. Cap 0 remains the default; cap 2 is deliberately retained for
this regression despite its 20 slices for a 40-layer chunk with decode work.

Also exercise peer-only prefill/decode and swap busy ranks, graph replay on/off,
both SM layouts and TP8/DP2. CPU tests cover local lane submission and result
retirement, HiCache pump cadence, TP-only communicator scope, DP metadata and
model-specific host-sync avoidance. Their simulated CUDA streams do not
establish NCCL kernel liveness or GPU overlap.
GPU liveness, full-model output parity and performance remain unverified on this
CPU-only machine. Capture a CUDA timeline demonstrating simultaneous decode and
prefill computation as well as 30 minutes of progress with the original cap-2
configuration; a healthy service with serialized kernels is not a passing result.

Run the native CUDA submission check on the GPU node before the full-model test:

```bash
PYTHONPATH=python python3 -m pytest -q \
  test/registered/unit/layers/test_dsv41_prefill_submission.py
```

It checks the real FP4 dequant pool and Engram kernels with CUDA sync-debug mode
set to error, plus selection/hash parity. This small test does not replace the
eight-GPU regression.

Check that every rank has a split-prefill participant whenever any rank has
prefill work. An IDLE rank can legitimately skip the indexer but must execute the
same MoE layer collectives as active ranks. Capture per-rank stacks, stream IDs,
communicator IDs and layer indices if progress stops; record completed requests,
output parity, TTFT/ITL and throughput when it succeeds.

Before multi-GPU CUDA PDMux communicator creation, set and validate
`NCCL_LAUNCH_ORDER_IMPLICIT=1`, NCCL >= 2.26 in both callers, and CUDA runtime/driver
>= 12.3. This supplies device ordering while retaining overlap, without the
rejected five handshakes or per-phase GPU drains. An explicit value of 0 is
rejected; unsupported stacks are not silently serialized. Graph mixing retains
its configured/default policy. The original cap2 GPU regression is still required.

Enable `SGLANG_PDMUX_TRACE=1` for the next diagnostic run. Host-only phase records
distinguish D/P submission return from GPU completion and include global counts,
mode, layer interval, stream and TP handle. Actual graph begin/returned records
are separate from the worker result's graph flag. Capture DP5's native stack if
the new candidate freezes. Detailed interpretation and the optional multi-GPU
graph/eager probe are in [the adaptation guide](model_adaptation_guide.md#111-区分host-未提交与已提交但-gpu-不推进).
Disable trace for performance measurements. Require c10 progress for 30 minutes
before attempting c24; the latest failed c10 did not proceed to c24.

### Why the reported V4-Flash standard configuration is a different control

The working H100 command also enables TP8/DP8, DSpark and HiCache. It uses
`DeepSeek-V4-Flash-0731`, budget 65536 without a layer cap, EP's default of 1,
auto MoE selection with `SGLANG_DSV4_FP4_DEQUANT=1`, prefill SM 112 and full-SM
decode. There is no prefill-mode override, so it uses the standard branch's
default layer_split loop. The failed V4.1 command uses EP8, explicit
flashinfer_mxfp4 and 1024-token chunks capped at two layers per slice. With 40
layers and concurrent decode, the latter makes 20 slices; without the cap, the
same 1024-token chunk fits in the 65536 budget and runs in one slice.

The model dispatches ratio-4 layers through the C4 indexer and ratio-1/2 layers
through the V4.1 indexer, whose DeepGEMM path requires SM100+. On SM90 it uses the
Torch request scorer. The standard branch also contains that scorer's host
syncs, so a successful V4 run is not proof that standard fixes the V4.1 hang.
For a branch A/B, keep the same V4.1 checkpoint, EP/MoE backend, slice cap, SM
layout and cold-prompt workload. Preserve the successful V4 command separately.

## Single-layer MTP/EAGLE and DSpark layer

For checkpoint MTP, add these options to the launch above using a checkpoint
that contains the compatible single NextN head:

```bash
--speculative-algorithm EAGLE --speculative-draft-model-path "$MODEL_PATH" \
--speculative-num-steps 3 --speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

The target captures FULL pre-mHC hidden states shaped
`[tokens, hc_mult * hidden_size]`. Intermediate slices do not run draft extend.
Only the final slice hands an independent batch view to the draft. MTP draft
decode/extend runs eager; target decode/verify retains its per-stream graph path.
`NEXTN` is an alias of EAGLE. Multi-layer EAGLE, EAGLE3, adaptive parameters and
other speculative algorithms are rejected with PDMux.

For DSpark, set `DSPARK_MODEL_PATH` to the matching independent draft checkpoint
and use:

```bash
--speculative-algorithm DSPARK \
--speculative-draft-model-path "$DSPARK_MODEL_PATH" \
--speculative-draft-attention-backend flashinfer \
--speculative-num-draft-tokens 8
```

With DSpark attention DP, also add `--enable-dp-lm-head`, as required by the
ordinary DSpark adapter. Keep `--moe-a2a-backend none` for this TP/attention-DP
matrix. Existing DSpark backend and static ragged-verify checks still apply if
you enable the optional DP speculative-prefill coordination environment setting.

DSpark accumulates aux hidden states across target slices, injects draft KV once
at completion and preserves its existing per-stream draft graph support. Target
verify and DSpark draft select the active decode backend; Eagle draft preserves
the caller's per-step backend. Backend isolation remains necessary for DeepSeek:
decode/IDLE metadata updates must not invalidate an unfinished split prefill's
metadata. Final draft/injection work follows the reference branch without an
extra bidirectional stream fence. The optional overlap planner stream follows
`SGLANG_ENABLE_OVERLAP_PLAN_STREAM`, as in the reference branch.

Compare each speculative configuration with its ordinary-scheduler counterpart.
Cover single-request greedy output and acceptance, long prefill during decode,
chunk continuation/abort, HiCache on/off, both SM layouts, TP8/DP8 and TP8/DP2,
peer-only work and IDLE completion without logits. Confirm draft KV injection
counts, final-only draft extension and graph/backend selection across stream
switches. Preserve the upstream restriction that decoder bounded replay/tail
does not support MTP FULL capture or DP; DSpark tail is tested without DP.
