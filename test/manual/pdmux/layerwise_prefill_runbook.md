# DeepSeek-V4.1-Flash layerwise PDMux validation

Use the same local DeepSeek-V4.1-Flash checkpoint, GPU topology, sampling settings
and workload for PDMux and the ordinary scheduler. Set `MODEL_PATH` to that
checkpoint. CPU tests cover state and scheduling; the GPU checks below remain
required before claiming output parity or performance.

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
leave prefill/decode overlap intact. Buffer-mode load acks can release auxiliary
device slots and also need that dependency.

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
The prefill lane switches only the full-TP handle to its duplicate communicator;
attention and MoE group handles retain the ordinary groups, as in the reference
branch. Stream selection uses the local running decode batch before decode's DP
metadata gather. Peer-only work still creates IDLE participants through that
gather. Scheduling keeps a separate global token vector even when the model
consumes local MLP counts, and honors the ordinary scheduler's optional skip-gather
environment setting. Each slice uses the existing DP pad/unpad path without a
second unpadded token-count snapshot.

This matrix covers TP/attention DP. EP, CP and DCP need their own validation.

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
