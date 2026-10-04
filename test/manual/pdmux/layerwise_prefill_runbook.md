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
There is no additional layer-count cap.

Example for a 132-SM GPU (adjust both counts to your actual GPU):

```yaml
sm_group_num: 3
manual_divisions:
  - [104, 28, 1]
split_forward_token_budget: 8192
overlap_decode_full_sm: false
```

Save as `/tmp/dsv41-pdmux.yaml`, then launch plain TP8:

```bash
python -m sglang.launch_server --model-path "$MODEL_PATH" --trust-remote-code \
  --tp 8 --enable-pdmux --disable-overlap-schedule --sm-group-num 3 \
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
