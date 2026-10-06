# DeepSeek-V4.1-Flash layerwise PDMux validation

Use the same local DeepSeek-V4.1-Flash checkpoint, GPU topology, sampling settings
and workload for PDMux and the ordinary scheduler. Set `MODEL_PATH` to that
checkpoint. CPU tests cover state and scheduling; the GPU checks below remain
required before claiming output parity or performance.

PDMux uses layerwise prefill directly. Token chunking and layer slicing can be
combined. Intermediate slices preserve batch-owned mHC/carry state; only the last
slice computes logits. A prefill with no decode work executes its remaining layers
in one call. With decode work, the slice budget divides by the sum of prefill
tokens across attention DP ranks, then applies `max_split_forward_layers`.

Example for a 132-SM GPU (adjust both counts to your actual GPU):

```yaml
sm_group_num: 3
manual_divisions:
  - [104, 28, 1]
split_forward_token_budget: 8192
max_split_forward_layers: 2
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

This core layer supports plain TP. Attention DP and speculative decoding are
enabled by the later stack layers and are rejected at startup in the core branch.
