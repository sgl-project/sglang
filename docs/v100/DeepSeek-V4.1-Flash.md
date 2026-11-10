# DeepSeek-V4.1-Flash on 8× V100

The most capable model this engine serves, for reasoning and world knowledge. As far as we know, no other engine runs it on V100s for local agentic coding. It is slow; it runs.

Official `deepseek-ai/DeepSeek-V4.1-Flash` on 8× V100-SXM2-32GB. This build has carried a multi-hour Claude Code session on a single conversation. Continuations that match a recorded chunk or request stop are not re-prefilled (a few dozen new tokens reach the first token in about 1 s). A suffix of several thousand tokens that was never computed runs at about 300 tok/s: each rank holds 30 of its 48 experts on the GPU, and prefill copies the spilled experts a chunk uses from host memory over PCIe, which is most of the prefill time. Expect 9 to 17 minutes from launch to ready (three launches on 2026-10-05: 2× Xeon Gold 6130, 352 GB RAM, checkpoint on a PCIe NVMe SSD; the spread was the SSD's read speed). Set up host memory before the first launch ([below](#host-memory)).

## Get the model

Use the official DeepSeek mixed-quant checkpoint. Dense weights are block FP8 (`quant_method: fp8`, 32×32 `ue8m0`); routed experts are native FP4 (`expert_dtype: fp4`, MXFP4). The DSpark draft lives in the same repo. Runtime on this port is FP16 activations and an FP8-E4M3 KV cache. Leave the dense MXFP8 packed; unpacking it to FP16 does not fit in 32 GB. Skip third-party NVFP4 / GPTQ / AWQ re-quants, and skip DeepSeek-V4-Flash: that is a different architecture.

Weights: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

```bash
pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --local-dir ~/models/DeepSeek-V4.1-Flash

export MODEL_PATH=~/models/DeepSeek-V4.1-Flash
```

~476 GB (48 shards). The export is multimodal. The vision tower is one fp32 copy on rank 0; each GEMM streams through that GPU as fp16 and is dropped. The model is MIT-licensed.

| | |
|---|---|
| checkpoint | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash |
| quantisation | native mixed: MXFP8 dense (e4m3 + UE8M0, 32×32) + MXFP4 routed experts |
| do not use | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash , https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 , https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark (V4-Flash NVFP4, not V4.1) |

## Host memory

Two large host-side pieces, both on the NUMA node(s) next to the GPUs. Find that node in the `NUMA Affinity` column of `nvidia-smi topo -m`.

- **Engram tables, ~189 GiB**, mapped from a boot-reserved 1 GiB hugetlb pool on the GPU-local node. The launch fails loudly if the pool is short instead of falling back to 4 KiB pages. On the measured box all eight GPUs sit on node 1, reserved on the kernel command line:

  ```
  default_hugepagesz=1G hugepagesz=1G hugepages=1:190
  ```

  `hugepages=<node>:<count>` reserves on one node. The server finds the GPUs' node at startup; `SGLANG_DSV41_ENGRAM_NUMA_NODE=<node>` overrides it.
- **Pinned expert spill, 13 GiB per GPU** (~104 GiB), on transparent huge pages, not from the hugetlb pool. It is striped layer by layer over the GPU-local node first, then the other nodes (both sockets on a two-socket host, one node on a single-socket host). `SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES=1,0` sets the list by hand.

Before each launch:

- Drop the page cache, especially right after another model: `sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`. Another checkpoint's cached pages compete with this load for memory.
- Turn MGLRU off if `/sys/kernel/mm/lru_gen/enabled` is not `0x0000`: `echo n | sudo tee /sys/kernel/mm/lru_gen/enabled`. With it on, the kernel swaps out the server's own memory ahead of the checkpoint's page cache during the load. On the measured box (Ubuntu 26.04, where it is on by default) the same code swapped out 102 GiB and filled the 64 GiB swap with MGLRU on, and swapped nothing with it off. The serve script warns when it is on. The setting does not survive a reboot.

## Measured performance

Rows dated 2026-10-05 are streamed chat requests on `scripts/serve_dsv41_v100.sh` as committed, with the expert cold set. "Per step" is decoded tokens per verify step, at most 6 with DSpark's 5 draft tokens. Undated rows are 2026-09-24, `llm-decode-bench` 0.6.2, temperature 0. All use the 8-card recipe (DSpark on, sticky last-seq, radix cache off, advertised 256k, `np=1`). The ship script pins `--max-running-requests 1`.

| check | result |
|---|---|
| Coding, `merge_sorted`, natural stop at ~101 tokens, 3 runs each (2026-10-05): temperature 0 | **11.1** tok/s, 5.9 per step |
| Same, temperature 1, top_p 0.95 | **10.2** tok/s, 5.4 per step |
| Same, temperature 0.7, top_p 0.95 | **10.6** tok/s, 5.7 per step |
| Review of a 2,977-token source file, 160 tokens out, temperature 1, top_p 0.95 (2026-10-05) | about **7** tok/s decode, 3.0 per step; 33 s end to end |
| Sustained padding decode, 20 s, `ignore_eos` | **2.8** tok/s (ITL 322 ms) |
| Cold prefill scout, server counted 5,286 prompt tokens | TTFT **45.3 s**, **117** tok/s |
| Warmer one-token prefill, 8,004 tokens, spill already touched | TTFT **14.3 s**, **558** tok/s |

Acceptance depends on the text: code and tool calls accept 4 to 6 tokens per step, free prose (reviews, explanations, reasoning) 2.5 to 3.5, so prose decodes at roughly two thirds of the coding rate. The 2.8 tok/s cell is greedy padding, the case this model loops on. It is a different number from the 11.1 tok/s coding rate and from the ~300 tok/s uncached suffix above. The 117 tok/s scout is the cold first touch; 558 tok/s is the same box after that spill was already warm. Temperature 0 is right for short code and wrong for long prose. Use `temperature=1`, `top_p=0.95` for chat. The ship script leaves `/health` as a liveness probe (no generation), so a load balancer GET does not drop the sticky pin.

## Reference recipe

The wrapper is the supported entry. It exports the env knobs that are not CLI flags, then launches the server.

```bash
export MODEL_PATH=~/models/DeepSeek-V4.1-Flash
export SGLANG_DSV41_DSPARK=1
export SGLANG_DSV41_STICKY_LAST_SEQ=1
export SGLANG_DSV41_CONTEXT_LEN=262144
bash scripts/serve_dsv41_v100.sh
```

Expanded (what the script actually runs when DSpark is on). Spill, Engram, and sticky are set by the env block, not implied by the CLI flags.

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private
export SGLANG_DSV41_EXPERT_SPILL_APPLY=1
export SGLANG_DSV41_EXPERT_SPILL_GB=13
export SGLANG_DSV41_SPILL_LANDING=36
export SGLANG_DSV41_PREFILL_LANDING=1
export SGLANG_DSV41_EXPERT_SPILL_COLD_SET=scripts/dsv41_flash_cold_set_ep8.json
export SGLANG_DSV41_STICKY_LAST_SEQ=1
export NCCL_ALGO=allreduce:tree
export NCCL_BUFFSIZE=2097152
export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=4

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp 8 --ep-size 8 \
  --dtype float16 \
  --moe-runner-backend marlin \
  --attention-backend dsv4 \
  --context-length 262144 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.87 \
  --max-running-requests 1 \
  --max-total-tokens 262144 \
  --max-prefill-tokens 262144 \
  --pre-warm-nccl \
  --warmups dsv41_chunk,sampling \
  --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 1 \
  --disable-radix-cache \
  --reasoning-parser deepseek-v41 \
  --tool-call-parser deepseekv41 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --sleep-on-idle \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "${MODEL_PATH}" \
  --disable-overlap-schedule \
  --host 0.0.0.0 \
  --port 11435
```

| knob | ship value | why |
|---|---|---|
| DSpark | on (`γ=5` from the checkpoint) | Best measured TG on this box. `SGLANG_DSV41_DSPARK=0` is greedy Tree |
| sticky last-seq | on | One resident conversation. Exact continuation, or a shorter prefix saved at a chunk or request stop. Anything else drops the pin and prefills from 0, unless it continues a stored conversation (next row). Radix stays off (a radix hit desyncs CSA2 pending state) |
| conversation store | on, last 4 kept | When a request starts a different conversation, the resident one's sparse-attention state is written to disk (`SGLANG_DSV41_CSA2_SESSION_DIR`, `SGLANG_DSV41_CSA2_SESSION_KEEP`). A later request that continues a stored conversation from a recorded stop loads it back instead of prefilling from 0. Switching away and back, a 300-token continuation of a 3k-token conversation reached its first token in 3.1 s, against 9.5 s to prefill the 3k tokens cold (2026-10-07) |
| context / max tokens | 262144 | Advertised window. 8k, 32k and 250k prefills are smoked (250k takes about 17 minutes); 512k has not left ~300 MiB for the Engram MXFP8 unpack |
| `--mem-fraction-static` | 0.87 | 0.99 OOMs the Engram unpack on T=6 verify capture. 0.88 OOMed the same 300 MiB unpack on a 461-token sticky prefill (TP7 had 284 MiB). 0.86 raises: no KV pool after draft weights |
| expert spill | 13 GiB/rank (landing 36) | Spill 12 left 8k ~8 MiB short of that unpack |
| prefill landing | on | Prefill copies the spilled experts a chunk uses into the landing slots, with no swap back to host. 8.3k cold prefill 25 s (42 s with the swap), 300-700-token follow-ups 2.3-3.3 s to the first token (5-6 s with the swap). `SGLANG_DSV41_PREFILL_LANDING=0` restores the GPU-slot swap |
| expert cold set | `scripts/dsv41_flash_cold_set_ep8.json` | Picks which 18 of each GPU's 48 experts per layer live in host memory: the least used on a recorded prompt set (file review, coding, agent turns, reasoning). On held-out requests that is 19 expert reads from host per decoded token instead of 91. An empty `SGLANG_DSV41_EXPERT_SPILL_COLD_SET=` keeps the last experts of each GPU on the host instead |
| overlap scheduler | off (`--disable-overlap-schedule`) | Each DSpark step waits on the host for its accepted length, so overlap hides nothing and runs one extra verify step (~0.55 s) before a request's first token and after its last. Off, follow-up turns reach the first token ~0.55 s sooner and end ~1.2 s sooner, with the same greedy output. A request that ends on its last step's bonus token has no KV for it yet; the prompt cache then stops one token short and the next turn computes that token again (2026-10-06) |
| `--max-running-requests` | 1 | DSpark would otherwise inflate this |
| `--sleep-on-idle` | on | Without it the eight idle scheduler loops keep about 6 CPU cores busy |
| `--chunked-prefill-size` | 2048 | Vestigial SWA floor is sized for this chunk |

Leave `--speculative-dspark-block-size` at the checkpoint default. Checkpoint weights stay mixed MXFP4 experts + packed MXFP8 dense.

## Limitations

- One conversation on the GPU. A new conversation prefills from zero; continuing one of the last four stored conversations loads it from disk (conversation store above). The resident image does not survive a restart.
- A long suffix that was never computed runs at about 300 tok/s (each prefill chunk copies its spilled experts from host memory over PCIe).
- A few GiB of HBM are left after load. Open-ended greedy decoding (temperature 0) can loop; use `temperature=1`, `top_p=0.95` for chat.
- Image requests stream the rank-0 vision tower through GPU GEMMs: fine for casual use, not fast.
