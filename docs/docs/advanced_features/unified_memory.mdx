---
title: "Unified Memory for Hybrid Models"
metatags:
    description: "SGLang unified memory (--enable-unified-memory): one KV arena shared by full-attention KV, sliding-window KV and Mamba/linear-attention state, with supported models, flags and feature interactions."
---
A hybrid model stores more than one kind of per-request memory: full-attention KV, sliding-window (SWA) KV, and recurrent state for Mamba or linear-attention layers. By default SGLang gives each kind its own pool and fixes the split at startup (`--swa-full-tokens-ratio`, `--mamba-full-memory-ratio`). When the workload does not match that split, one pool runs out while the other has room to spare.

With `--enable-unified-memory`, every kind lives in one device buffer, the *arena*. Each kind is a sub-pool that occupies a contiguous *band* of the arena's bytes, and bands grow and shrink as requests come and go. Memory moves between kinds as the workload shifts: long prompts take more full-attention KV, many short concurrent requests take more state slots and window KV.

This page covers when and how to enable it. The [design page](/docs/advanced_features/unified_memory_design) describes how it works and how to debug and test it.

## What changes

| | Separate pools (default) | Unified memory |
|---|---|---|
| Device memory | One buffer per pool, split by ratios at startup | One arena; every sub-pool takes pages from it at run time |
| Capacity | Fixed per kind; one kind can be full while another is idle | Shared; every capacity decision asks whether one layout of the arena serves a concrete demand. Per-sub-pool availability figures are estimates for observers |
| Token ids | `req_to_token` holds the ids the kernels index | `req_to_token` holds virtual ids; each sub-pool maps them to physical pages, so pages can move without rewriting request tables |
| KV layout | One buffer per layer | Page-major entries: one entry per token slot (or state slot) holds every layer's rows, so `--enable-page-major-kv-layout` is implied |
| Compaction | None | Pages move lazily to close holes and make room, ordered behind the forwards and transfers that may still touch them |

### Sub-pools and bands

Each kind of memory a model's layers declare becomes one sub-pool:

| Sub-pool | Name | Holds | Unit |
|---|---|---|---|
| Full attention | `full` | The full-attention KV. It owns the token id space. | A page of `--page-size` tokens |
| Sliding window | `window_<length>`, for example `window_4096` (`window` when the model does not state the length) | The KV of layers that attend through one window length | A page |
| Recurrent state | `mamba` | One state entry per slot (conv and SSM state of every state layer) | One slot |

Sub-pools are laid out in the arena by retention: the state sub-pool takes the low end and grows up, the full-attention sub-pool takes the high end and grows down, and window sub-pools sit between them. With no state sub-pool, the longest window takes the low end. A sub-pool between the two ends is a *float*: it occupies a span between its neighbours and is relocated when the layout needs it.

```text
low byte                                                                  high byte
| sink | mamba ->          | <- window_4096 -> |                 <- full |
         grow-up end          float                                grow-down end
```

The low bytes are a padding sink that every sub-pool's dummy writes land in.

## Supported models

Unified memory replaces the pools of hybrid models only. At startup, `KVCacheConfigurator._kv_layer_decls` declares the model's layers and refuses anything else.

| Model family | Supported |
|---|---|
| Hybrid sliding-window attention with MHA full attention (for example gpt-oss, Gemma 4) | Yes |
| Hybrid Mamba or linear attention (for example Qwen3-Next and Qwen3.5 GDN models, Kimi Linear with MLA) | Yes |
| Sliding window and recurrent state together | Yes |
| Full attention only (plain MHA or MLA) | No: refused at startup |
| MLA full attention with sliding-window layers | No |
| DeepSeek-V4 (compressed-KV attention) | No |
| Qwen4-Exp, Kimi-K3.1, Unlimited-OCR v2, MiniCPM with its sparse attention backends | No: refused by their model overrides |

The target model's sliding-window layers form one window sub-pool. A fused draft model whose window has a different length adds a second one. Only the Triton attention backend serves several window sub-pools per runner; another backend bound to more windows than it supports raises at startup, naming itself.

### Attention, linear-attention and Mamba backends

Page-major entries make each layer's rows strided, so only backends audited for strided views are accepted. The table is that page-major allow-list (`arg_groups/kv_cache_hook.py::handle_page_major_kv_layout`); a backend on it may still refuse a model when it is built:

| Layers | Page-major allow-list |
|---|---|
| MHA full and sliding-window attention | `triton`, `fa3`, `fa4`, `flashinfer`, `trtllm_mha` |
| MHA with asymmetric K/V head dims | `triton`, `fa3`, `fa4` |
| MLA full attention | `triton`, `fa3`, `trtllm_mla`, `flashinfer`, `cutedsl_mla`, `flashmla` |
| Linear attention, decode | `triton`, `flashinfer`; also `cutedsl` and `helion` on MLA (KDA) hybrids |
| Linear attention, prefill | `triton`, `flashkda`, `flashinfer`; also `cutedsl` and `helion` on MLA (KDA) hybrids |
| Mamba (`--mamba-backend`) | `triton` |

`fa3` and `fa4` still refuse two kinds of model when the backend is built: local-attention models, such as Llama 4, whose default backend on SM90 is `fa3` (`FlashAttentionBackend._maybe_init_local_attn_builder`), and models that decode with prefill-aware SWA (`FlashAttentionBackend.__init__`).

If the FULL prefill CUDA-graph backend is configured with an attention backend other than `fa3` or `fa4`, it is disabled with a warning; decode capture and the other prefill graph backends are unaffected.

## Quick start

Platform: NVIDIA CUDA. On other platforms, read [Known limitations](#known-limitations) first.

```bash
python3 -m sglang.launch_server \
  --model-path MODEL_PATH \
  --tp 2 \
  --enable-unified-memory
```

`MODEL_PATH` is a Hugging Face repo id or local path of a supported hybrid model.

## Flags and environment variables

| Flag or variable | Effect under unified memory |
|---|---|
| `--enable-unified-memory` | Builds the arena, the per-sub-pool views and one allocator over all of them. Implies `--enable-page-major-kv-layout`. |
| `--mem-fraction-static` | The arena's byte budget comes from the static memory fraction, as for separate pools. If the arena cannot hold the floor of one request at batch size 1 (its window rows, its state slots and the sink), startup fails and names the terms (`kv/sizing.py::check_bs1_floor`). Whether a whole request fits, its full-attention tokens included, is checked at the memory clip and at admission. |
| `--max-total-tokens` | Sizes the arena: its bytes are the smallest budget, within the measured one, whose sizing gives at least that many full-attention tokens, with windows and state sized by the same rules. It sizes the full-attention target; it does not cap how far full attention can grow into the arena at run time. |
| `--max-running-requests` | Bounds the concurrency the sizing solves for. The result is also capped at half the full-attention tokens. |
| `--swa-full-tokens-ratio` | Sets the window sub-pools' sizing target relative to full attention. It is a boot target, not a partition. |
| `--mamba-full-memory-ratio`, `--max-mamba-cache-size` | Set the state sub-pool's boot target: a share of the arena, or an explicit slot count. The arena may give state more slots at run time. |
| `--radix-eviction-policy joint_lru` | On full/SWA hybrids with exactly one window sub-pool and no state sub-pool, evicts full-attention and window KV in one shared least-recently-used order; elsewhere, several windows included, each component is evicted in its own least-recently-used order. See [radix eviction policies](/docs/advanced_features/radix_eviction_policy). |
| `SGLANG_DISABLE_LAZY_COMPACTION` | Must stay unset: unified memory requires lazy compaction and refuses to start with it set. |
| `SGLANG_LAZY_COMPACTION_MAX_MOVES_PER_CALL` | Caps the page moves of one opportunistic compaction pass per band, so a large backlog does not stall the scheduler loop. Urgent compaction is uncapped. |
| `SGLANG_LOG_LAZY_COMPACTION_STATS`, `SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC` | Log per-sub-pool compaction statistics periodically. |
| `SGLANG_ENABLE_WAR_BARRIER` | Not needed: unified memory turns the scheduler's write-after-read barrier on, on every device. |
| `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` | Sizes the arena after CUDA-graph capture and grows its backing in place (CUDA only). |

A prompt that cannot fit even an otherwise empty arena is aborted at admission with `Request prompt exceeds the KV memory budget`. The memory clip and admission price a request with the same demand model (`ChainAllocator.max_new_tokens_for_memory`). On a model with sliding-window and Mamba layers whose cache keeps state checkpoints, its window bound assumes the checkpoints advance; when no tracking or replacement state slot is free, a request can hold more window rows than it was priced for, so passing the clip does not guarantee that its chunks finish (see [Known limitations](#known-limitations)).

## Interaction with other features

| Feature | Status under unified memory |
|---|---|
| Overlap scheduler | Supported (the default). Freed pages stay pending until the forwards that may still touch them are processed. |
| Radix cache | Supported, with one prefix-tree slot per sub-pool. On a model with sliding-window and Mamba layers whose cache keeps state checkpoints (`--mamba-radix-cache-strategy extra_buffer`), a prefill frees window rows only behind its latest checkpoint, so every prefill chunk but a prompt's last is cut on the checkpoint grid, and a `--chunked-prefill-size` below that grid is refused at startup. |
| HiCache (`--enable-hierarchical-cache`) | Supported, with one host pool per sub-pool (the two end sub-pools may share one host backing when both hold MHA pages). `--hicache-mem-layout page_first_direct` is switched to `page_first` with the `kernel` I/O backend, because the direct copies assume packed rows. Not supported together with EAGLE/EAGLE3. With `--pp-size` above 1, `--hicache-host-memory-mode buffer_only` is refused: each stage drains its own backup completions, so the stages' eviction credit, and with it their capacity decisions, could differ. |
| PD disaggregation | Supported with `--disaggregation-transfer-backend mooncake` only. Refused with: `--pp-size` above 1, speculative decoding, `SGLANG_DISAGG_STAGING_BUFFER` (heterogeneous-TP staging), `--enable-hisparse`, and `--dcp-size` above 1. Prefill and decode must both enable unified memory, with the same attention TP size (with or without the staging buffer), page size and model. Decode KV offload needs `--hicache-storage-backend file` and does not support MLA, Mamba or sliding-window models. `--disaggregation-decode-retraction-backup host_pool` does not support Mamba models. |
| Speculative decoding | `EAGLE`, `EAGLE3` (`NEXTN` resolves to `EAGLE`), `DFLASH` and `DSPARK`, with `--speculative-eagle-topk` unset or 1. EAGLE/EAGLE3 needs a hybrid target: the draft's KV lives inside the target's sub-pool entries. Other algorithms, including the Gemma 4 assistant draft's frozen-KV MTP, are refused. |
| Pipeline parallelism | Runs without PD disaggregation. Each stage sizes its own arena, and the stages agree on the minimum of their concurrency limits. At startup they exchange their arena geometry; each stage then answers every capacity question from every stage's geometry and the request state they share, so all stages make the same decisions. Every stage needs a full-attention layer, which owns the virtual ids, and all stages must plan the same sub-pools on the same page size: a layer partition that leaves a stage without full-attention layers, or with a layer kind or window length another stage lacks, is refused. `--enable-streaming-session` and `--enable-session-radix-cache` are refused too: each stage closes timed-out sessions on its own clock, so the memory the stages hold could differ. See [Known limitations](#known-limitations). |
| Decode context parallelism (`--dcp-size` above 1) | MLA models only (for example Kimi Linear), no sliding-window layers, no PD disaggregation, and attention backends `flashinfer`, `trtllm_mla` or `cutedsl_mla`. With speculative decoding, the backend that runs target verification must be `cutedsl_mla`: the decode backend with `--speculative-attention-mode decode`, otherwise the prefill backend. With `DSPARK` or `DFLASH`, `flashinfer` is refused. |
| Data-parallel attention | Each attention-DP rank has its own arena. |

Refused at startup:

- `--enable-two-batch-overlap`
- `--enable-lmcache`, `--enable-unified-cache-external-linker`
- `--enable-kv-cache-sharding`
- `--enable-int8-mamba-checkpoint`
- `--enable-prefill-weight-versions`
- FP4 KV cache (`--kv-cache-dtype nvfp4` or `fp4_mx_block16`)
- `--cp-kv-layout` values that keep each rank's KV sharded and padded
- out-of-tree platforms

## Observability

Pool usage is reported per sub-pool: each window sub-pool has its own label, and state is reported in slots. Because sub-pools share bytes, the available figures are estimates; the held figures are exact. At idle, the scheduler checks every sub-pool for leaked pages and the arena for byte conservation. The [design page](/docs/advanced_features/unified_memory_design#debugging-and-testing) lists the stricter checks and the test tools.

## Known limitations

- **Priority preemption.** With `--enable-priority-scheduling`, unified memory turns priority preemption off, with a startup warning: a higher-priority request is admitted first but does not retract running requests, so under load it waits for running requests to finish.
- **MPS and MLX.** Refused at startup: unified memory orders the scheduler's writes after the running forward's reads with device stream waits, and the scheduler has no device streams there.
- **Window rows when state checkpoints stall.** On a model with sliding-window and Mamba layers whose cache keeps state checkpoints, the memory clip and admission assume each request's checkpoints advance. When no tracking or replacement state slot is free, a checkpoint stalls and the request holds more window rows than it was priced for, so an admitted request is not guaranteed to finish. See [the checkpoint retention bound](/docs/advanced_features/unified_memory_design#known-limitations).
- **PD failed transfers.** A failed or timed-out transfer's pages are released at its deferred-release timeout, or a failed send's at once, without proof that the remote transfer engine stopped touching them.

The [design page](/docs/advanced_features/unified_memory_design#known-limitations) names the code paths.
