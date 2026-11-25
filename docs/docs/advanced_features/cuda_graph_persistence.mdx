---
title: "CUDA Graph Persistence"
metatags:
    description: "Save the captured CUDA graphs once and restore them on later starts with Foundry (--cuda-graph-persistence)."
---

## Overview

CUDA graph capture (kernel warmup, attention-metadata setup and one capture per batch size) is a large part of a server's cold start. With `--cuda-graph-persistence`, [Foundry](https://github.com/foundry-org/foundry) saves the captured graphs, together with the memory layout they reference, on a first start (`save`) and rebuilds them from that archive on later starts (`load`) instead of capturing.

Foundry runs a CUDA driver hook in the scheduler processes that places every device allocation at the same address on `save` and `load`.

## Installation

Foundry is an optional dependency, published on PyPI as `foundry-core` (import name `foundry`):

```bash
pip install "sglang[foundry]"
```

The PyPI wheel (Linux x86_64) is built against the torch SGLang pins. For another torch / CUDA pair, add the pair's package index (`whl/<cuda>/torch<A.B>/`, see the [Foundry README](https://github.com/foundry-org/foundry#installation)):

```bash
pip install "sglang[foundry]" --extra-index-url https://foundry-org.github.io/foundry/whl/cu130/torch2.13/
```

## Usage

```bash
# First start: capture as usual and write the archive (./foundry_archive by default).
sglang serve --model-path Qwen/Qwen3-1.7B --cuda-graph-persistence save

# Later starts: restore the graphs from the archive.
sglang serve --model-path Qwen/Qwen3-1.7B --cuda-graph-persistence load
```

**Engines with `--cuda-graph-persistence` run with `TORCHDYNAMO_DISABLE=1`.** Foundry pins it for the scheduler processes (a contrary value disables persistence, see below): with TorchDynamo on, `save` fails at the first `@torch.compile` helper inside the capture window on torch 2.14 and later. On the validated models outputs are identical; a model whose MoE activation is a `torch.compile` fusion (gpt-oss) decodes up to ~9% slower under the pin.

`--cuda-graph-persistence-config` names an optional Foundry TOML (`workspace_root` for the archive directory, `base_addr` / `region_size` for the address region, `scratch_space_size`); its `mode` key is ignored, the flag's mode wins:

```toml
workspace_root = "/data/foundry_archive/qwen3-1.7b"
region_size = "256GB"
```

`region_size` is the address range every device allocation of a scheduler lands in after Foundry's setup: weights, KV pool, graph memory and symmetric-memory reservations under 2 GiB. Only address space is reserved up front; memory is mapped per allocation. You size it: it must cover what one GPU allocates, which at the default `--mem-fraction-static` is about the GPU's memory. The default is 512GB.

`save` and `load` must use the same model, flags, parallel layout and GPU type; remove the archive before a `save` of another configuration.

The launcher and every scheduler log one activation line, and `load` logs the restore per runner:

```text
[Foundry] cuda graph persistence active: pid=<pid> mode=load config=<toml or None>
[Foundry] Loaded <N> SGLang graphs in <s>s
```

Run engines with and without `--cuda-graph-persistence` in separate processes.

## What it changes

- Decode graphs use the `full` backend; prefill graphs are `full` or `disabled` (default `disabled`). Other backends are not supported. With decode graphs disabled there is nothing to persist: the flag has no effect and Foundry is not loaded.
- FlashInfer autotuning is off on both sides (`--disable-flashinfer-autotune`): tuning allocates timing-dependently inside the recorded memory layout, and `save` and `load` must select the same kernels. This covers every autotuned path in SGLang, so a kernel that would tune itself at run time uses its default tactic instead. To get tuned tactics for the model's FlashInfer MoE / GEMM kernels, start the model once without `--cuda-graph-persistence` (same model and flags): it writes the autotune cache file under `$SGLANG_CACHE_DIR/flashinfer/autotune/`. `save` archives those tactics and `load` installs them and tunes nothing; without that file the kernels use FlashInfer's default tactics and `save` logs a warning.
- Flags and environment variables that select state a restored graph cannot replay (custom all-reduce IPC buffers, NCCL symmetric memory / NVLS / buffer registration, MSCCL++, two-batch overlap, the memory saver, graph-pool borrow / precarve, the metadata glue graph, the DCP push-combine all-reduce) are pinned; each pin prints a `[Foundry] pin:` line, and an explicit contrary value is not supported.
- The graph backends and the autotune flag above also print a `[Foundry] pin:` line each when `--cuda-graph-persistence` changes them.
- Speculative decoding is supported for EAGLE / EAGLE3 (target-verify, draft-decode and draft-extend graphs; not adaptive or multi-layer EAGLE) and for the verify-only algorithms DFLASH, NGRAM and DSPARK (target-verify and draft-verify graphs; DSPARK with `SGLANG_DSPARK_USE_GREEDY_DRAFT=1`, since its sampling draft allocates inside the first capture); other speculative algorithms, LoRA, elastic-EP graph recapture, unified memory (`--enable-unified-memory`), post-capture KV sizing and graph variants (attention variants, sharded-verify logits variants) are not supported.

## Unsupported configurations

A configuration Foundry does not support still runs: Foundry logs the step and the reason, persistence is disabled and the server initialises natively (normal graph capture).

```text
[Foundry] unsupported at capture state: DeepEPv3Dispatcher has no prebuild; cuda-graph persistence disabled, continuing natively
```

Foundry detects this while resolving the server args, at the first capture (a MoE dispatcher without `prebuild()`, or with its state still pending) and on `save` at each capture (state created inside the graph capture). A `save` that falls back marks its archive invalid, and a `load` of it also runs natively. Once `load` starts restoring graphs, a Foundry failure is an error.

A memory failure at startup is not a fallback either: the scheduler exits and the log names the cause.

```text
# region_size too small for what the GPU allocates (cuMemAlloc / cuMemAllocPitch, or a torch symmetric-memory reservation under 2 GiB):
[HOOK] ERROR: Foundry allocation region exhausted (<call> of <n> bytes at offset <o>, region_size <s> bytes = <g> GiB); raise region_size in the Foundry TOML
# the GPU itself is full while the region still has room:
[HOOK] ERROR: GPU out of physical memory (<call>: cuMemCreate of <n> bytes at region offset <o> failed; the region still has <g> GiB of address space, so this is not region_size: lower --mem-fraction-static or the batch sizes)
# the region cannot be reserved at base_addr:
RuntimeError: [Foundry] could not reserve the allocation region at base_addr=<base> with region_size=<size> (...): choose another base_addr or a smaller region_size in the Foundry TOML
```

The first two are followed by the CUDA out-of-memory error of the allocation that failed.
