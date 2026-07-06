---
name: clean-startup-log
description: Clean up noisy startup warnings and spurious prints in SGLang server logs. Use when users ask to clean up unwanted warnings, deprecation messages, or third-party noise in the server startup output.
disable-model-invocation: true
---

# Clean Up SGLang Server Startup Logs

Goal: ensure the server startup log is clean and minimal, with no spurious warnings, deprecation messages, or unformatted prints from third-party libraries.

## Workflow

### 1. Launch a server and capture the log

```bash
uv run sglang serve --model-path Qwen/Qwen3-8B 2>&1 | tee /tmp/startup_log.txt
```

Wait until the server prints `The server is fired up and ready to roll!`, then Ctrl-C.

For TP>1 testing:
```bash
uv run sglang serve --model-path Qwen/Qwen3-8B --tp 2 2>&1 | tee /tmp/startup_log.txt
```

For MoE / hybrid-SWA models (e.g. gpt-oss), test separately — they exercise different code paths:
```bash
uv run sglang serve --model-path openai/gpt-oss-20b 2>&1 | tee /tmp/startup_log.txt
```

### 2. Compare against the clean reference log

Read `/tmp/startup_log.txt` and compare it against the reference log at the bottom of this file. Identify lines that:

- Do NOT have the `[timestamp]` or `[timestamp TPx]` logger prefix
- Contain `WARNING`, `deprecated`, `is deprecated`, or similar noise
- Are printed by third-party libraries (transformers, torchao, NCCL, Gloo, tqdm, etc.)
- Are duplicate/redundant with information already logged by SGLang
- Appear multiple times due to `ModelConfig` being constructed in multiple processes

### 3. Classify each noisy line

For each noisy line, determine:

| Category | Action |
|----------|--------|
| **SGLang code using wrong API** | Fix the SGLang code (e.g., replace deprecated API with new one) |
| **SGLang code logging at wrong level** | Change log level (e.g., warning -> debug for non-actionable messages) |
| **Duplicated across processes** | Downgrade to debug — info logged in one process becomes noise in 3-4 |
| **Third-party lib prints at import time** | Suppress the logger or redirect stdout during that import |
| **C-level print from .so library** | Redirect fd 1 during the specific C call, or accept it if too invasive |
| **Real warning the user should see** | Keep it |

### 4. Present findings before fixing

List all noisy lines with their source and proposed fix. Ask the user to review before making changes.

### 5. Apply fixes and verify

After approval, apply fixes one at a time, re-launch the server, and verify each fix works.

## Key Architecture: Why Logs Repeat

`ModelConfig` is constructed **3-4 times** during startup across different processes:
1. Main process: `ServerArgs.__post_init__()` → `get_model_config()` → `ModelConfig()`
2. Scheduler subprocess: `Scheduler.init_model_config()` → `ModelConfig.from_server_args()`
3. Scheduler subprocess: `TpModelWorker._init_model_config()` → `ModelConfig.from_server_args()`
4. Main process: `TokenizerManager.init_model_config()` → `ModelConfig.from_server_args()`

Similarly, `get_tokenizer()` is called **5 times** across processes:
1. `resolve_auto_parsers` (main) — `template_detection.py`
2. `Scheduler.init_tokenizer()` (scheduler subprocess) — `scheduler.py`
3. `DetokenizerManager` (detokenizer subprocess) — `detokenizer_manager.py`
4. `TpModelWorker.__init__()` (scheduler subprocess) — `tp_worker.py`
5. `TokenizerManager` (main) — `tokenizer_manager.py`

Any `logger.info()` or `logger.warning()` in `ModelConfig.__init__()` or `get_tokenizer()` will appear 3-5 times. **Keep these at `logger.debug()`.**

## Known Noise Sources and Fixes (from past sessions)

### 1. torchao "Skipping import of cpp extensions due to incompatible torch version"

- **Source:** `torchao/__init__.py` — printed via `logger.warning()` when torch version < 2.11.0
- **Trigger:** `sglang/__init__.py` -> `_apply_hf_patches()` -> `_patch_removed_symbols()` -> `from transformers.models.llama import modeling_llama` -> deep import chain -> `transformers/quantizers/auto.py` -> `from .quantizer_torchao import TorchAoHfQuantizer` -> imports torchao
- **Fix:** In `hf_transformers_patches.py::_patch_removed_symbols()`, temporarily set the `torchao` logger level to `ERROR` around the `modeling_llama` import:
  ```python
  _torchao_logger = logging.getLogger("torchao")
  _prev_level = _torchao_logger.level
  _torchao_logger.setLevel(logging.ERROR)
  try:
      from transformers.models.llama import modeling_llama
  finally:
      _torchao_logger.setLevel(_prev_level)
  ```

### 2. "`torch_dtype` is deprecated! Use `dtype` instead!" (PARTIALLY FIXED)

- **Source:** `transformers/configuration_utils.py` — the `torch_dtype` property warns via `logger.warning_once()`
- **Trigger:** Model files accessing `config.torch_dtype` instead of `config.dtype`
- **Fix applied so far:** Only `models/gpt_oss.py` (lines 222, 471) — tested with `openai/gpt-oss-20b`.
- **Remaining files that still use `config.torch_dtype`** (fix each only after testing with the corresponding model):
  - `models/bailing_moe.py` (line 302)
  - `models/llada2.py` (line 313)
  - `models/qwen3_next.py` (lines 192, 209)
  - `models/qwen3_5.py` (line 245)
  - `models/nano_nemotron_vl.py` (lines 79, 102, 284)
  - `models/llava.py` (lines 732, 734-737)
  - `model_loader/loader.py` (line 649)
- **Note:** `common.py` was already fixed in a prior session. If new model files are added with `config.torch_dtype`, the warning will reappear — grep for `\.torch_dtype` to find them.
- **Important:** Only change `config.torch_dtype` → `config.dtype` for models you have actually tested. The `dtype` property should return the same value, but verify per-model to avoid regressions.

### 3. "`BaseImageProcessorFast` is deprecated"

- **Source:** `transformers/utils/import_utils.py` — the lazy module `__getattr__` warns when `BaseImageProcessorFast` is accessed
- **Trigger:** `base_processor.py` and `ernie45_vl.py` have `from transformers import BaseImageProcessorFast` at top level. These are imported eagerly via `tokenizer_manager.py` -> `multimodal_processor.py` -> `base_processor.py`, even for non-multimodal models.
- **Fix:** Replace `from transformers import BaseImageProcessorFast` with `from transformers import BaseImageProcessor` and update all `isinstance(..., BaseImageProcessorFast)` checks to `isinstance(..., BaseImageProcessor)`

### 4. "No platform detected. Using base SRTPlatform with defaults."

- **Source:** `sglang/srt/platforms/__init__.py` — `logger.warning()`
- **Fix:** Change to `logger.debug()` — this is expected on machines without a platform plugin and not actionable.

### 5. `NCCL version 2.27.7+cuda13.0`

- **Source:** C-level print from `libnccl.so` during `ncclCommInitRank()` call
- **Status:** Accepted as-is. SGLang already logs the version via `sglang is using nccl==X.Y.Z`. The C-level print cannot be suppressed without redirecting stdout fd, which is too invasive. `NCCL_DEBUG=WARN` does not suppress it in NCCL 2.27+.

### 6. `[Gloo] Rank X is connected to Y peer ranks`

- **Source:** C++ Gloo library print during process group init
- **Status:** Accepted as-is. From C++ code inside PyTorch's Gloo backend.

### 7. `torchao SyntaxWarning: invalid escape sequence`

- **Source:** `torchao/quantization/quant_api.py` — a raw string with unescaped `\.`
- **Status:** Upstream torchao bug. Cannot fix from SGLang side.

### 8. tqdm progress bars (e.g., `Multi-thread loading shards`, `Capturing batches`)

- **Status:** These are expected and useful. They show progress during weight loading and CUDA graph capture. Keep them.

### 9. CUTE_DSL "Unexpected error during package walk" — double-logged (FIXED)

- **Source:** `nvidia-cutlass-dsl` package at `.venv/.../cutlass/cutlass_dsl/cutlass.py`, line 391. Logger named `CUTE_DSL` with its own `StreamHandler`.
- **Trigger:** During CUDA graph capture, cutlass DSL walks packages and hits an unexpected error for `cutlass.cute.experimental`.
- **Root cause of double-logging:** The CUTE_DSL logger has `propagate=True` (default), so the warning is emitted by both the CUTE_DSL handler (with its format) and the root logger (SGLang's format).
- **Fix applied:** In `entrypoints/engine.py`, changed `CUTE_DSL_LOG_LEVEL` from `"30"` (WARNING) to `"40"` (ERROR). This suppresses the WARNING at both the CUTE_DSL logger and root propagation levels. The env var controls both `logger.setLevel()` and `console_handler.setLevel()` in cutlass's `setup_log()`.

### 10. ModelConfig init logs repeated 3x (FIXED)

- **Lines:** `"Downcasting torch.float32 to ..."`, `"Hybrid swa model: ..."`, `"DeepGemm is enabled but ..."`
- **Source:** `configs/model_config.py` — `_get_and_verify_dtype()` (line 1457), `_derive_hybrid_model()` (line 497), `_verify_quantization()` (line 1236)
- **Root cause:** `ModelConfig.__init__()` is called 3-4 times in different processes (see "Key Architecture" above). Each construction fires the same log lines.
- **Fix applied:** Downgraded all three from `logger.info()`/`logger.warning()` to `logger.debug()`. The dtype is already visible in `server_args` and `Load weight end`. Hybrid SWA info appears in `Tree cache initialized`. DeepGemm is not actionable.

### 11. Tokenizer retry/fallback messages repeated 3-4x (FIXED)

- **Lines:** `"Tokenizer loaded as generic TokenizersBackend ... retrying"`, `"Loading tokenizer ... directly as PreTrainedTokenizerFast"`, `"Tokenizer for ... loaded as generic TokenizersBackend. Set --trust-remote-code"`
- **Source:** `utils/hf_transformers/tokenizer.py` — `_resolve_tokenizers_backend()` (line 215), `_load_tokenizer_by_declared_class()` (line 110), final warning (line 244)
- **Root cause:** 5 separate `get_tokenizer()` calls across processes (see "Key Architecture" above). Each produces 3 log lines. Concurrent subprocess launches cause interleaved/doubled output.
- **Fix applied:** Downgraded all three from `logger.warning()`/`logger.info()` to `logger.debug()`.

### 12. Template detection logs — 5 lines consolidated to 1 (FIXED)

- **Lines:** `"Detected reasoning config '...' from template rule '...'"`, `"Detected reasoning parser '...' from template rule '...'"`, `"Detected tool-call parser '...' from template rule '...'"`, `"Auto-detected reasoning parser: ..."`, `"Auto-detected tool-call parser: ..."`
- **Source:** `managers/template_detection.py` (lines 337, 370) logged each detection rule match. `managers/template_manager.py` (lines 177-182) logged summary lines that duplicated the detection logs.
- **Fix applied:** Removed per-rule logs from `template_detection.py`. Consolidated the 5 lines in `template_manager.py` into a single summary: `"Auto-detected template features: reasoning_config=..., reasoning_parser=..., tool_call_parser=..."`

### 13. KV cache dtype logged separately from allocation (FIXED)

- **Lines:** `"Using KV cache dtype: torch.bfloat16"` then `"KV Cache is allocated. #tokens: ..., K size: ..., V size: ..."`
- **Source:** `model_executor/model_runner.py` (line 2217) and `mem_cache/memory_pool.py` (line 740)
- **Fix applied:** Removed the standalone dtype log from `model_runner.py`. Added `dtype` field to the allocation log in `memory_pool.py`: `"KV Cache is allocated. dtype: torch.bfloat16, #tokens: ..., K size: ..., V size: ..."`

### 14. CUTLASS backend warning — B200 → SM100, warning → info (FIXED)

- **Line:** `"CUTLASS backend is disabled when piecewise cuda graph is enabled due to TMA descriptor initialization issues on B200."`
- **Source:** `layers/attention/flashinfer_backend.py` (line 249)
- **Fix applied:** Changed "B200" to "SM100 GPUs" (the condition checks `is_sm100_supported()` which matches SM10x, not just B200). Downgraded from `logger.warning()` to `logger.info()` since it's an expected automatic fallback.

### 15. `max_total_num_tokens` and `Tree cache initialized` log ordering

- **Issue:** `max_total_num_tokens=...` appears before `Tree cache initialized:...` even though tree cache is conceptually part of memory setup.
- **Root cause:** `max_total_num_tokens` is logged inside `init_model_worker()` (scheduler.py:972), which runs before `build_kv_cache()` (scheduler.py:425) where tree cache is created.
- **Status:** Not fixed — reordering was reverted. Acceptable as-is.

### 16. `Ignore import error when loading sglang.srt.models.midashenglm`

- **Source:** `models/registry.py` (line 109) — `logger.warning()` during `import_model_classes()` which iterates all model modules via `pkgutil.iter_modules`
- **Trigger:** The `midashenglm` model depends on `torchaudio`, which fails to load
- **Status:** Should be downgraded to `logger.debug()` — not actionable when loading an unrelated model. Same pattern exists in `managers/multimodal_processor.py`, `dllm/algorithm/__init__.py`, `multimodal_gen/runtime/models/registry.py`.

### 17. `Multiple NUMA nodes found for GPU X`

- **Source:** `utils/numa_utils.py` (line 112) — `logger.warning()`
- **Status:** Could be downgraded to `logger.info()`. The situation is handled gracefully ("Using the first one") and not actionable.

### 18. Warmup `/model_info` access log

- **Source:** Uvicorn access log, triggered by SGLang's own warmup at `entrypoints/http_server.py` (line 1877)
- **Status:** SGLang talking to itself. Could suppress uvicorn access logger during warmup, or exclude `/model_info` from warmup access logging.

## Investigation Techniques

### Trace what triggers an import
```python
import sys
_real_import = __builtins__.__import__
def _tracing_import(name, *args, **kwargs):
    if 'TARGET_MODULE' in name:
        import traceback
        print(f'=== Importing {name} ===')
        traceback.print_stack()
    return _real_import(name, *args, **kwargs)
__builtins__.__import__ = _tracing_import
```

### Trace what triggers a logger warning
```python
import logging, traceback
class TraceHandler(logging.Handler):
    def emit(self, record):
        if 'SEARCH_STRING' in record.getMessage():
            traceback.print_stack()
h = TraceHandler()
h.setLevel(logging.WARNING)
logging.getLogger('TARGET_LOGGER_NAME').addHandler(h)
```

### Find C-level prints in .so files
```bash
strings /path/to/library.so | grep "SEARCH_STRING"
```

### Find all config.torch_dtype accesses (for deprecation warning)
```bash
grep -rn '\.torch_dtype' python/sglang/srt/models/ python/sglang/srt/model_loader/ python/sglang/srt/utils/hf_transformers/
```

## Reference: Clean Startup Log (TP=1, Qwen3-8B)

```
[2026-05-24 00:52:39] Attention backend not specified. Use trtllm_mha backend by default.
[2026-05-24 00:52:39] TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing page_size from None to 64.
[2026-05-24 00:52:40] server_args=ServerArgs(model_path='Qwen/Qwen3-8B', ...)
[2026-05-24 00:52:40] Multiple NUMA nodes found for GPU 0: [...]. Using the first one.
[2026-05-24 00:52:42] Using default HuggingFace chat template with detected content format: string
[2026-05-24 00:52:42] Auto-detected template features: reasoning_config=..., reasoning_parser=qwen3, tool_call_parser=qwen
[2026-05-24 00:52:50] Init torch distributed begin.
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2026-05-24 00:52:50] Init torch distributed ends. elapsed=0.21 s, mem usage=0.10 GB
[2026-05-24 00:52:51] Load weight begin. avail mem=275.75 GB
[2026-05-24 00:52:51] Found local HF snapshot for Qwen/Qwen3-8B at ...; skipping download.
Multi-thread loading shards: 100% Completed | 5/5 [00:01<00:00,  2.62it/s]
[2026-05-24 00:52:54] Load weight end. elapsed=2.62 s, type=Qwen3ForCausalLM, avail mem=260.48 GB, mem usage=15.28 GB.
[2026-05-24 00:52:54] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1707904, K size: 117.28 GB, V size: 117.28 GB
[2026-05-24 00:52:54] Memory pool end. avail mem=25.28 GB
[2026-05-24 00:52:54] CUTLASS backend is disabled when piecewise cuda graph is enabled due to TMA descriptor initialization issues on SM100 GPUs. Using auto backend instead for stability.
[2026-05-24 00:52:54] Capture cuda graph begin. This can take up to several minutes. avail mem=24.16 GB
[2026-05-24 00:52:54] Capture cuda graph bs [1, 2, 4, ...]
Capturing batches (bs=1 avail_mem=23.56 GB): 100% | 52/52 [00:05<00:00, 10.36it/s]
[2026-05-24 00:53:00] Capture cuda graph end. Time elapsed: 5.38 s. mem usage=0.60 GB. avail mem=23.56 GB.
[2026-05-24 00:53:00] Capture piecewise CUDA graph begin. avail mem=23.56 GB
[2026-05-24 00:53:00] Capture cuda graph num tokens [4, 8, 12, ...]
Compiling num tokens (num_tokens=4): 100% | 74/74 [00:09<00:00, 7.44it/s]
Capturing num tokens (num_tokens=4 avail_mem=21.24 GB): 100% | 74/74 [00:07<00:00, 10.44it/s]
[2026-05-24 00:53:18] Capture piecewise CUDA graph end. Time elapsed: 18.18 s. mem usage=2.32 GB. avail mem=21.24 GB.
[2026-05-24 00:53:20] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
[2026-05-24 00:53:20] max_total_num_tokens=1707904, chunked_prefill_size=16384, max_prefill_tokens=16384, max_running_requests=4096, context_len=40960, available_gpu_mem=21.24 GB
[2026-05-24 00:53:20] INFO:     Started server process [1964249]
[2026-05-24 00:53:20] INFO:     Waiting for application startup.
[2026-05-24 00:53:20] Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}
[2026-05-24 00:53:20] INFO:     Application startup complete.
[2026-05-24 00:53:20] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
[2026-05-24 00:53:21] Prefill batch, #new-seq: 1, #new-token: 64, ...
[2026-05-24 00:53:21] INFO:     127.0.0.1:... - "POST /generate HTTP/1.1" 200 OK
[2026-05-24 00:53:21] The server is fired up and ready to roll!
```

Note: `[Gloo]` messages and tqdm progress bars are acceptable. The key is no warnings or deprecation messages from transformers, torchao, or other third-party libraries. The `CUTLASS backend is disabled` message is now `info` level, not a warning.
