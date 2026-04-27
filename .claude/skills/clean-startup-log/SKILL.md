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

### 2. Compare against the clean reference log

Read `/tmp/startup_log.txt` and compare it against the reference log at the bottom of this file. Identify lines that:

- Do NOT have the `[timestamp]` or `[timestamp TPx]` logger prefix
- Contain `WARNING`, `deprecated`, `is deprecated`, or similar noise
- Are printed by third-party libraries (transformers, torchao, NCCL, Gloo, tqdm, etc.)
- Are duplicate/redundant with information already logged by SGLang

### 3. Classify each noisy line

For each noisy line, determine:

| Category | Action |
|----------|--------|
| **SGLang code using wrong API** | Fix the SGLang code (e.g., replace deprecated API with new one) |
| **SGLang code logging at wrong level** | Change log level (e.g., warning -> debug for non-actionable messages) |
| **Third-party lib prints at import time** | Suppress the logger or redirect stdout during that import |
| **C-level print from .so library** | Redirect fd 1 during the specific C call, or accept it if too invasive |
| **Real warning the user should see** | Keep it |

### 4. Present findings before fixing

List all noisy lines with their source and proposed fix. Ask the user to review before making changes.

### 5. Apply fixes and verify

After approval, apply fixes one at a time, re-launch the server, and verify each fix works.

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

### 2. "`torch_dtype` is deprecated! Use `dtype` instead!"

- **Source:** `transformers/configuration_utils.py` — the `torch_dtype` property warns via `logger.warning_once()`
- **Trigger:** `get_hf_text_config()` in `sglang/srt/utils/hf_transformers/common.py` accesses `config.torch_dtype`
- **Fix:** Replace all `getattr(config, "torch_dtype", ...)` with `getattr(config, "dtype", ...)` and `config.torch_dtype = X` with `config.dtype = X` in `common.py`

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

## Reference: Clean Startup Log (TP=1, Qwen3-8B)

```
[2026-04-27 02:35:53] Attention backend not specified. Use trtllm_mha backend by default.
[2026-04-27 02:35:53] TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing page_size from None to 64.
[2026-04-27 02:35:54] server_args=ServerArgs(model_path='Qwen/Qwen3-8B', ...)
[2026-04-27 02:35:56] Using default HuggingFace chat template with detected content format: string
[2026-04-27 02:36:03] Init torch distributed begin.
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2026-04-27 02:36:03] Init torch distributed ends. elapsed=0.27 s, mem usage=0.09 GB
[2026-04-27 02:36:04] Load weight begin. avail mem=177.57 GB
[2026-04-27 02:36:04] Found local HF snapshot for Qwen/Qwen3-8B at ...; skipping download.
Multi-thread loading shards: 100% Completed | 5/5 [00:01<00:00,  3.08it/s]
[2026-04-27 02:36:06] Load weight end. elapsed=1.97 s, type=Qwen3ForCausalLM, avail mem=162.30 GB, mem usage=15.28 GB.
[2026-04-27 02:36:06] Using KV cache dtype: torch.bfloat16
[2026-04-27 02:36:06] KV Cache is allocated. #tokens: 992896, K size: 68.18 GB, V size: 68.18 GB
[2026-04-27 02:36:06] Memory pool end. avail mem=25.26 GB
[2026-04-27 02:36:06] Capture cuda graph begin. This can take up to several minutes. avail mem=24.14 GB
[2026-04-27 02:36:06] Capture cuda graph bs [1, 2, 4, ...]
Capturing batches (bs=1 avail_mem=23.54 GB): 100% | 52/52 [00:03<00:00, 16.76it/s]
[2026-04-27 02:36:09] Capture cuda graph end. Time elapsed: 3.74 s. mem usage=0.60 GB. avail mem=23.54 GB.
[2026-04-27 02:36:09] Capture piecewise CUDA graph begin. avail mem=23.54 GB
[2026-04-27 02:36:09] Capture cuda graph num tokens [4, 8, 12, ...]
Compiling num tokens (num_tokens=4): 100% | 74/74 [00:09<00:00, 8.16it/s]
Capturing num tokens (num_tokens=4 avail_mem=21.23 GB): 100% | 74/74 [00:08<00:00, 9.11it/s]
[2026-04-27 02:36:27] Capture piecewise CUDA graph end. Time elapsed: 17.62 s. mem usage=2.32 GB. avail mem=21.22 GB.
[2026-04-27 02:36:28] max_total_num_tokens=992896, chunked_prefill_size=16384, ...
[2026-04-27 02:36:29] INFO:     Started server process [399368]
[2026-04-27 02:36:29] INFO:     Waiting for application startup.
[2026-04-27 02:36:29] Using default chat sampling params from model generation config: ...
[2026-04-27 02:36:29] INFO:     Application startup complete.
[2026-04-27 02:36:29] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
[2026-04-27 02:36:30] Prefill batch, #new-seq: 1, #new-token: 64, ...
[2026-04-27 02:36:30] INFO:     127.0.0.1:34916 - "POST /generate HTTP/1.1" 200 OK
[2026-04-27 02:36:30] The server is fired up and ready to roll!
```

Note: `[Gloo]` messages and tqdm progress bars are acceptable. The key is no warnings or deprecation messages from transformers, torchao, or other third-party libraries.
