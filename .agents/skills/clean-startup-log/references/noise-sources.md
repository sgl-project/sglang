# Startup noise-source hints

Read this only for an observed signature that needs investigation. Paths and
behaviors can change; search the current checkout and installed dependency before
proposing a fix. The accepted-output policy in `../SKILL.md` takes precedence.
These are investigation hints, not a list of changes to apply.

## Common signatures

| Observed output | Where to investigate | Decision to verify |
|---|---|---|
| HF warning printed in two formats | `utils/common.py`, HF logger handlers and propagation | A library handler plus root propagation can emit the same record twice. Keep one timestamped copy, including before full server logging setup. |
| `Skipping import of cpp extensions due to incompatible torch version` | `torchao/__init__.py`; imports through `hf_transformers_patches.py` and transformers quantizers | Determine whether torchao is needed for the requested model before proposing narrow import-time suppression. |
| `torch_dtype` is deprecated | Model code, loader, and HF utilities accessing `config.torch_dtype` | Prefer `config.dtype` where supported; change and verify only affected, tested models. |
| `BaseImageProcessorFast` is deprecated | Multimodal processor imports and type checks | Check the installed transformers replacement and compatible `isinstance` behavior. |
| `No platform detected. Using base SRTPlatform with defaults.` | Platform detection | Decide whether fallback is expected on this machine or indicates a missing plugin. |
| `Unexpected error during package walk` in CUTE_DSL | Installed cutlass package, its handler, root propagation, and `CUTE_DSL_LOG_LEVEL` | Inspect the actual exception and duplicate-handler path; do not assume the current log level or hide all warnings. |
| Repeated dtype, hybrid-model, or tokenizer fallback messages | Model-config and tokenizer construction in launcher, scheduler, and detokenizer | Multiple constructors may explain repetition. Retain meaningful per-process information and distinguish these from handler duplication. |
| Repeated template detection messages | `managers/template_detection.py` and `managers/template_manager.py` | Check whether a summary already contains the same information. |
| KV cache dtype logged separately from allocation | Model runner and memory pool | Check whether the allocation summary already carries dtype; retain useful allocation and SWA details. |
| CUTLASS backend disabled during graph capture | Attention backend selection | Expected fallback may be useful information; retain the reason and selected behavior. |
| `Ignore import error when loading ...` | Model, multimodal, or algorithm registries | Distinguish an unused optional model dependency from failure to load the requested model. |
| `Multiple NUMA nodes found for GPU ...` | `utils/numa_utils.py` | This differs from the accepted NUMA permission warning; evaluate its context separately. |
| `OpenAI Responses API (/v1/responses) disabled` | `entrypoints/http_server.py`, serving responses, and Harmony vocabulary loading | A real endpoint limitation should remain visible even if generation warmup succeeds. |
| `SyntaxWarning: invalid escape sequence` from a dependency | Installed package file named by the warning | Identify the upstream issue; avoid editing the installed package as a repository fix. |

Repeated constructors and separate processes are not blanket reasons to downgrade
messages. Constructor counts and import paths vary by model and checkout. Likewise,
cache-summary ordering and native library formats need no change merely because
they differ from a historical reference log.

## Targeted tracing

Start with source searches. For example:

```bash
rg -n 'SEARCH_STRING' python/sglang/srt/FOCUSED_DIRECTORY
rg -n '\.torch_dtype' python/sglang/srt/models/MODEL.py
```

If the emitter is clear but its call path is not, attach a temporary handler in an
isolated reproduction:

```python
import logging
import traceback


class TraceHandler(logging.Handler):
    def emit(self, record):
        if "SEARCH_STRING" in record.getMessage():
            traceback.print_stack()


target = logging.getLogger("TARGET_LOGGER_NAME")
handler = TraceHandler()
target.addHandler(handler)
try:
    reproduce_observed_warning()
finally:
    target.removeHandler(handler)
```

For suspected native output, search only the identified shared library:

```bash
strings /path/to/library.so | rg -F 'SEARCH_STRING'
```

Keep diagnostic instrumentation out of committed fixes and baseline logs.
