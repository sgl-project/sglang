"""Backwards-compatibility shim for the unified LLM torch-profiler entrypoint.

The real implementation now lives in ``analyze_llm_torch_profile`` because this
skill covers SGLang, vLLM, and TensorRT-LLM. Older scripts and runbooks that
still invoke ``analyze_sglang_torch_profile.py`` keep working by forwarding to
that module.
"""

from __future__ import annotations

import sys

from analyze_llm_torch_profile import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
