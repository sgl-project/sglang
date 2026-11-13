"""Correctness-only oracle: hold GEMM reduction order fixed with real caching."""

import os

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import mm_batch_invariant

# This sitecustomize takes precedence over the optional Olmo fixture's module.
if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") == "olmo_adapter":
    import olmo_adapter  # noqa: F401

# Keep the library alive. Override only mm, without enabling deterministic
# inference globally: that mode disables radix caching with FlashInfer.
_mm_library = torch.library.Library("aten", "IMPL")
_mm_library.impl("mm", mm_batch_invariant, "CUDA")
