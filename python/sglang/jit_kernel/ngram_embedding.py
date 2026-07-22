"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.speculative.ngram_embedding."""

from sglang.kernels.ops.speculative import ngram_embedding as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
