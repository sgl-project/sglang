"""Chunked KDA prefill kernel (pypto-gym commit 5405139, unmodified).

``chunk_kda_wrapper`` is imported lazily from ``chunk_kda_impl`` by callers (not here)
so that ``import pypto`` only happens on first use, keeping sglang startup free of the
pypto dependency on non-NPU / fresh environments.
"""
