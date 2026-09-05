"""Base-MoE providers for the MoE LoRA execution engine.

The seam lives in :mod:`base.py`; each provider owns its resident weight
format, physical row domain, activation join, and finalize, and binds the
kernels that serve it -- the in-tree CuTeDSL grouped GEMM lives in
``moe/kernels/cutedsl``; the Triton arm runs sglang's fused_moe kernels.

Providers are imported explicitly by the engine rather than re-exported here,
so importing this package pulls in no kernels.  When the second provider lands
(FP8), replace the engine's direct import with a registry keyed on quant type
and resident layout — see execution plan section 29.
"""
