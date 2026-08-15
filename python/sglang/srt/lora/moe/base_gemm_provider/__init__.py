"""Base-MoE providers for the MoE LoRA execution engine.

The seam lives in :mod:`base.py`; each provider owns its resident weight
format, physical row domain, activation join, finalize, and any kernels that
only make sense in that domain.

Providers are imported explicitly by the engine rather than re-exported here,
so importing this package pulls in no kernels.  When the second provider lands
(FP8), replace the engine's direct import with a registry keyed on quant type
and resident layout — see execution plan section 29.
"""
