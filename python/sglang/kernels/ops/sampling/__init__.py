"""Sampling kernels (top-k / top-p renorm, sampling-from-probs, ...).

Reserved group in the ``sglang.kernels`` namespace (RFC #29630). No public
wrappers are exposed here yet; the implementations still live under ``sgl_kernel.sampling`` / ``sgl_kernel.top_k``. These will be migrated into this group
in later phases. Until then, import the underlying implementations from
``sglang.jit_kernel`` / ``sgl_kernel`` / the relevant ``triton_ops`` module
directly.
"""

__all__ = []
