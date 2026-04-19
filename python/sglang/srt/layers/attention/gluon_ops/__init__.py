"""Gluon kernel sources vendored into SGLang for the opt-in Gluon
attention paths on MI350X (gfx950 / CDNA 4).

This package holds the kernel code imported by
``sglang.srt.layers.attention.gluon_mla_prefill`` and
``sglang.srt.layers.attention.gluon_extend_attention``. They are plain
Triton / Gluon kernels with no native ``sgl_kernel`` dependency, so they
ship as ``.py`` source files and are compiled on first use (and on
prewarm at server boot).

Each sub-package is self-contained (its own copies of ``_common.py`` and
``_layouts.py``) so the wrappers never have to reach outside the sglang
tree.
"""
