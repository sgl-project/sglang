"""Architecture-scoped grouping layer for CDNA 4 / gfx950 Gluon kernels.

Adds a ``cdna4/`` level under ``gluon_ops/`` so future architectures
(CDNA 5, etc.) can live alongside without name collisions. Current
tenants:

* ``extend_attention/`` -- Triton-replacement extend attention for
  MI350X, vendored from ``AMD-Triton/gluon-kernels`` branch
  ``tussingh/extend-attention-experiments``.
* ``mla_prefill/``       -- FP8 D192 MLA prefill for DeepSeek-R1.
  Migrated from ``gluon_ops/mla_prefill/`` on the MLA-prefill branch
  (whichever PR merges second rebases over the first).
"""
