"""Volta (sm70) element-wise kernels.

The generic element-wise kernels that used to live in
``sglang/srt/layers/elementwise.py`` moved upstream to
``sglang.kernels.ops.elementwise.elementwise`` (RFC #29630).  They are
re-exported here so any remaining ``sglang.srt.layers.elementwise`` import keeps
resolving; the canonical implementations are upstream's.

The modules beside this one -- ``fast_topk``, ``hc_combine``, ``hc_mix`` and
``sigmoid_mul`` -- are Volta-specific and have no upstream equivalent, which is
why this package exists at all.

Note: base's module also defined ``fused_softcap*``, ``experts_combine*``,
``Softcap`` and ``FusedDualResidualRMSNorm``.  Upstream deleted all of them and
nothing in this tree calls them, so they are not carried forward.
"""

from sglang.kernels.ops.elementwise.elementwise import *  # noqa: F401,F403
