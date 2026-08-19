"""Request-scoped fusion *policy* -- module-tree rewriting, not kernels.

A fusion whose result is not bit-exact vs the reference chain may not be on by
default: multi-step denoising amplifies per-step rounding differences into
visible quality loss.  Such fusions are mounted onto marked ``nn.Module`` sites
only for ``quality="high"`` requests, at batch boundaries, all-or-nothing per
transformer (:mod:`.quality_gate`).  Fusions that *are* bit-exact mount
unconditionally but still verify themselves against the live eager chain on
first sight and fall back permanently on mismatch (:mod:`.bitexact_gate`).

Because these modules inspect and rewrite model modules, they are the one place
in this package allowed to reference ``multimodal_gen`` types, and they do so
lazily inside functions.
"""
