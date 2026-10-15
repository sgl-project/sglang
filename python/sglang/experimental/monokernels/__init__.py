"""Monokernels: persistent kernels that run most of a decoder layer, including
its TP all-reduce, in one or two launches (RFC #43949). Experimental.

SGLang's runtime calls three hooks, for every model, each a no-op while
``--enable-monokernel`` is off: ``handle_monokernel`` in the config pipeline
(``guards``: refuse an unsupported setup with every reason, set the decode capture
sizes), ``watch_weights`` in the model loader, and ``attach`` once the model is
loaded (``runtime``), which switches the model's forward: each step a spec serves
runs the spec's forward, and every other step the model's own. No model code
changes. ``spec`` holds what a model declares, and the registry.

This package is the framework only. Vendor packages hold each model's spec,
forward and kernels; SGLang loads them as general plugins (``sglang.srt.plugins``
entry points), and each calls ``register`` for its specs (AMD's: sgl-amd-mono,
https://github.com/sgl-project/sgl-amd-mono). Without one,
``--enable-monokernel`` refuses to start and says why.

Every wait in a vendor's kernels has a deadline: a failed wait stops its launch
and flags every peer, whose waits stop too, so a rank finishes a step only with
every peer's data for it; no token of a failed step is sent and SGLang stops the
process tree.
"""

from sglang.experimental.monokernels.guards import handle_monokernel
from sglang.experimental.monokernels.runtime import attach, watch_weights
from sglang.experimental.monokernels.spec import API_VERSION, register

__all__ = ["API_VERSION", "attach", "handle_monokernel", "register", "watch_weights"]
