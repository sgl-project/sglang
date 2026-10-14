"""Monokernels: persistent kernels that run most of a decoder layer, including
its TP all-reduce, in one or two launches (RFC #43949). Experimental, in-tree,
one directory per model.

SGLang calls the plugin's five hooks, each a no-op while ``--enable-monokernel``
is off: ``handle_monokernel`` in the config pipeline (``guards``: refuse an
unsupported setup with every reason, set the decode capture sizes), then
``load_tensor``, ``attach``, ``route`` and ``kernel`` in each process
(``runtime``). ``spec`` holds what a model declares, and the registry.

``common/hip/`` is the shared HIP device runtime. Each model's directory holds
its spec and glue (``<model>/spec.py``) and its kernels and runner per ISA
(``<model>/<isa>/``, e.g. ``deepseek_v41/gfx950/``).

A failed wait stops its launch and flags every peer, whose waits stop too
(``common/hip/guard.py``): a rank finishes a step only with every peer's data for
it, so no token of a failed step is sent and SGLang stops the process tree.
"""

from sglang.experimental.monokernels.guards import handle_monokernel
from sglang.experimental.monokernels.runtime import attach, kernel, load_tensor, route

__all__ = ["attach", "handle_monokernel", "kernel", "load_tensor", "route"]
