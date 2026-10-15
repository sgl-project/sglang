"""Monokernels: persistent kernels that run most of a decoder layer, including its TP all-reduce, in one or two
launches (RFC #43949).

SGLang calls the plugin's hooks, each a no-op while ``--enable-monokernel`` is off: ``handle_monokernel`` in the
config pipeline (``guards``: refuse an unsupported setup with every reason), then ``load_tensor`` / ``loaded``,
``attach``, ``route`` and ``kernel`` in each process (``runtime``). ``spec`` holds what a model declares and the
registry of specs. Shared device primitives live in ``common_<vendor>/``, each model's kernels, spec and glue in
``<model>_<vendor>/``.

A failed wait stops its launch and flags every peer, whose waits stop too (``common_hip/guard.py``): a rank finishes a
step only with every peer's data for it, so no token of a failed step is sent and SGLang stops the process tree.
"""

from sglang.kernels.ops.mono.guards import handle_monokernel
from sglang.kernels.ops.mono.runtime import attach, kernel, load_tensor, loaded, route

__all__ = ["attach", "handle_monokernel", "kernel", "load_tensor", "loaded", "route"]
