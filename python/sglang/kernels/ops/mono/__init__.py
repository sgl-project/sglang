"""Monokernels: persistent kernels that run most of a decoder layer, including
its TP all-reduce, in one or two launches (RFC #43949).

SGLang calls the plugin's five hooks, each a no-op while ``--enable-monokernel``
is off: ``handle_monokernel`` in the config pipeline (``guards``: refuse an
unsupported setup with every reason, set the decode capture sizes), then
``load_tensor``, ``attach``, ``route`` and ``kernel`` in each process
(``runtime``). ``spec`` holds what a model declares.

Shared device primitives live in ``common_<vendor>/`` and each model's kernels
in ``<model>_<vendor>/``, next to its spec (``<model>_<vendor>/spec.py``). The
kernels are reached only through a spec, so the group registers no KernelSpec:
its launch builders and runners are not kernel APIs.

A failed wait stops its launch and flags every peer, whose waits stop too
(``common_hip/guard.py``): a rank finishes a step only with every peer's data for
it, so no token of a failed step is sent and SGLang stops the process tree.
"""

from sglang.kernels.ops.mono.guards import handle_monokernel
from sglang.kernels.ops.mono.runtime import attach, kernel, load_tensor, route

__all__ = ["attach", "handle_monokernel", "kernel", "load_tensor", "route"]
