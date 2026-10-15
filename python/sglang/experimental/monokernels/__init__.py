"""Monokernels: persistent kernels that run most of a decoder layer, including
its TP all-reduce, in one or two launches (RFC #43949). Experimental.

SGLang calls the plugin's five hooks, each a no-op while ``--enable-monokernel``
is off: ``handle_monokernel`` in the config pipeline (``guards``: refuse an
unsupported setup with every reason, set the decode capture sizes), then
``load_tensor``, ``attach``, ``route`` and ``kernel`` in each process
(``runtime``). ``spec`` holds what a model declares, and the registry.

This package is the framework only. Vendor packages hold each model's spec,
glue and kernels, and list each spec under the ``sglang.monokernels`` entry
points (AMD's: sgl-amd-mono, https://github.com/sgl-project/sgl-amd-mono).
Without one, ``--enable-monokernel`` refuses to start and says why.

Every wait in a vendor's kernels has a deadline: a failed wait stops its launch
and flags every peer, whose waits stop too, so a rank finishes a step only with
every peer's data for it; no token of a failed step is sent and SGLang stops the
process tree.
"""

from sglang.experimental.monokernels.guards import handle_monokernel
from sglang.experimental.monokernels.runtime import attach, kernel, load_tensor, route

__all__ = ["attach", "handle_monokernel", "kernel", "load_tensor", "route"]
