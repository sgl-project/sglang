"""Monokernels: persistent kernels that run most of a decoder layer, including
its TP all-reduce, in one or two launches (RFC #43949).

Shared device primitives live in ``common_<vendor>/`` and each model's kernels
in ``<model>_<vendor>/``, next to its spec (``<model>_<vendor>/spec.py``).
``hooks`` holds the calls SGLang makes and ``spec`` what a model declares. The
kernels are reached only through a spec, so the group registers no KernelSpec:
its launch builders and runners are not kernel APIs.
"""

__all__ = []
