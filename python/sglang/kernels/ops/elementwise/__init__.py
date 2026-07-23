"""Generic elementwise / fused-pointwise kernels.

Home for cross-cutting pointwise kernels that do not belong to a single
functional group: the fused-pointwise Triton collection (``elementwise``:
softcap, sigmoid-mul, gated-activation and fused-rmsnorm variants shared
across models) and the ``add_constant`` JIT reference kernel used by the
developer guide. Individual functions register (or are imported) under the
functional op id they logically belong to.
"""

__all__ = []
