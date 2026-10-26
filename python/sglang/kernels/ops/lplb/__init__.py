"""LPLB (LP-based Load Balancer) JIT kernels for expert parallelism."""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
)

# Mirrors the SM >= 9.0 check in lplb.torch_solver.
register_kernel(
    KernelSpec(
        op="lplb.solve_ipm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.lplb.cuda_solver:solve_ipm",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
    )
)
for _fn in ("prep_lp_inputs", "extract_log2phy_prob", "dispatch_probability"):
    register_kernel(
        KernelSpec(
            op=f"lplb.{_fn}",
            backend=KernelBackend.JIT,
            target=f"sglang.kernels.ops.lplb.cuda_solver:{_fn}",
            capabilities=CUDA_ONLY,
        )
    )
del _fn
