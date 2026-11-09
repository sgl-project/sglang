"""KV-cache canary kernels: plan, write and verify canary slots, and scatter
each request's expected token ids into its pool row."""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

# The launchers fall back to their torch references off CUDA, so no device pin.
for _mod, _fn, _backend in (
    ("plan.api", "launch_canary_plan_kernels", KernelBackend.JIT),
    ("write", "launch_canary_write_kernel", KernelBackend.JIT),
    ("verify", "launch_canary_verify_kernel", KernelBackend.JIT),
    ("canary_info", "launch_canary_info_kernel", KernelBackend.JIT),
    (
        "scatter_req_token_ids",
        "launch_scatter_req_token_ids_kernel",
        KernelBackend.TRITON,
    ),
):
    register_kernel(
        KernelSpec(
            op=f"kv_canary.{_fn}",
            backend=_backend,
            target=f"sglang.kernels.ops.kv_canary.{_mod}:{_fn}",
        )
    )
del _mod, _fn, _backend
