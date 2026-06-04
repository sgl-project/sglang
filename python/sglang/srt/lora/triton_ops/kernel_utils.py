import triton
import triton.language as tl

from sglang.jit_kernel.utils import is_arch_support_pdl


def shapecap_dump(tag: str, **fields) -> None:
    """ADHOC shape-capture instrumentation (task 2026-06-04-qwen35-several-gemms step 1-i).

    Prints exactly one single-line tagged record per call: for each tensor arg it
    emits get_tensor_info (shape/dtype/device/stride/min/max/mean/samples), for
    scalars it emits repr. No dedup. This is debug-only and will be reverted.
    """
    import torch

    from sglang.srt.debug_utils.dumper import get_tensor_info

    parts = []
    for key, value in fields.items():
        if isinstance(value, torch.Tensor):
            info = get_tensor_info(value).replace("\n", " ")
            parts.append(f"{key}=[{info}]")
        else:
            parts.append(f"{key}={value!r}")
    print(f"[SHAPECAP {tag}] " + " ".join(parts), flush=True)


def get_pdl_launch_metadata() -> tuple[bool, dict]:
    """Return (ENABLE_PDL constexpr value, extra launch kwargs) for LoRA kernels.

    ``launch_pdl`` is NVIDIA-only Triton launch metadata; the HIP backend
    rejects unknown kwargs, so it is only included when PDL is supported.
    """
    enable_pdl = is_arch_support_pdl()
    return enable_pdl, ({"launch_pdl": True} if enable_pdl else {})


@triton.jit
def _resolve_token_positions(
    sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER: tl.constexpr
):
    """Map logical segment offsets to physical token positions.

    When SORTED_BY_ADAPTER is True, segments are grouped by adapter and
    sorted_token_ids provides the indirection to the original token rows.
    When False, tokens are already contiguous starting at seg_start.
    """
    if SORTED_BY_ADAPTER:
        return tl.load(
            sorted_token_ids + seg_start + s_offset, mask=s_offset < seg_len
        ).to(tl.int64)
    return (seg_start + s_offset).to(tl.int64)
