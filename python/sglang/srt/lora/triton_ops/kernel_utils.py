import triton
import triton.language as tl

from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.environ import envs


def lora_pdl_enabled() -> bool:
    """Whether to launch the LoRA Triton kernels with Programmatic Dependent Launch.

    Opt-in via ``SGLANG_LORA_ENABLE_PDL`` and AND'd with the arch check
    (``is_arch_support_pdl()`` is Hopper+, and False on HIP/MUSA where the
    ``launch_pdl`` launch kwarg would be rejected). Read per launch so the env
    var can be flipped/overridden at runtime; the arch probe is cached.
    """
    return bool(envs.SGLANG_LORA_ENABLE_PDL.get()) and is_arch_support_pdl()


def lora_pdl_launch_kwargs(enable_pdl: bool) -> dict:
    """``launch_pdl`` is an NVIDIA-only launch kwarg; the HIP backend rejects
    unknown kwargs, so only pass it when PDL is actually enabled."""
    return {"launch_pdl": True} if enable_pdl else {}


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
