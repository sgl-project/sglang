from typing import Optional

_HIP_GDN_SUPPORTED_LOCAL_HEAD_SHAPES = frozenset(
    {
        (2, 4),
        (4, 8),
        (2, 8),
        (8, 16),
        (16, 32),
    }
)


def get_local_gdn_head_shape(
    total_num_k_heads: int, total_num_v_heads: int, tp_size: int
) -> tuple[int, int]:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if total_num_k_heads % tp_size != 0 or total_num_v_heads % tp_size != 0:
        raise ValueError(
            "GDN head counts must be divisible by tp_size: "
            f"num_k_heads={total_num_k_heads}, "
            f"num_v_heads={total_num_v_heads}, tp_size={tp_size}"
        )
    return total_num_k_heads // tp_size, total_num_v_heads // tp_size


def supports_hip_gdn_decode_local_heads(
    local_num_k_heads: int, local_num_v_heads: int
) -> bool:
    return (local_num_k_heads, local_num_v_heads) in (
        _HIP_GDN_SUPPORTED_LOCAL_HEAD_SHAPES
    )


def select_vk_gdn_decode_backend(
    local_num_k_heads: int,
    local_num_v_heads: int,
    hip_decode_available: bool,
    flydsl_decode_available: bool,
) -> Optional[str]:
    if hip_decode_available and supports_hip_gdn_decode_local_heads(
        local_num_k_heads, local_num_v_heads
    ):
        return "hip"
    if flydsl_decode_available:
        return "flydsl"
    return None
