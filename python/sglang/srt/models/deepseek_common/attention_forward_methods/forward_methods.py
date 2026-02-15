from enum import IntEnum, auto


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use multi-head attention, execute the MHA for prefix and extended kv in a single kernel
    # when the sequence lengths are below the threshold.
    MHA_ONE_SHOT = auto()

    # Use MLA but with fused RoPE
    MLA_FUSED_ROPE = auto()

    # Use MLA with fused RoPE kernel for CPU
    MLA_FUSED_ROPE_CPU = auto()

    # Use multi-head attention for NPU
    MHA_NPU = auto()

    # Use absorbed multi-latent attention for NPU
    MLA_NPU = auto()

    # Use Deepseek V3.2 sparse multi-latent attention for NPU
    DSA_NPU = auto()
