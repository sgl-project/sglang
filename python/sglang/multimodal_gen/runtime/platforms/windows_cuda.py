"""Windows-specific CUDA platform policy for native diffusion runtime."""

from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatform
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WindowsCudaPlatform(CudaPlatform):
    """CUDA platform variant with conservative defaults for Windows."""

    @classmethod
    def get_torch_distributed_backend_str(cls) -> str:
        # NCCL is not available on native Windows PyTorch builds.
        return "gloo"

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        _head_size: int,
        _dtype,
    ) -> str:
        if selected_backend in (None, AttentionBackendEnum.TORCH_SDPA):
            logger.info("Using Torch SDPA backend on Windows CUDA.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        raise ValueError(
            f"Attention backend '{selected_backend.name.lower()}' is not supported on native Windows CUDA path. "
            "Use --attention-backend torch_sdpa or --backend diffusers."
        )

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        return False
