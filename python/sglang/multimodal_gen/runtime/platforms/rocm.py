# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from rocm/vllm: https://github.com/ROCm/vllm/blob/v0.7.3%2Brocm/vllm/platforms/rocm.py
"""
This file is a platform abstraction for ROCm GPUs,
adjusted to match the structure and interface of `cuda.py`.
"""

import types
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ROCm uses the same torch.cuda interface
class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"  # torch uses 'cuda' backend string
    dispatch_key: str = "CUDA"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        return torch.device(f"cuda:{envs.LOCAL_RANK}")

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA graph. "
                "Since enforce-eager is enabled, async output processor cannot be used"
            )
            return False
        return True

    @classmethod
    def log_warnings(cls) -> None:
        pass  # ROCm-specific warnings can be added here

    @classmethod
    def get_current_memory_usage(cls, device: torch.device | None = None) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        if empty_cache:
            torch.cuda.empty_cache()

        free_gpu_memory, _ = torch.cuda.mem_get_info(device_id)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        elif selected_backend in (AttentionBackendEnum.FA, None):
            pass

        elif selected_backend == AttentionBackendEnum.AITER:
            if dtype not in (torch.float16, torch.bfloat16):
                logger.warning(
                    "AITer backend works best with fp16/bf16 inputs but got dtype=%s. "
                    "Proceeding with AITer anyway.",
                    dtype,
                )
            logger.info("Using AITer backend on ROCm.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerBackend"

        elif selected_backend == AttentionBackendEnum.AITER_SAGE:
            if dtype in (torch.float16, torch.bfloat16):
                logger.info("Using AITER Sage backend on ROCm.")
                return "sglang.multimodal_gen.runtime.layers.attention.backends.aiter_sage.AITERSageBackend"
            else:
                logger.warning(
                    "AITER Sage backend only supports bf16/fp16 inputs but got dtype=%s.",
                    dtype,
                )

        elif selected_backend in (
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
        ):
            raise ValueError(
                f"{selected_backend.name} is not supported on {cls.device_name}."
            )
        elif selected_backend:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}: {selected_backend}"
            )

        target_backend = AttentionBackendEnum.FA
        if dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention backend for dtype other than "
                "torch.float16 or torch.bfloat16."
            )
            target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.FA:
            try:
                import flash_attn  # noqa: F401

                from sglang.jit_kernel.flash_attention_v3 import _is_fa3_supported
                from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend,
                )

                if not _is_fa3_supported():
                    logger.info(
                        "FlashAttention backend now dispatches through FA3 "
                        "(CUDA-only). Using Torch SDPA backend on ROCm."
                    )
                    target_backend = AttentionBackendEnum.TORCH_SDPA

                if target_backend == AttentionBackendEnum.FA:
                    supported_sizes = FlashAttentionBackend.get_supported_head_sizes()
                    if head_size not in supported_sizes:
                        logger.info(
                            "Cannot use FlashAttention-2 backend for head size %d.",
                            head_size,
                        )
                        target_backend = AttentionBackendEnum.TORCH_SDPA
            except ImportError:
                logger.info(
                    "Cannot use FlashAttention backend because the "
                    "flash_attn package is not found. "
                    "Make sure that flash_attn was built and installed "
                    "(on by default)."
                )
                target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")

            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        logger.info("Using Flash Attention backend.")

        return "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn.FlashAttentionBackend"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # works for ROCm too

    @classmethod
    def optimize_vae(cls, vae: torch.nn.Module) -> torch.nn.Module:
        """Apply ROCm-specific optimizations to VAE.

        - Enable MIOpen benchmark mode so that the best convolution algorithm
          is selected for each distinct input shape (benefits Conv3d-heavy VAE
          decode).
        - Replace nn.GroupNorm with AITer GroupNorm when available.
        - Replace CausalConv3d (3x3x3) with temporal-unfolded batched Conv2D.
        """
        if envs.SGLANG_USE_ROCM_CUDNN_BENCHMARK and not torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info(
                "Enabled cudnn.benchmark (MIOpen auto-tuning) for VAE conv layers"
            )

        if envs.SGLANG_USE_ROCM_VAE:
            try:
                from aiter.ops.groupnorm import GroupNorm as AiterGroupNorm

                count = cls._replace_groupnorm(vae, AiterGroupNorm)
                if count > 0:
                    logger.info(
                        "Replaced %d nn.GroupNorm modules with AITer GroupNorm in VAE",
                        count,
                    )
            except Exception:
                logger.warning(
                    "Failed to apply AITer GroupNorm to VAE.",
                    exc_info=True,
                )

        use_bf16 = envs.SGLANG_USE_ROCM_VAE_CONV2D_BF16
        use_conv2d = envs.SGLANG_USE_ROCM_VAE_CONV2D or use_bf16
        if use_conv2d:
            count = cls._replace_conv3d_with_conv2d(vae, use_bf16=use_bf16)
            if count > 0:
                mode = "BF16" if use_bf16 else "same dtype"
                logger.info(
                    "Replaced %d CausalConv3d modules with batched Conv2D "
                    "(compute=%s) in VAE",
                    count,
                    mode,
                )

        return vae

    @staticmethod
    def _replace_groupnorm(module: torch.nn.Module, aiter_gn_cls: type) -> int:
        count = 0
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GroupNorm) and child.affine:
                replacement = aiter_gn_cls(
                    num_groups=child.num_groups,
                    num_channels=child.num_channels,
                    eps=child.eps,
                    affine=True,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                replacement.weight = child.weight
                replacement.bias = child.bias
                setattr(module, name, replacement)
                count += 1
            else:
                count += RocmPlatform._replace_groupnorm(child, aiter_gn_cls)
        return count

    @staticmethod
    def _conv3d_as_batched_conv2d(
        x_padded: torch.Tensor,
        weight_2d: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple[int, ...],
        kt: int,
        compute_bf16: bool = False,
    ) -> torch.Tensor:
        """Replace F.conv3d with temporal-unfolded batched Conv2D.

        ``x_padded`` must already be spatially/temporally padded so that
        ``F.conv3d(x_padded, weight, bias, stride, padding=0)`` would produce
        the correct output.  This routine unfolds along the temporal axis,
        reshapes into a batch of 2-D frames, runs ``F.conv2d``, and folds the
        result back.

        *weight_2d* is the pre-transformed 2-D kernel
        ``[C_out, Kt*C_in, Kh, Kw]``, cached at patch time to avoid
        redundant permute/reshape on every forward call.

        When *compute_bf16* is True the convolution is executed in BF16 and
        the output is cast back to the original dtype.
        """
        orig_dtype = x_padded.dtype
        N, C_in, T, H, W = x_padded.shape
        C_out = weight_2d.shape[0]
        stride_t, stride_h, stride_w = stride

        T_out = (T - kt) // stride_t + 1

        # (N, C_in, T, H, W) -> (N, T_out, Kt, C_in, H, W) -> (N*T_out, Kt*C_in, H, W)
        unfolded = x_padded.unfold(2, kt, stride_t)
        unfolded = unfolded.permute(0, 2, 5, 1, 3, 4).reshape(
            N * T_out, kt * C_in, H, W
        )

        w = weight_2d
        if compute_bf16 and orig_dtype != torch.bfloat16:
            unfolded = unfolded.to(torch.bfloat16)
            w = w.to(torch.bfloat16)
            b = bias.to(torch.bfloat16) if bias is not None else None
        else:
            b = bias

        out = F.conv2d(unfolded, w, b, stride=(stride_h, stride_w))

        if compute_bf16 and orig_dtype != torch.bfloat16:
            out = out.to(orig_dtype)

        _, _, H_out, W_out = out.shape
        return out.reshape(N, T_out, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _replace_conv3d_with_conv2d(
        module: torch.nn.Module, use_bf16: bool = False
    ) -> int:
        """Walk *module* and patch every CausalConv3d that has a 3-D kernel.

        A ``CausalConv3d`` is identified as any ``nn.Conv3d`` subclass that
        carries a ``_padding`` attribute (set by the Wan / diffusers causal
        conv wrapper).  Only modules whose kernel is truly 3-D (Kt>1, Kh>1,
        Kw>1) are replaced; pointwise or 1-D-temporal convolutions are left
        untouched.  Modules with non-default ``groups`` or ``dilation`` are
        skipped as the 2-D decomposition assumes groups=1 and dilation=1.
        """
        patched = 0
        skipped = 0
        for _name, child in module.named_modules():
            if not isinstance(child, nn.Conv3d):
                continue
            if not hasattr(child, "_padding"):
                continue
            kt, kh, kw = child.kernel_size
            if kt <= 1 or kh <= 1 or kw <= 1:
                skipped += 1
                continue
            if child.groups != 1 or any(d != 1 for d in child.dilation):
                skipped += 1
                continue

            padding = child._padding
            stride = child.stride

            # Pre-compute the 2-D weight: [C_out, C_in, Kt, Kh, Kw]
            # -> [C_out, Kt*C_in, Kh, Kw]  (cached as a buffer)
            weight_2d = (
                child.weight.data.permute(0, 2, 1, 3, 4)
                .reshape(child.out_channels, kt * child.in_channels, kh, kw)
                .contiguous()
            )
            child.register_buffer("_weight_2d", weight_2d)

            def _patched_forward(
                self,
                x,
                cache_x=None,
                *,
                _padding=padding,
                _stride=stride,
                _kt=kt,
                _bf16=use_bf16,
            ):
                pad = list(_padding)
                if cache_x is not None and _padding[4] > 0:
                    cache_x = cache_x.to(x.device)
                    x = torch.cat([cache_x, x], dim=2)
                    pad[4] -= cache_x.shape[2]
                x = F.pad(x, pad)
                x = x.to(self.weight.dtype)
                return RocmPlatform._conv3d_as_batched_conv2d(
                    x,
                    self._weight_2d,
                    self.bias,
                    _stride,
                    _kt,
                    compute_bf16=_bf16,
                )

            child.forward = types.MethodType(_patched_forward, child)
            patched += 1

        logger.info(
            "Conv3D→Conv2D: patched %d CausalConv3d (3D kernel, compute=%s), "
            "skipped %d (1D/pointwise/grouped)",
            patched,
            "BF16" if use_bf16 else "same dtype",
            skipped,
        )
        return patched

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        """ROCm performs better without DIT layerwise offload on Wan."""
        return False
