# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
KV cache quantization strategy pattern.

Three-player design:
  quant_method (pure compute)  ►  Pool (buffer + batch dequant)  ►  Backend (view adaptation)
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from sglang.srt.layers.quantization.kvfp4_tensor import E2M1_MAX


class FP4KVCacheQuantMethod(ABC):
    """Abstract base for FP4 KV cache quantization strategies.

    Owns the quantize/dequantize computation.  The Pool owns the buffers and
    orchestrates the batch dequant loop.  Backends only do view/reshape.

    All operations (quantize_and_store, dequantize_prev_kv) use FlashInfer
    kernels or pure tensor ops, so they are CUDA-graph compatible.
    """

    name: str
    SCALE_BLOCK_SIZE: int = 1

    def needs_dequant_workspace(self) -> bool:
        """Whether the pool should allocate dq_k_buffer / dq_v_buffer for prefill."""
        return False

    def needs_global_scale(self) -> bool:
        """Whether this method uses a per-layer global FP32 scale."""
        return False

    @abstractmethod
    def create_buffers(
        self, size: int, head_num: int, head_dim: int, layer_num: int, device: str
    ) -> dict:
        """Allocate and return a buffer dict:
        {
            "k_buffer": list[Tensor],       # per-layer, shape (size, head_num, head_dim//2)
            "v_buffer": list[Tensor],
            "k_scale_buffer": list[Tensor] | None,
            "v_scale_buffer": list[Tensor] | None,
            "dq_k_buffer": Tensor | None,   # shared across layers (FP8 E4M3)
            "dq_v_buffer": Tensor | None,
            "store_dtype": torch.dtype,
        }
        """

    @abstractmethod
    def quantize_and_store(
        self,
        k_buffer: Tensor,
        v_buffer: Tensor,
        k_scale_buffer: Optional[Tensor],
        v_scale_buffer: Optional[Tensor],
        loc: Tensor,
        cache_k: Tensor,
        cache_v: Tensor,
        k_scale=None,
        v_scale=None,
    ) -> None:
        """Quantize cache_k / cache_v and write into buffers at loc."""

    @abstractmethod
    def dequantize_prev_kv(
        self,
        k_fp4: Tensor,
        k_scales: Tensor,
        v_fp4: Tensor,
        v_scales: Tensor,
        layer_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Dequantize stored FP4 KV (selected token indices already applied).

        Returns:
            (k_fp8, v_fp8): Both in torch.float8_e4m3fn dtype with shape
            matching the input (after unpacking). These are written into the
            shared dequant workspace buffer for the FlashInfer FP8 prefill kernel.
        """

    @abstractmethod
    def compute_cell_size(
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int
    ) -> int:
        """Per-token memory footprint in bytes (for capacity estimation)."""

    def load_scales_from_model(self, model_runner, sm_version: int = None) -> None:
        """Load per-layer global scales from model weights (no-op by default)."""
        pass


class NVFP4KVMethod(FP4KVCacheQuantMethod):
    """NVFP4 two-level scaling: global FP32 + per-block FP8 E4M3.

    Supported on SM100 and SM120.
    """

    name = "nvfp4"
    SCALE_BLOCK_SIZE = 16

    def __init__(self, num_layers: int, device: str, sm_version: int = 120):
        self.num_layers = num_layers
        self.device = device
        self.sm_version = sm_version
        # Per-layer global FP32 scales; filled by load_scales_from_model()
        self.k_scales_gpu = torch.ones(num_layers, dtype=torch.float32, device=device)
        self.v_scales_gpu = torch.ones(num_layers, dtype=torch.float32, device=device)

    def needs_dequant_workspace(self) -> bool:
        return (
            True  # prefill uses FP8 dequant workspace; future native FP4 kernel → False
        )

    def needs_global_scale(self) -> bool:
        return True

    def load_scales_from_model(self, model_runner, sm_version: int = None) -> None:
        if sm_version is not None:
            self.sm_version = sm_version

        from sglang.srt.model_executor.model_runner import resolve_language_model

        language_model = resolve_language_model(model_runner.model)

        attention_layers = []
        for layer in language_model.layers:
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attention_layers.append(layer.self_attn.attn)
                elif hasattr(layer.self_attn, "attn_mqa"):
                    attention_layers.append(layer.self_attn.attn_mqa)
            elif hasattr(layer, "attn"):
                attention_layers.append(layer.attn)
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    attention_layers.append(layer.attention.attn)

        if not attention_layers:
            return

        # k_scales_gpu is indexed by global (absolute) layer_id.  Resize if the model
        # has layers with global IDs larger than what was pre-allocated.
        # This happens in hybrid models (e.g., GDN) where only a subset of layers
        # are full-attention, but their layer_ids are non-contiguous.
        max_global_id = max(layer.layer_id for layer in attention_layers)
        required_size = max_global_id + 1
        if required_size > len(self.k_scales_gpu):
            self.k_scales_gpu = torch.ones(
                required_size, dtype=torch.float32, device=self.device
            )
            self.v_scales_gpu = torch.ones(
                required_size, dtype=torch.float32, device=self.device
            )

        k_scales_cpu = self.k_scales_gpu.cpu().clone()
        v_scales_cpu = self.v_scales_gpu.cpu().clone()

        for layer in attention_layers:
            layer_id = layer.layer_id  # global id
            k_scale = (
                float(layer.k_scale)
                if hasattr(layer, "k_scale") and layer.k_scale is not None
                else 1.0
            )
            v_scale = (
                float(layer.v_scale)
                if hasattr(layer, "v_scale") and layer.v_scale is not None
                else 1.0
            )
            # SM100 uses TRT-LLM XQA kernels that expect KV scales as
            # amax / 448, but the calibrated checkpoint stores amax / (6 * 448).
            # We multiply by E2M1_MAX (6.0) to bridge the gap.  SM120 uses a
            # different kernel path where scales already include this factor.
            # The FP4 data type itself is identical on both architectures.
            # Reference: TRT-LLM FP8QDQLinearMethod.process_weights_after_loading_fused_qkv_linear
            # https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/linear.py
            if self.sm_version == 100:
                k_scale *= E2M1_MAX
                v_scale *= E2M1_MAX
            k_scales_cpu[layer_id] = k_scale
            v_scales_cpu[layer_id] = v_scale

        self.k_scales_gpu.copy_(k_scales_cpu, non_blocking=True)
        self.v_scales_gpu.copy_(v_scales_cpu, non_blocking=True)

    def create_buffers(
        self, size: int, head_num: int, head_dim: int, layer_num: int, device: str
    ) -> dict:
        m = size
        n = head_num
        k = head_dim
        store_dtype = torch.uint8
        dq_dtype = torch.float8_e4m3fn

        k_buffer = [
            torch.zeros((m, n, k // 2), dtype=store_dtype, device=device)
            for _ in range(layer_num)
        ]
        v_buffer = [
            torch.zeros((m, n, k // 2), dtype=store_dtype, device=device)
            for _ in range(layer_num)
        ]
        k_scale_buffer = [
            torch.zeros(
                (m, n, k // self.SCALE_BLOCK_SIZE), dtype=store_dtype, device=device
            )
            for _ in range(layer_num)
        ]
        v_scale_buffer = [
            torch.zeros(
                (m, n, k // self.SCALE_BLOCK_SIZE), dtype=store_dtype, device=device
            )
            for _ in range(layer_num)
        ]
        # Shared dequant workspace — one copy, reused per layer during prefill
        dq_k_buffer = torch.zeros((m, n, k), dtype=dq_dtype, device=device)
        dq_v_buffer = torch.zeros((m, n, k), dtype=dq_dtype, device=device)

        return {
            "k_buffer": k_buffer,
            "v_buffer": v_buffer,
            "k_scale_buffer": k_scale_buffer,
            "v_scale_buffer": v_scale_buffer,
            "dq_k_buffer": dq_k_buffer,
            "dq_v_buffer": dq_v_buffer,
            "store_dtype": store_dtype,
        }

    def quantize_and_store(
        self,
        k_buffer: Tensor,
        v_buffer: Tensor,
        k_scale_buffer: Optional[Tensor],
        v_scale_buffer: Optional[Tensor],
        loc: Tensor,
        cache_k: Tensor,
        cache_v: Tensor,
        k_scale=None,
        v_scale=None,
    ) -> None:
        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        cache_k, cache_k_fp4_sf, _ = NVFP4KVQuantizeUtil.quantize(
            cache_k.contiguous(), k_scale
        )
        cache_v, cache_v_fp4_sf, _ = NVFP4KVQuantizeUtil.quantize(
            cache_v.contiguous(), v_scale
        )

        k_buffer[loc] = cache_k.view(torch.uint8)
        v_buffer[loc] = cache_v.view(torch.uint8)
        k_scale_buffer[loc] = cache_k_fp4_sf.view(torch.uint8)
        v_scale_buffer[loc] = cache_v_fp4_sf.view(torch.uint8)

    def dequantize_prev_kv(
        self,
        k_fp4: Tensor,
        k_scales: Tensor,
        v_fp4: Tensor,
        v_scales: Tensor,
        layer_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Dequantize FP4 KV (indexed tokens) → FP8 E4M3."""
        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        cur_k_scale = self.k_scales_gpu[layer_id : layer_id + 1]
        cur_v_scale = self.v_scales_gpu[layer_id : layer_id + 1]
        k_bf16 = NVFP4KVQuantizeUtil.dequantize(
            k_fp4.view(torch.uint8), k_scales, cur_k_scale
        )
        v_bf16 = NVFP4KVQuantizeUtil.dequantize(
            v_fp4.view(torch.uint8), v_scales, cur_v_scale
        )
        return k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)

    def compute_cell_size(
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int
    ) -> int:
        # FP4 data: per-layer, K+V
        fp4_size = head_num * (head_dim // 2) * num_layers * 2 * kv_size
        # Block scales: per-layer, K+V (uint8)
        scale_size = (
            head_num * (head_dim // self.SCALE_BLOCK_SIZE) * num_layers * 2 * kv_size
        )
        # Dequant workspace: shared across layers (not multiplied by num_layers), FP8
        dq_size = head_num * head_dim * 2 * kv_size
        return fp4_size + scale_size + dq_size


class BlockFP4KVMethod(FP4KVCacheQuantMethod):
    """Block-wise FP4 single-level scaling (similar to MXFP4 but block_size=16)."""

    name = "blockfp4"
    SCALE_BLOCK_SIZE = 16

    def needs_dequant_workspace(self) -> bool:
        return True

    def create_buffers(
        self, size: int, head_num: int, head_dim: int, layer_num: int, device: str
    ) -> dict:
        m = size
        store_dtype = torch.uint8
        dq_dtype = torch.float8_e4m3fn

        k_buffer = [
            torch.zeros((m, head_num, head_dim // 2), dtype=store_dtype, device=device)
            for _ in range(layer_num)
        ]
        v_buffer = [
            torch.zeros((m, head_num, head_dim // 2), dtype=store_dtype, device=device)
            for _ in range(layer_num)
        ]
        # MXFP4 flattens head dimensions for scale storage
        k_scale_buffer = [
            torch.zeros(
                (m, (head_num * head_dim) // self.SCALE_BLOCK_SIZE),
                dtype=store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        v_scale_buffer = [
            torch.zeros(
                (m, (head_num * head_dim) // self.SCALE_BLOCK_SIZE),
                dtype=store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        dq_k_buffer = torch.zeros(
            (m, head_num, head_dim), dtype=dq_dtype, device=device
        )
        dq_v_buffer = torch.zeros(
            (m, head_num, head_dim), dtype=dq_dtype, device=device
        )

        return {
            "k_buffer": k_buffer,
            "v_buffer": v_buffer,
            "k_scale_buffer": k_scale_buffer,
            "v_scale_buffer": v_scale_buffer,
            "dq_k_buffer": dq_k_buffer,
            "dq_v_buffer": dq_v_buffer,
            "store_dtype": store_dtype,
        }

    def quantize_and_store(
        self,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        loc,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
    ) -> None:
        from sglang.srt.layers.quantization.kvfp4_tensor import BlockFP4KVQuantizeUtil

        cache_k_fp4, cache_k_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_k)
        cache_v_fp4, cache_v_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_v)
        k_buffer[loc] = cache_k_fp4
        v_buffer[loc] = cache_v_fp4
        k_scale_buffer[loc] = cache_k_sf
        v_scale_buffer[loc] = cache_v_sf

    def dequantize_prev_kv(
        self,
        k_fp4: Tensor,
        k_scales: Tensor,
        v_fp4: Tensor,
        v_scales: Tensor,
        layer_id: int,
    ) -> tuple[Tensor, Tensor]:
        from sglang.srt.layers.quantization.kvfp4_tensor import BlockFP4KVQuantizeUtil

        k_bf16 = BlockFP4KVQuantizeUtil.batched_dequantize(k_fp4, k_scales)
        v_bf16 = BlockFP4KVQuantizeUtil.batched_dequantize(v_fp4, v_scales)
        return k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)

    def compute_cell_size(
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int
    ) -> int:
        fp4_size = head_num * (head_dim // 2) * num_layers * 2 * kv_size
        scale_size = (
            (head_num * head_dim // self.SCALE_BLOCK_SIZE) * num_layers * 2 * kv_size
        )
        dq_size = head_num * head_dim * 2 * kv_size
        return fp4_size + scale_size + dq_size


# Registry: name → class.  Only classes for fp4_e2m1 dtype need to be listed.
FP4_KV_CACHE_QUANT_REGISTRY: dict[str, type[FP4KVCacheQuantMethod]] = {
    "nvfp4": NVFP4KVMethod,
    "blockfp4": BlockFP4KVMethod,
}


def get_fp4_kv_cache_quant_method(name: str, **kwargs) -> FP4KVCacheQuantMethod:
    """Instantiate a FP4KVCacheQuantMethod by recipe name."""
    if name not in FP4_KV_CACHE_QUANT_REGISTRY:
        raise ValueError(
            f"Unknown fp4_kv_cache_recipe: '{name}'. "
            f"Available: {list(FP4_KV_CACHE_QUANT_REGISTRY)}"
        )
    return FP4_KV_CACHE_QUANT_REGISTRY[name](**kwargs)
