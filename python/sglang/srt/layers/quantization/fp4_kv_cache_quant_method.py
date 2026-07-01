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
Canonical KV cache quantization methods.

This module owns the public runtime abstraction for quantized KV cache storage:

* ``KVCacheQuantMethodBase`` defines the buffer/quantize/dequantize contract.
* ``NVFP4KVCacheMethod`` and ``FP4MXBlock16KVCacheMethod`` implement the two FP4 recipes exposed by
  ``--kv-cache-dtype nvfp4`` and ``--kv-cache-dtype fp4_mx_block16``.
* ``kvfp4_tensor.py`` contains only low-level tensor/FlashInfer helpers.

Recipe selection is explicit. This file must not infer the FP4 recipe from the
GPU architecture; hardware only affects per-recipe implementation details such
as NVFP4 scale conversion.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from sglang.srt.layers.quantization.kvfp4_tensor import E2M1_MAX
from sglang.srt.utils.common import is_sm100_supported
from torch import Tensor


class KVCacheQuantMethodBase(ABC):
    """Abstract base for KV cache quantization strategies.

    Owns the quantize/dequantize computation.  The Pool owns the buffers and
    orchestrates the batch dequant loop.  Backends only do view/reshape.
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

    def load_scales_from_model(self, model_runner) -> None:
        """Load per-layer global scales from model weights (no-op by default)."""
        pass


class UnquantizedKVCacheMethod(KVCacheQuantMethodBase):
    """Identity method for BF16 / FP8 KV cache: no extra quantization."""

    name = "unquantized"
    SCALE_BLOCK_SIZE = 1

    def create_buffers(self, size, head_num, head_dim, layer_num, device) -> dict:
        pass

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
        raise RuntimeError(
            "Unquantized KV cache writes are handled by MHATokenToKVPool.set_kv_buffer."
        )

    def dequantize_prev_kv(
        self, k_fp4, k_scales, v_fp4, v_scales, layer_id
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "Unquantized KV cache does not support FP4 KV dequantization."
        )

    def compute_cell_size(
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int
    ) -> int:
        raise NotImplementedError(
            "Unquantized KV cache capacity is computed by the default pool configurator."
        )


class NVFP4KVCacheMethod(KVCacheQuantMethodBase):
    """NVFP4 two-level scaling: global FP32 + per-block FP8 E4M3.

    Supported on SM100 (B200) and SM120 (RTX 5090 / RTX PRO 6000 Blackwell).
    """

    name = "nvfp4"
    SCALE_BLOCK_SIZE = 16

    def __init__(self, num_layers: int, device: str):
        self.num_layers = num_layers
        self.device = device
        # Per-layer global FP32 scales; filled by load_scales_from_model()
        self.k_scales_gpu = torch.ones(num_layers, dtype=torch.float32, device=device)
        self.v_scales_gpu = torch.ones(num_layers, dtype=torch.float32, device=device)
        self.k_scales_float = [1.0] * num_layers
        self.v_scales_float = [1.0] * num_layers

    def needs_dequant_workspace(self) -> bool:
        return (
            True  # prefill uses FP8 dequant workspace; future native FP4 kernel → False
        )

    def needs_global_scale(self) -> bool:
        return True

    def load_scales_from_model(self, model_runner) -> None:
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
        max_global_id = max(layer.layer_id for layer in attention_layers)
        required_size = max_global_id + 1
        if required_size > len(self.k_scales_gpu):
            old_size = len(self.k_scales_gpu)
            self.k_scales_gpu = torch.ones(
                required_size, dtype=torch.float32, device=self.device
            )
            self.v_scales_gpu = torch.ones(
                required_size, dtype=torch.float32, device=self.device
            )
            extra_layers = required_size - old_size
            self.k_scales_float.extend([1.0] * extra_layers)
            self.v_scales_float.extend([1.0] * extra_layers)

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
            if is_sm100_supported():
                k_scale *= E2M1_MAX
                v_scale *= E2M1_MAX
            k_scales_cpu[layer_id] = k_scale
            v_scales_cpu[layer_id] = v_scale
            self.k_scales_float[layer_id] = k_scale
            self.v_scales_float[layer_id] = v_scale

        self.k_scales_gpu.copy_(k_scales_cpu, non_blocking=True)
        self.v_scales_gpu.copy_(v_scales_cpu, non_blocking=True)

    def get_bmm_scales(self, layer_id: int) -> tuple[float, float]:
        return self.k_scales_float[layer_id], self.v_scales_float[layer_id]

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

        cache_k = cache_k.view(torch.uint8)
        cache_v = cache_v.view(torch.uint8)
        cache_k_fp4_sf = cache_k_fp4_sf.view(torch.uint8)
        cache_v_fp4_sf = cache_v_fp4_sf.view(torch.uint8)

        k_buffer[loc] = cache_k
        v_buffer[loc] = cache_v
        k_scale_buffer[loc] = cache_k_fp4_sf
        v_scale_buffer[loc] = cache_v_fp4_sf

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


class FP4MXBlock16KVCacheMethod(KVCacheQuantMethodBase):
    """Block-16 FP4 E2M1 single-level scaling.

    This is intentionally not called MXFP4: standard MXFP4 uses a block size of
    32, while this KV cache recipe stores one scale per 16 FP4 values.
    """

    name = "fp4_mx_block16"
    SCALE_BLOCK_SIZE = 16

    def __init__(
        self,
        num_layers: Optional[int] = None,
        device: Optional[str] = None,
    ):
        pass

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
        # Block-16 FP4 flattens head dimensions for scale storage
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
        from sglang.srt.layers.quantization.kvfp4_tensor import FP4MXBlock16KVQuantizeUtil

        cache_k_fp4, cache_k_sf = FP4MXBlock16KVQuantizeUtil.batched_quantize(cache_k)
        cache_v_fp4, cache_v_sf = FP4MXBlock16KVQuantizeUtil.batched_quantize(cache_v)

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
        from sglang.srt.layers.quantization.kvfp4_tensor import FP4MXBlock16KVQuantizeUtil

        k_bf16 = FP4MXBlock16KVQuantizeUtil.batched_dequantize(k_fp4, k_scales)
        v_bf16 = FP4MXBlock16KVQuantizeUtil.batched_dequantize(v_fp4, v_scales)
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


# Registry: explicit --kv-cache-dtype value -> method class.
KV_CACHE_QUANT_REGISTRY: dict[str, type[KVCacheQuantMethodBase]] = {
    "nvfp4": NVFP4KVCacheMethod,
    "fp4_mx_block16": FP4MXBlock16KVCacheMethod,
}


def resolve_kv_cache_quant(kv_cache_dtype) -> Optional[str]:
    """Resolve the explicit FP4 KV cache recipe from ``--kv-cache-dtype``."""

    if not isinstance(kv_cache_dtype, str):
        if (
            hasattr(torch, "float4_e2m1fn_x2")
            and kv_cache_dtype == torch.float4_e2m1fn_x2
        ):
            raise ValueError(
                "FP4 KV cache storage dtype does not identify the recipe. "
                "Pass the explicit --kv-cache-dtype value: 'nvfp4' or 'fp4_mx_block16'."
            )
        return None

    if kv_cache_dtype == "fp4_e2m1":
        raise ValueError(
            "--kv-cache-dtype=fp4_e2m1 no longer auto-selects an FP4 KV recipe. "
            "Use --kv-cache-dtype=nvfp4 or --kv-cache-dtype=fp4_mx_block16."
        )
    if kv_cache_dtype == "mxfp4":
        raise ValueError(
            "--kv-cache-dtype=mxfp4 is reserved for true MXFP4 block-size-32 "
            "semantics. Use --kv-cache-dtype=fp4_mx_block16 for the current "
            "block-size-16 FP4 KV recipe."
        )
    if kv_cache_dtype in KV_CACHE_QUANT_REGISTRY:
        return kv_cache_dtype
    return None


def get_kv_cache_quant_method(name: str, **kwargs) -> KVCacheQuantMethodBase:
    """Instantiate a KVCacheQuantMethodBase by internal method name."""
    if name not in KV_CACHE_QUANT_REGISTRY:
        raise ValueError(
            f"Unknown KV cache quantization method: '{name}'. "
            f"Available: {list(KV_CACHE_QUANT_REGISTRY)}"
        )
    return KV_CACHE_QUANT_REGISTRY[name](**kwargs)
