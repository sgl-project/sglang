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

Why do we need attention access rules?
- Problem: torch.float4_e2m1fn_x2 only describes packed FP4 storage. It does
  not say whether the recipe is NVFP4 or fp4_mx_block16, nor how scales are
  interpreted.
- Problem: prefill and decode may use different KV views for the same recipe.
  For example, NVFP4 uses a FlashInfer dequant workspace for prefill, but TRTLLM
  MHA consumes native packed FP4 plus scales for decode.
- Problem: putting these recipe/backend combinations directly into each backend
  as dtype checks makes unsupported paths hard to spot and future recipes hard
  to add safely.
- Approach: the bottom registry declares KVCacheAttentionAccess entries for
  each recipe. A backend resolves one entry by (phase, backend_name, tags),
  then either uses the declared access pattern or fails fast if that
  combination is unsupported.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import torch
from torch import Tensor

from sglang.srt.layers.quantization.kvfp4_tensor import E2M1_MAX
from sglang.srt.utils.common import is_sm100_supported


class KVCacheAttentionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class KVCacheAttentionAccessKind(str, Enum):
    # KV cache is already in the dtype/layout expected by the attention backend.
    PLAIN = "plain"
    # KV cache is stored quantized, then dequantized and decompressed into a
    # temporary workspace before attention.
    DEQUANT_WORKSPACE = "dequant_workspace"
    # Attention backend directly consumes FP4 KV cache storage and scales.
    NATIVE_FP4 = "native_fp4"


@dataclass(frozen=True)
class KVCacheBackendMatcher:
    exact: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()
    any_backend: bool = False

    def matches(self, backend_name: str, backend_tags: Iterable[str]) -> bool:
        backend_tags = frozenset(backend_tags)
        return (
            self.any_backend
            or backend_name in self.exact
            or (bool(self.tags) and self.tags.issubset(backend_tags))
        )


@dataclass(frozen=True)
class KVCacheAttentionAccess:
    """Describes how one attention backend reads KV cache for one phase.

    Fields:
    - phase: prefill or decode stage where this rule applies.
    - kind: access mode, such as plain KV, dequant workspace, or native FP4.
    - backend_matcher: backend names/tags that select this rule, e.g.
      exact={"trtllm_mha"} or a FlashInfer dequant-workspace tag.
    - storage_dtype: dtype stored in the KV pool, e.g. torch.float4_e2m1fn_x2.
    - attention_kv_dtype: dtype consumed by attention after any conversion, e.g.
      FP8 workspace for FlashInfer prefill or packed FP4 for TRTLLM decode.
    - scale_recipe: scale semantics for this FP4 recipe, e.g. "nvfp4" or
      "fp4_mx_block16".
    - workspace_dtype: temporary workspace dtype for dequant/decompress paths;
      None means no temporary workspace is needed.
    """

    phase: KVCacheAttentionPhase
    kind: KVCacheAttentionAccessKind
    backend_matcher: KVCacheBackendMatcher
    storage_dtype: Optional[torch.dtype] = None
    attention_kv_dtype: Optional[torch.dtype] = None
    scale_recipe: Optional[str] = None
    workspace_dtype: Optional[torch.dtype] = None

    def matches(self, phase, backend_name: str, backend_tags: Iterable[str]) -> bool:
        return self.phase == KVCacheAttentionPhase(
            phase
        ) and self.backend_matcher.matches(backend_name, backend_tags)


class KVCacheQuantMethodBase(ABC):
    """Abstract base for KV cache quantization strategies.

    Owns the quantize/dequantize computation.  The Pool owns the buffers and
    orchestrates the batch dequant loop.  Backends only do view/reshape.
    """

    name: str
    SCALE_BLOCK_SIZE: int = 1

    def attention_accesses(self) -> tuple[KVCacheAttentionAccess, ...]:
        return KV_CACHE_ATTENTION_ACCESS_REGISTRY.get(self.name, ())

    def resolve_attention_access(
        self, phase, backend_name: str, backend_tags: Iterable[str] = ()
    ) -> Optional[KVCacheAttentionAccess]:
        for access in self.attention_accesses():
            if access.matches(phase, backend_name, backend_tags):
                return access
        return None

    def describe_attention_accesses(self, phase=None) -> str:
        accesses = self.attention_accesses()
        if phase is not None:
            phase = KVCacheAttentionPhase(phase)
            accesses = tuple(access for access in accesses if access.phase == phase)
        if not accesses:
            return "none"
        return "; ".join(
            f"{access.phase.value}:{access.kind.value}:"
            f"exact={sorted(access.backend_matcher.exact)}:"
            f"tags={sorted(access.backend_matcher.tags)}"
            for access in accesses
        )

    def needs_dequant_workspace(self) -> bool:
        """Whether the pool should allocate dq_k_buffer / dq_v_buffer."""
        return any(
            access.kind == KVCacheAttentionAccessKind.DEQUANT_WORKSPACE
            for access in self.attention_accesses()
        )

    def needs_plain_kv_dequant_read(self) -> bool:
        """Whether plain attention reads require dequantizing packed KV first."""
        return any(
            access.kind == KVCacheAttentionAccessKind.PLAIN
            and access.storage_dtype is not None
            for access in self.attention_accesses()
        )

    def dequant_workspace_dtype(self) -> Optional[torch.dtype]:
        """Workspace dtype required by DEQUANT_WORKSPACE access rules."""
        workspace_dtypes = set()
        for access in self.attention_accesses():
            if access.kind != KVCacheAttentionAccessKind.DEQUANT_WORKSPACE:
                continue
            if access.workspace_dtype is None:
                raise ValueError(
                    f"KV cache method {self.name!r} declares DEQUANT_WORKSPACE "
                    "without workspace_dtype."
                )
            workspace_dtypes.add(access.workspace_dtype)

        if not workspace_dtypes:
            return None
        if len(workspace_dtypes) != 1:
            raise ValueError(
                f"KV cache method {self.name!r} declares multiple dequant "
                f"workspace dtypes: {sorted(str(dtype) for dtype in workspace_dtypes)}."
            )
        return next(iter(workspace_dtypes))

    def kv_storage_dtype(self) -> torch.dtype:
        """Packed KV storage dtype declared by attention access rules."""
        storage_dtypes = {
            access.storage_dtype
            for access in self.attention_accesses()
            if access.storage_dtype is not None
        }
        if not storage_dtypes:
            return torch.uint8
        if len(storage_dtypes) != 1:
            raise ValueError(
                f"KV cache method {self.name!r} declares multiple storage "
                f"dtypes: {sorted(str(dtype) for dtype in storage_dtypes)}."
            )
        return next(iter(storage_dtypes))

    def plain_attention_kv_dtype(self) -> Optional[torch.dtype]:
        """Dtype produced when packed KV is dequantized for PLAIN attention."""
        attention_dtypes = {
            access.attention_kv_dtype
            for access in self.attention_accesses()
            if access.kind == KVCacheAttentionAccessKind.PLAIN
            and access.storage_dtype is not None
            and access.attention_kv_dtype is not None
        }
        if not attention_dtypes:
            return None
        if len(attention_dtypes) != 1:
            raise ValueError(
                f"KV cache method {self.name!r} declares multiple plain attention "
                f"KV dtypes: {sorted(str(dtype) for dtype in attention_dtypes)}."
            )
        return next(iter(attention_dtypes))

    def needs_global_scale(self) -> bool:
        """Whether this method uses a per-layer global FP32 scale."""
        return False

    def scale_buffer_view_dtype(self) -> Optional[torch.dtype]:
        """Optional dtype view for stored per-block scales."""
        return None

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
            Dequantized K/V tensors with shape matching the input after unpacking.
        """

    def dequantize_kv_tensor(
        self,
        fp4_tensor: Tensor,
        scales: Tensor,
        layer_id: int,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Dequantize one packed FP4 KV tensor for plain attention reads."""
        raise NotImplementedError(
            f"KV cache method {self.name!r} does not support plain KV dequant reads."
        )

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

    Supported on SM100 and SM120.
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

    def needs_global_scale(self) -> bool:
        return True

    def scale_buffer_view_dtype(self) -> Optional[torch.dtype]:
        return torch.float8_e4m3fn

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
        store_dtype = self.kv_storage_dtype()
        dq_dtype = self.dequant_workspace_dtype()

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
        # Shared dequant workspace: one copy, reused per layer during prefill.
        dq_k_buffer = (
            torch.zeros((m, n, k), dtype=dq_dtype, device=device)
            if dq_dtype is not None
            else None
        )
        dq_v_buffer = (
            torch.zeros((m, n, k), dtype=dq_dtype, device=device)
            if dq_dtype is not None
            else None
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
        # Dequant workspace is shared across layers, not multiplied by num_layers.
        dq_dtype = self.dequant_workspace_dtype()
        dq_size = (
            head_num
            * head_dim
            * 2
            * kv_size
            * torch.empty((), dtype=dq_dtype).element_size()
            if dq_dtype is not None
            else 0
        )
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

    def create_buffers(
        self, size: int, head_num: int, head_dim: int, layer_num: int, device: str
    ) -> dict:
        m = size
        store_dtype = self.kv_storage_dtype()
        dq_dtype = self.dequant_workspace_dtype()

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
        dq_k_buffer = (
            torch.zeros((m, head_num, head_dim), dtype=dq_dtype, device=device)
            if dq_dtype is not None
            else None
        )
        dq_v_buffer = (
            torch.zeros((m, head_num, head_dim), dtype=dq_dtype, device=device)
            if dq_dtype is not None
            else None
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
        from sglang.srt.layers.quantization.kvfp4_tensor import (
            FP4MXBlock16KVQuantizeUtil,
        )

        cache_k_fp4, cache_k_sf = FP4MXBlock16KVQuantizeUtil.batched_quantize(cache_k)
        cache_v_fp4, cache_v_sf = FP4MXBlock16KVQuantizeUtil.batched_quantize(cache_v)

        k_buffer[loc] = cache_k_fp4
        v_buffer[loc] = cache_v_fp4
        k_scale_buffer[loc] = cache_k_sf
        v_scale_buffer[loc] = cache_v_sf

    def dequantize_kv_tensor(
        self,
        fp4_tensor: Tensor,
        scales: Tensor,
        layer_id: int,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        from sglang.srt.layers.quantization.kvfp4_tensor import (
            FP4MXBlock16KVQuantizeUtil,
        )

        target_dtype = dtype or self.plain_attention_kv_dtype() or torch.bfloat16
        return FP4MXBlock16KVQuantizeUtil.batched_dequantize(
            fp4_tensor, scales, dtype=target_dtype
        )

    def dequantize_prev_kv(
        self,
        k_fp4: Tensor,
        k_scales: Tensor,
        v_fp4: Tensor,
        v_scales: Tensor,
        layer_id: int,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.dequantize_kv_tensor(k_fp4, k_scales, layer_id),
            self.dequantize_kv_tensor(v_fp4, v_scales, layer_id),
        )

    def compute_cell_size(
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int
    ) -> int:
        fp4_size = head_num * (head_dim // 2) * num_layers * 2 * kv_size
        scale_size = (
            (head_num * head_dim // self.SCALE_BLOCK_SIZE) * num_layers * 2 * kv_size
        )
        dq_dtype = self.dequant_workspace_dtype()
        dq_size = (
            head_num
            * head_dim
            * 2
            * kv_size
            * torch.empty((), dtype=dq_dtype).element_size()
            if dq_dtype is not None
            else 0
        )
        return fp4_size + scale_size + dq_size


# Registry: method name -> attention access rules.
_PREFILL = KVCacheAttentionPhase.PREFILL
_DECODE = KVCacheAttentionPhase.DECODE
_PLAIN_KIND = KVCacheAttentionAccessKind.PLAIN
_DQ_WORKSPACE_KIND = KVCacheAttentionAccessKind.DEQUANT_WORKSPACE
_NATIVE_FP4_KIND = KVCacheAttentionAccessKind.NATIVE_FP4
_ANY_BACKEND = KVCacheBackendMatcher(any_backend=True)
_NVFP4_SCALE = "nvfp4"
_FP4_MX_SCALE = "fp4_mx_block16"
_FP8_E4M3 = torch.float8_e4m3fn
_TORCH_FP4 = getattr(torch, "float4_e2m1fn_x2", None)
_BF16 = torch.bfloat16
_NVFP4_PREFILL_BACKENDS = frozenset({"flashinfer"})
_NVFP4_DECODE_BACKENDS = frozenset({"trtllm_mha"})
_FP4_MX_MHA_BACKENDS = frozenset(
    {"triton", "torch_native", "flex_attention", "trtllm_mha"}
)
_FP4_MX_PREFILL_BACKENDS = _FP4_MX_MHA_BACKENDS | frozenset({"fa4"})


def _backend_matcher(backends) -> KVCacheBackendMatcher:
    if isinstance(backends, KVCacheBackendMatcher):
        return backends
    return KVCacheBackendMatcher(exact=backends)


def _plain(
    phase: KVCacheAttentionPhase,
    backends,
    scale: Optional[str] = None,
    attention_dtype: Optional[torch.dtype] = None,
) -> KVCacheAttentionAccess:
    return KVCacheAttentionAccess(
        phase,
        _PLAIN_KIND,
        _backend_matcher(backends),
        storage_dtype=torch.uint8 if scale is not None else None,
        attention_kv_dtype=attention_dtype,
        scale_recipe=scale,
    )


def _dq_workspace(
    phase: KVCacheAttentionPhase,
    backends,
    scale: str,
    attention_dtype: torch.dtype,
) -> KVCacheAttentionAccess:
    return KVCacheAttentionAccess(
        phase,
        _DQ_WORKSPACE_KIND,
        _backend_matcher(backends),
        storage_dtype=torch.uint8,
        attention_kv_dtype=attention_dtype,
        scale_recipe=scale,
        workspace_dtype=attention_dtype,
    )


def _native_fp4(
    phase: KVCacheAttentionPhase,
    backends,
    scale: str,
    attention_dtype: Optional[torch.dtype],
) -> KVCacheAttentionAccess:
    return KVCacheAttentionAccess(
        phase,
        _NATIVE_FP4_KIND,
        _backend_matcher(backends),
        storage_dtype=torch.uint8,
        attention_kv_dtype=attention_dtype,
        scale_recipe=scale,
    )


KV_CACHE_ATTENTION_ACCESS_REGISTRY: dict[str, tuple[KVCacheAttentionAccess, ...]] = {
    UnquantizedKVCacheMethod.name: (
        _plain(_PREFILL, _ANY_BACKEND),
        _plain(_DECODE, _ANY_BACKEND),
    ),
    NVFP4KVCacheMethod.name: (
        _dq_workspace(_PREFILL, _NVFP4_PREFILL_BACKENDS, _NVFP4_SCALE, _FP8_E4M3),
        _native_fp4(_DECODE, _NVFP4_DECODE_BACKENDS, _NVFP4_SCALE, _TORCH_FP4),
    ),
    FP4MXBlock16KVCacheMethod.name: (
        _plain(_PREFILL, _FP4_MX_PREFILL_BACKENDS, _FP4_MX_SCALE, _BF16),
        _plain(_DECODE, _FP4_MX_MHA_BACKENDS, _FP4_MX_SCALE, _BF16),
    ),
}


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
            "--kv-cache-dtype=fp4_e2m1 is deprecated. "
            "Use --kv-cache-dtype=fp4_mx_block16."
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
