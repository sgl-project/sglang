# SPDX-License-Identifier: Apache-2.0
"""ModelSlim MLA FP8 with static shared latent and dynamic per-token/head Q."""

import logging

import torch

from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

from .modelslim_kv_cache import ModelSlimKVCacheLayout, ModelSlimKVCacheScheme

logger = logging.getLogger(__name__)


class ModelSlimQFP8DynamicKVFP8Scheme(ModelSlimKVCacheScheme):
    SUPPORTED_KV_CACHE_DTYPES = ("fp8_e4m3",)

    def create_weights(self, layer, *, kv_lora_rank, qk_rope_head_dim, params_dtype):
        if not is_npu_arch35():
            raise ValueError("ModelSlim MLA FP8 KV cache requires NPU arch35 (A5).")
        self.compute_dtype = params_dtype
        self.cache_layout = ModelSlimKVCacheLayout(
            layout="mla_fp8",
            key_head_dim=kv_lora_rank,
            value_head_dim=qk_rope_head_dim,
            local_kv_heads=1,
            key_dtype=torch.float8_e4m3fn,
            value_dtype=torch.bfloat16,
        )
        # MLA's exported fa_v is compatibility data: QK and PV share fa_k's C.
        self._create_parameter(layer, "fa_k_scale", (1, 1), torch.float32, 1.0)
        self._create_parameter(layer, "fa_k_offset", (1, 1), torch.float32, 0.0)

    def process_weights_after_loading(self, layer):
        scale, offset = layer.fa_k_scale, layer.fa_k_offset
        if not scale.modelslim_loaded:
            logger.warning(
                "%s.fa_k.scale is missing; using scale=1",
                self.prefix,
            )
            scale.data.fill_(1)
        if scale.modelslim_loaded and (
            not torch.isfinite(scale).all() or not (scale > 0).all()
        ):
            raise ValueError(f"{self.prefix}.fa_k.scale must be finite and positive")
        if offset.modelslim_loaded and torch.count_nonzero(offset):
            raise ValueError(f"{self.prefix}.fa_k.offset must be zero for FP8")

        self.k_scale = scale.detach()
        self.k_inv_scale = self.k_scale.reciprocal()

    def get_cache_layout(self, *, use_mla_backend):
        if not use_mla_backend:
            raise ValueError(
                "ModelSlim Q_FP8_DYNAMIC_KV_FP8 currently supports MLA only."
            )
        return self.cache_layout

    def quantize_query(self, q):
        import torch_npu

        shape = q.shape
        quantized, scale = torch_npu.npu_dynamic_quant(
            q.reshape(-1, shape[-1]), dst_type=torch.float8_e4m3fn
        )
        return quantized.view(shape), scale.view(*shape[:-1], 1).float()

    @staticmethod
    def _quantize(x, inverse_scale):
        import torch_npu

        # Out of place: MLA prefill still uses the original BF16 latent.
        if x.dtype == torch.float8_e4m3fn:
            return x
        # AscendQuant requires matching input/scale dtypes. Keep the checkpoint
        # scale in FP32; axis -2 applies one scale to each KV head.
        return torch_npu.npu_quantize(
            x.float(),
            inverse_scale.reshape(-1),
            None,
            torch.float8_e4m3fn,
            axis=-2,
            div_mode=False,
        )

    def quantize_cache(self, k, v):
        k = self._quantize(k, self.k_inv_scale)
        return k, v.to(torch.bfloat16), None

    def dequantize_cache(self, k, v=None):
        import torch_npu

        # Explicit singleton axes avoid ambiguity when head count equals D.
        k = torch_npu.npu_anti_quant(
            k,
            self.k_scale.view(*([1] * (k.ndim - 2)), -1, 1),
            dst_dtype=self.compute_dtype,
        )
        return k, v
