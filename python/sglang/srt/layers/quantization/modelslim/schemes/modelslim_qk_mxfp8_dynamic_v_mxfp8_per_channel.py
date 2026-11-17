# SPDX-License-Identifier: Apache-2.0
"""ModelSlim dynamic grouped Q/K with static per-channel MXFP8 V."""

import logging

import torch

from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

from .modelslim_kv_cache import ModelSlimKVCacheLayout, ModelSlimKVCacheScheme

logger = logging.getLogger(__name__)


def init_mxfp8_kv_scales(model, pool, cache_layout):
    """Populate static V scales after weight postprocessing and pool allocation."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    # Draft steps share the same pool; initialize its static scales only once.
    full_pool = pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool
    if full_pool._mxfp8_scales_initialized:
        return
    for module in model.modules():
        scheme = getattr(module, "scheme", None)
        if getattr(scheme, "cache_layout", None) == cache_layout:
            # Keep the hybrid pool's global-to-local layer mapping.
            _, v_scale = pool.get_kv_scale_buffer(module.layer_id)
            v_scale.copy_(
                scheme.v_scale_encoded.view(
                    1, 1, v_scale.shape[2], v_scale.shape[-2], 1
                )
            )
    full_pool._mxfp8_scales_initialized = True


class _MXFP8KVCacheLayout(ModelSlimKVCacheLayout):
    group_size = 32

    @property
    def bytes_per_token(self):
        # K has one scale per channel group; static V scales are repeated in
        # the cache's paged per-channel layout for QFA.
        return super().bytes_per_token + (
            self.local_kv_heads
            * (self.key_head_dim + self.value_head_dim)
            // self.group_size
            * torch._utils._element_size(self.scale_dtype)
        )


class ModelSlimQKMXFP8DynamicVMXFP8PerChannelScheme(ModelSlimKVCacheScheme):
    SUPPORTED_KV_CACHE_DTYPES = ("fp8_e4m3",)

    def create_weights(self, layer, *, qkv_proj, params_dtype):
        if not is_npu_arch35():
            raise ValueError("ModelSlim MXFP8 KV cache requires NPU arch35 (A5).")
        self.compute_dtype = params_dtype
        self.cache_layout = _MXFP8KVCacheLayout(
            layout="gqa_mxfp8",
            key_head_dim=qkv_proj.head_size,
            value_head_dim=qkv_proj.v_head_size,
            local_kv_heads=qkv_proj.num_kv_heads,
            key_dtype=torch.float8_e4m3fn,
            value_dtype=torch.float8_e4m3fn,
            scale_dtype=torch.uint8,
        )
        self._create_head_parameter(layer, "fa_v_scale", qkv_proj, 127)
        self._create_head_parameter(layer, "fa_v_offset", qkv_proj, 0)

    def _create_head_parameter(self, layer, name, qkv_proj, default):
        channels = qkv_proj.v_head_size
        local_heads = qkv_proj.num_kv_heads
        total_heads = qkv_proj.total_num_kv_heads
        kv_tp_rank, _ = get_group_rank_size(qkv_proj.kv_tp_group)
        first_head = kv_tp_rank // qkv_proj.num_kv_head_replicas * local_heads
        parameter = self._create_parameter(
            layer, name, (local_heads, channels), torch.uint8, default
        )
        load_parameter = parameter.weight_loader

        def weight_loader(param, loaded_weight):
            # Match the QKV projection's KV-head sharding/replication. A local
            # checkpoint shard already contains exactly this rank's heads.
            if loaded_weight.numel() != param.numel():
                loaded_weight = loaded_weight.reshape(total_heads, channels).narrow(
                    0, first_head, local_heads
                )
            load_parameter(param, loaded_weight.reshape_as(param))

        parameter.weight_loader = weight_loader

    def process_weights_after_loading(self, layer):
        scale, offset = layer.fa_v_scale, layer.fa_v_offset
        if not scale.modelslim_loaded:
            logger.warning(
                "%s.fa_v.scale is missing; using scale=1 (E8M0 byte 127)",
                self.prefix,
            )
            scale.data.fill_(127)
        if (scale == 255).any():
            raise ValueError(f"{self.prefix}.fa_v.scale contains E8M0 NaN")
        if offset.modelslim_loaded and torch.count_nonzero(offset):
            raise ValueError(f"{self.prefix}.fa_v.offset must be zero for MXFP8")
        # ModelSlim minmax uses zero for channels with zero calibration absmax.
        # Match the Ascend consumer in both the cache writer and attention.
        self.v_scale_encoded = torch.where(scale == 0, 127, scale).to(torch.uint8)
        self.v_inv_scale = torch.exp2(self.v_scale_encoded.float() - 127).reciprocal()
        self.p_scale = torch.ones(1, dtype=torch.float32, device=scale.device)

    def get_cache_layout(self, *, use_mla_backend):
        if use_mla_backend:
            raise ValueError("ModelSlim MXFP8 KV cache requires independent K/V.")
        return self.cache_layout

    def quantize_query(self, q):
        import torch_npu

        if q.numel() == 0:
            return (
                q.new_empty(q.shape, dtype=torch.float8_e4m3fn),
                q.new_empty((*q.shape[:-1], q.shape[-1] // 32), dtype=torch.uint8),
            )
        quantized, scale = torch_npu.npu_dynamic_mx_quant(
            q, dst_type=torch.float8_e4m3fn
        )
        return quantized, scale.view(torch.uint8).reshape(
            *q.shape[:-1], q.shape[-1] // 32
        )

    def prepare_query(self, query, scale_layout):
        query, q_scale = self.quantize_query(query)
        q_heads = query.shape[-2]
        kv_heads = self.cache_layout.local_kv_heads
        groups = q_heads // kv_heads
        scale_pairs = q_scale.shape[-1] // 2
        q_scale = q_scale.reshape(query.shape[0], q_heads, scale_pairs, 2)
        # Use the layout selected once for the shared QFA metadata plan.
        if scale_layout == "N2TGD":
            q_scale = (
                q_scale.view(query.shape[0], kv_heads, groups, scale_pairs, 2)
                .permute(1, 0, 2, 3, 4)
                .contiguous()
            )
        return query, q_scale

    def quantize_cache(self, k, v):
        import torch_npu

        k, k_scale = self.quantize_query(k)
        # Flatten heads/channels to the single static quantization axis.
        v = torch_npu.npu_quantize(
            v.float().flatten(-2),
            self.v_inv_scale.reshape(-1),
            None,
            torch.float8_e4m3fn,
            axis=-1,
            div_mode=False,
        ).view(v.shape)
        return k, v, k_scale
