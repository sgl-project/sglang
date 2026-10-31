"""CPU storage for the V4.1 SWA, shared compressed latents and index keys."""

import torch

from sglang.kernels.ops.attention.dsv4 import fused_norm_rope_inplace
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.quantization.fp4_fake_quant import (
    fake_quant_compressed_kv,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
    _num_dsv4_physical_kv_pages,
)


class CPUDeepSeekV4SingleKVPool(DeepSeekV4SingleKVPool):
    def get_bytes_per_token(self):
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) * 2

    def create_buffer(self, *, num_pages):
        self.kv_cache_total_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.bytes_per_page_padded = self.page_size * self.get_bytes_per_token()
        return torch.zeros(
            num_pages,
            self.page_size * self.kv_cache_total_dim,
            dtype=torch.bfloat16,
        )

    def get_key_buffer(self, layer_id):
        return self.kv_buffer[layer_id].view(-1, self.kv_cache_total_dim)

    def set_key_buffer_fused(self, layer_id, loc, cache_k, freqs_cis=None):
        if freqs_cis is not None:
            cache_k = fake_quant_compressed_kv(
                rope_tail(cache_k, freqs_cis, self.qk_rope_head_dim)
            )
        elif not self.kv_layout.is_fp4:
            # Match the V4 cache's per-64 UE8M0 NoPE rounding and BF16 RoPE.
            nope = cache_k[..., : self.qk_nope_head_dim].float().unflatten(-1, (-1, 64))
            scale = torch.exp2(
                torch.ceil(
                    torch.log2(nope.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
                )
            )
            nope = ((nope / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2)
            cache_k = torch.cat(
                [nope, cache_k[..., self.qk_nope_head_dim :]], dim=-1
            ).to(torch.bfloat16)
        valid = loc > 0
        self.get_key_buffer(layer_id)[loc[valid].long()] = cache_k[valid]


class CPUDeepSeekV4IndexerPool(DeepSeekV4IndexerPool):
    def get_bytes_per_token(self):
        return self.index_head_dim * 2

    def _create_buffer(self):
        num_pages = _num_dsv4_physical_kv_pages(
            self.size, self.page_size, self.global_page_size
        )
        self.k_buffer = [
            torch.zeros(
                num_pages * self.page_size,
                self.index_head_dim,
                dtype=torch.bfloat16,
            )
            for _ in range(self.layer_num)
        ]
        self.index_k_with_scale_buffer = [
            buffer.view(torch.uint8).view(
                -1, self.page_size * self.get_bytes_per_token()
            )
            for buffer in self.k_buffer
        ]

    def set_index_fp4(self, layer_id, loc, cache_k):
        valid = loc > 0
        self.k_buffer[layer_id][loc[valid].long()] = cache_k[valid]

    def get_index_k_dequant(self, layer_id, slots):
        return self.k_buffer[layer_id][slots.long()]


class DeepSeekV4CPUTokenToKVPool(DeepSeekV4TokenToKVPool):
    def compressed_kv_layout(self, compress_ratio):
        if compress_ratio not in (1, 2):
            raise ValueError("CPU V4.1 compressed KV requires ratio 1 or 2")
        return KVLayout.V41_FP4

    def _make_kv_pool(self, *, cls=None, **kwargs):
        return super()._make_kv_pool(cls=CPUDeepSeekV4SingleKVPool, **kwargs)

    def _make_indexer_pool(
        self,
        size,
        page_size,
        dtype,
        index_head_dim,
        layer_num,
        device,
        enable_memory_saver,
        force_fp4=False,
        global_page_size=None,
        index_k_page8=False,
    ):
        return CPUDeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            use_fp4_indexer=True,
            global_page_size=global_page_size,
        )

    def set_swa_key_buffer_radix_fused(self, layer_id, swa_loc, cache_k):
        self.swa_kv_pool.set_key_buffer_fused(
            self._swa_local_layer_id(layer_id), swa_loc, cache_k
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self, layer_id, swa_loc, kv, kv_weight, eps, freqs_cis, positions, q=None
    ):
        if q is not None:
            raise ValueError("CPU SWA store does not fuse query RoPE")
        # The V4 SWA writer quantizes normalized FP32 registers, not BF16 rows.
        kv = kv.float().contiguous()
        fused_norm_rope_inplace(kv, kv_weight, eps, freqs_cis, positions)
        self.set_swa_key_buffer_radix_fused(layer_id, swa_loc, kv)
