# SPDX-License-Identifier: Apache-2.0
"""Real SGLang pool -> prefill / fused decode integration on Hopper."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="The DSA NVFP4 backend requires SM90",
)


def _reference(q, kv, lengths, scale):
    scores = torch.einsum("qhd,kd->qhk", q.float(), kv[:, 0].float()) * scale
    mask = torch.arange(kv.shape[0], device=q.device)[None, :] < lengths[:, None]
    scores.masked_fill_(~mask[:, None, :], float("-inf"))
    return torch.einsum("qhk,kd->qhd", scores.softmax(-1), kv[:, 0, :512].float())


@pytest.mark.parametrize("has_prefix", [False, True])
@torch.inference_mode()
def test_pool_prefill_and_fused_decode(has_prefix):
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")
    try:
        pool = DSATokenToKVPool(
            size=256,
            page_size=64,
            dtype=torch.float4_e2m1fn_x2,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=416,
            index_head_dim=128,
        )
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.token_to_kv_pool = pool
        backend.dsa_kv_cache_store_nvfp4 = True
        backend.dsa_kv_cache_store_fp8 = False
        backend.dsa_index_kpool = 1
        backend.dsa_index_topk = 2048
        backend.dsa_prefill_impl = "flashmla_sparse"
        backend.dsa_decode_impl = "flashmla_kv"
        backend.use_mha = False
        backend.use_fused_topk = False
        backend.hisparse_coordinator = None
        backend.device_sm_major = 9
        backend.flashmla_kv_num_q_heads = 64
        backend.real_page_size = 64
        backend.kv_cache_dim = 416
        backend._sink_pad_cache = {}
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            tp_q_head_num=8,
            head_dim=576,
            v_head_dim=512,
            scaling=576**-0.5,
        )
        prefix = 4 if has_prefix else 0
        n = 8
        gen = torch.Generator(device="cuda").manual_seed(20261015)
        loc = torch.randperm(192, generator=gen, device="cuda")[: prefix + n] + 64
        kv = (
            torch.randn(prefix + n, 1, 576, generator=gen, device="cuda") * 0.3
        ).bfloat16()
        q = (torch.randn(n, 8, 576, generator=gen, device="cuda") * 0.3).bfloat16()
        lengths = torch.arange(
            prefix + 1, prefix + n + 1, device="cuda", dtype=torch.int32
        )
        indices = (
            torch.arange(2048, device="cuda", dtype=torch.int32).expand(n, -1).clone()
        )
        indices.masked_fill_(indices >= lengths[:, None], -1)
        if prefix:
            pool.set_mla_kv_buffer(
                layer, loc[:prefix], kv[:prefix, :, :512], kv[:prefix, :, 512:]
            )
        backend.forward_metadata = SimpleNamespace(
            topk_indices_offset=torch.zeros(n, device="cuda", dtype=torch.int32),
            page_table_1_flattened=loc,
            dsa_cache_seqlens_int32=lengths,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=loc[prefix:],
            extend_prefix_lens_cpu=[prefix],
        )
        output = backend.forward_extend(
            q,
            kv[prefix:, :, :512],
            kv[prefix:, :, :512],
            layer,
            batch,
            k_rope=kv[prefix:, :, 512:],
            topk_indices=indices,
        )
        latent, rope = pool.get_mla_kv_buffer(layer, loc)
        restored = torch.cat([latent, rope], dim=-1)
        expected = _reference(q, restored if prefix else kv, lengths, layer.scaling)
        torch.testing.assert_close(output.float(), expected, atol=0.003, rtol=0.025)

        decode_q = q[-2:].contiguous()
        decode_lengths = torch.full((2,), prefix + n, device="cuda", dtype=torch.int32)
        decode_indices = torch.full((2, 2048), -1, device="cuda", dtype=torch.int32)
        decode_indices[:, : prefix + n] = loc.to(torch.int32)
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=decode_lengths,
            flashmla_metadata=backend._compute_flashmla_metadata(
                decode_lengths, seq_len_q=1
            ),
        )
        decoded = backend._forward_flashmla_kv(
            q_all=decode_q,
            kv_cache=pool.get_key_buffer(0),
            v_head_dim=512,
            sm_scale=layer.scaling,
            layer=layer,
            metadata=metadata,
            page_table_1=decode_indices,
        )
        expected_decode = _reference(decode_q, restored, decode_lengths, layer.scaling)
        torch.testing.assert_close(
            decoded[:, 0].float(), expected_decode, atol=0.003, rtol=0.025
        )
        assert pool.get_key_buffer(0).dtype == torch.uint8
        assert pool.get_key_buffer(0).shape[-1] == 416
    finally:
        reset_context()
