"""Test fused AITER DSA decode with the FP8 contract supplied by PR #39083."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


class TestAiterDSAFusedPrep(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.version.hip or not torch.cuda.is_available():
            raise AssertionError("This regression requires ROCm and an AMD GPU")
        arch = torch.cuda.get_device_properties(0).gcnArchName
        if not arch.startswith("gfx95"):
            raise AssertionError("This regression requires gfx95 hardware")
        if not forward_mla_rocm._use_aiter_gfx95:
            raise AssertionError("This regression requires SGLANG_USE_AITER=1")

    def test_fused_fp8_query_runs_fp8_kv_aiter_decode(self):
        torch.manual_seed(1)
        device = torch.device("cuda")
        batch_size = 4
        num_heads = 16
        head_dim = 576
        value_dim = 512
        topk = 2048

        kv_cache = torch.randn(
            batch_size * topk,
            1,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ).to(fp8_dtype)
        pool = SimpleNamespace(
            get_key_buffer=lambda _layer_id: kv_cache,
        )
        attn = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            current_attention_backend="dsa",
            attn_mqa=SimpleNamespace(
                layer_id=0,
                k_scale=torch.ones((), dtype=torch.float32, device=device),
            ),
            rotary_emb=SimpleNamespace(
                cos_cache=torch.ones(8, 64, dtype=torch.bfloat16, device=device),
                sin_cache=torch.zeros(8, 64, dtype=torch.bfloat16, device=device),
                is_neox_style=False,
            ),
        )

        q_nope = torch.randn(
            batch_size,
            num_heads,
            value_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        q_rope = torch.randn(
            batch_size,
            num_heads,
            head_dim - value_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        k_nope = torch.randn(
            batch_size, 1, value_dim, dtype=torch.bfloat16, device=device
        )
        k_rope = torch.randn(
            batch_size,
            1,
            head_dim - value_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        positions = torch.zeros(batch_size, dtype=torch.int64, device=device)
        cache_locs = torch.arange(batch_size, dtype=torch.int64, device=device) * topk

        with (
            patch.object(
                forward_mla_rocm,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(dsa_decode_backend="aiter")
                ),
            ),
            patch.object(
                forward_mla_rocm,
                "get_token_to_kv_pool",
                return_value=pool,
            ),
        ):
            q_cat, _, _, _ = forward_mla_rocm._fused_rope_cat_and_cache(
                attn,
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                positions,
                cache_locs,
            )

        self.assertEqual(q_cat.dtype, fp8_dtype)
        self.assertTrue(torch.isfinite(q_cat.float()).all().item())

        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.device = device
        backend.need_pad_heads = False
        backend.num_head_padded = num_heads
        backend.dsa_index_topk = topk
        backend.aiter_dsa_max_split_per_batch = 64
        backend.kv_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        backend.kv_indices = torch.empty(
            batch_size * topk, dtype=torch.int32, device=device
        )
        backend.aiter_dsa_metadata_capacity = 0
        backend.aiter_dsa_metadata_max_seqlen_q = 0
        backend.aiter_dsa_metadata_q_dtype = None
        backend.aiter_dsa_metadata_kv_dtype = None
        backend.aiter_dsa_kv_last_page_lens = None
        backend.aiter_dsa_work_metadata = None
        backend.aiter_dsa_identity_scale = torch.ones(
            (), dtype=torch.float32, device=device
        )
        backend.aiter_dsa_decode_metadata_owner = None
        backend.aiter_dsa_decode_kv_last_page_lens = None
        backend.aiter_dsa_decode_persistent_kwargs = None

        metadata = SimpleNamespace(
            cu_seqlens_q=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
            max_seq_len_q=1,
        )
        layer = SimpleNamespace(
            tp_q_head_num=num_heads,
            head_dim=head_dim,
            v_head_dim=value_dim,
            k_scale=torch.ones((), dtype=torch.float32, device=device),
            scaling=1.0 / math.sqrt(head_dim),
            logit_cap=0.0,
        )
        page_table = torch.arange(
            batch_size * topk, dtype=torch.int32, device=device
        ).view(
            batch_size,
            topk,
        )

        output = backend._forward_aiter(
            q_all=q_cat,
            kv_cache=kv_cache,
            page_table_1=page_table,
            layer=layer,
            metadata=metadata,
            bs=batch_size,
        )
        torch.cuda.synchronize()

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
