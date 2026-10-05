import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_dcp import (
    dsa_dcp_head_groups,
    dsa_dcp_replicated_q_weight_bytes,
    dsa_dcp_runtime_reservation_bytes,
)
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextmanager
def _configuration(dcp_size, rank=0, **fields):
    with (
        get_context().override_server_args(
            **{
                "page_size": 64,
                "tp_size": 4,
                "dsa_prefill_backend": "trtllm",
                "dsa_decode_backend": "trtllm",
                "enable_dsa_cache_layer_split": False,
                **fields,
            },
        ),
        get_parallel().override(
            attn_dcp_size=dcp_size,
            attn_dcp_rank=rank,
            attn_tp_size=fields.get("tp_size", 4) // fields.get("attn_dp_size", 1),
            attn_dp_size=fields.get("attn_dp_size", 1),
            dcp_enabled=dcp_size > 1,
        ),
    ):
        yield


def _configurator(dtype=torch.bfloat16, *, share_topk=False):
    kvc = object.__new__(KVCacheConfigurator)
    kvc.device = "cuda"
    kvc.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            model_type="glm_moe_dsa",
            index_topk=2048,
            index_head_dim=128,
            q_lora_rank=2048,
            indexer_types=["full", "shared" if share_topk else "full"],
        ),
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        num_hidden_layers=2,
        num_attention_heads=64,
        context_len=8192,
        is_draft_model=False,
        linear_attn_registry_result=None,
    )
    kvc.layer_info = SimpleNamespace(start_layer=0, end_layer=2, num_effective_layers=2)
    kvc.model_config.hf_config.get_text_config = lambda: kvc.model_config.hf_config
    kvc.is_draft_worker = False
    kvc.server_args = get_server_args()
    kvc.kv_cache_dtype = dtype
    kvc.kv_cache_dtype_str = str(dtype)
    kvc.use_mla_backend = True
    kvc.page_size = 64
    kvc.pp_size = 1
    kvc.attn_dp_size = get_parallel().attn_dp_size
    kvc.mambaish_config = None
    kvc.is_hybrid_swa = False
    kvc.spec_algorithm = SpeculativeAlgorithm.NONE
    return kvc


def _build_cpu_pool(kvc, size):
    def allocate_cpu(*args, **kwargs):
        kwargs["device"] = "cpu"
        return DSATokenToKVPool(*args, **kwargs)

    with patch(
        "sglang.srt.mem_cache.kv_cache_configurator.DSATokenToKVPool",
        side_effect=allocate_cpu,
    ):
        return kvc._build_dsa_kv_pool(max_total_num_tokens=size, max_running_requests=8)


def _allocate_all(pool, dcp_size):
    allocator = PagedTokenToKVPoolAllocator(
        256 * dcp_size,
        page_size=64 * dcp_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )
    return allocator.alloc(allocator.available_size())


class TestDSADCPPool(CustomTestCase):
    def test_index_keys_cover_all_allocator_slots(self):
        for dcp_size in (1, 2, 4):
            with self.subTest(dcp_size=dcp_size), _configuration(dcp_size):
                pool = _build_cpu_pool(_configurator(), 256)
                slots = _allocate_all(pool, dcp_size)
                self.assertEqual(pool.index_buf_size, 256 * dcp_size)
                self.assertEqual(pool.kv_buffer[0].shape[0], 256 + 64)
                index = pool.get_index_k_with_scale_buffer(0)
                self.assertGreater(index.shape[0] * 64, int(slots.max()))

    def test_draft_pool_is_replicated_over_allocator_slots(self):
        for dcp_size in (2, 4):
            with self.subTest(dcp_size=dcp_size), _configuration(dcp_size):
                kvc = _configurator()
                kvc.is_draft_worker = True
                pool = _build_cpu_pool(kvc, 256 * dcp_size)
                slots = _allocate_all(pool, dcp_size)
                self.assertEqual(pool.page_size, 64)
                self.assertEqual(pool._write_loc_dcp_span, 1)
                self.assertGreater(pool.kv_buffer[0].shape[0], int(slots.max()))
                self.assertEqual(pool.index_buf_size, 256 * dcp_size)

    def test_pool_fits_the_sized_budget(self):
        budget = 4 << 20
        for dcp_size in (2, 4):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(dcp=dcp_size, dtype=dtype), _configuration(dcp_size):
                    target = _configurator(dtype, share_topk=True)
                    target.spec_algorithm = SpeculativeAlgorithm.EAGLE
                    target.spec_aux_config = SimpleNamespace(eagle_draft_num_layers=1)
                    draft = _configurator(dtype)
                    draft.is_draft_worker = True
                    draft.layer_info = SimpleNamespace(
                        start_layer=0, end_layer=1, num_effective_layers=1
                    )
                    size = (
                        DefaultPoolConfigurator(target)
                        .calculate_pool_sizes(budget, 64)
                        .max_total_num_tokens
                    )

                    def allocated(size):
                        return (
                            _build_cpu_pool(target, size).get_kv_size_bytes()
                            + _build_cpu_pool(
                                draft, size * dcp_size
                            ).get_kv_size_bytes()
                        )

                    self.assertLessEqual(allocated(size), budget)
                    self.assertGreater(allocated(size + 64), budget)

    def test_retraction_restores_index_keys_and_local_kv(self):
        dcp_size = 2
        for rank in range(dcp_size):
            with (
                self.subTest(rank=rank),
                _configuration(dcp_size, rank),
                patch("torch.cuda.synchronize"),
            ):
                pool = _build_cpu_pool(_configurator(share_topk=True), 256)
                page = 64 * dcp_size
                old = torch.arange(page, 2 * page)
                new = torch.arange(4 * page, 5 * page)
                old_rows = old[rank::dcp_size] // dcp_size
                new_rows = new[rank::dcp_size] // dcp_size
                kv = pool.kv_buffer[0]
                kv[old_rows] = torch.arange(64, dtype=kv.dtype).view(64, 1, 1)
                index = pool.index_k_with_scale_buffer[0]
                index[dcp_size : 2 * dcp_size].fill_(3)
                expected_kv = kv[old_rows].clone()

                backup = pool.get_cpu_copy(old)
                kv.zero_()
                index.zero_()
                pool.load_cpu_copy(backup, new)

                torch.testing.assert_close(kv[new_rows], expected_kv)
                self.assertTrue((index[4 * dcp_size : 5 * dcp_size] == 3).all())

    def test_runtime_reservation(self):
        reservations = []
        for dcp_size in (1, 2, 4):
            with _configuration(dcp_size, chunked_prefill_size=8192):
                reservations.append(dsa_dcp_runtime_reservation_bytes(_configurator()))
        self.assertEqual(reservations[0], 0)
        self.assertGreater(reservations[1], envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get())
        self.assertGreater(reservations[2], reservations[1])

        with _configuration(2, chunked_prefill_size=8192):
            kvc = _configurator()
            with get_parallel().override(dcp_replicate_q_proj=True):
                # 32 gathered heads of BF16 q_b_proj and w_kc in each layer.
                q_weight_bytes = 2 * 32 * (256 * 2048 + 192 * 512) * 2
                self.assertEqual(
                    dsa_dcp_replicated_q_weight_bytes(kvc.model_config, 32),
                    q_weight_bytes,
                )
                self.assertEqual(
                    dsa_dcp_runtime_reservation_bytes(kvc),
                    reservations[1] + q_weight_bytes,
                )

    def test_head_groups(self):
        for heads, expected in ((16, 1), (32, 1), (64, 2), (96, 3), (128, 4)):
            self.assertEqual(dsa_dcp_head_groups(heads), expected)


if __name__ == "__main__":
    unittest.main()
