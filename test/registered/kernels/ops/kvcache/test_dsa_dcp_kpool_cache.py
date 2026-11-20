"""DSA DCP keeps pooled index keys replicated and shards only target MLA KV."""

import unittest

import torch

from sglang.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_dcp_sharded_triton,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and not torch.version.hip, "requires CUDA"
)
class TestDsaDcpKpoolCache(CustomTestCase):
    def test_no_rope_can_be_omitted(self):
        for dim in (512, 576):
            with self.subTest(dim=dim):
                slots = torch.arange(16, device="cuda")
                latent = torch.randn(16, 1, dim, device="cuda", dtype=torch.bfloat16)
                for rank in range(2):
                    actual = torch.zeros(8, 1, dim, device="cuda", dtype=torch.bfloat16)
                    with get_parallel().override(attn_dcp_size=2, attn_dcp_rank=rank):
                        set_mla_kv_buffer_dcp_sharded_triton(
                            actual, slots, latent, None
                        )
                    expected = torch.zeros_like(actual)
                    owned = (slots % 2 == rank) & (slots != 0)
                    expected[slots[owned] // 2] = latent[owned]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_virtual_slot_capacity_and_target_draft_writes(self):
        """Exercise last-page keys and non-owner KV writes in both pool modes."""
        torch.manual_seed(43735)
        for kpool in (2, 4, 8):
            page = 64 * kpool
            size = 4 * page
            for dcp in (2, 4, 8):
                virtual_capacity = (size + page) * dcp
                # Include page boundaries and the very last virtual slot.
                slots = torch.arange(page * dcp, virtual_capacity, device="cuda")
                slots = slots[torch.randperm(len(slots), device="cuda")[:257]]
                slots = torch.unique(
                    torch.cat(
                        (
                            slots,
                            slots.new_tensor(
                                [0, 1, 2, 3, page * dcp, virtual_capacity - 1]
                            ),
                        )
                    )
                )
                latent = torch.randn(
                    len(slots), 1, 512, device="cuda", dtype=torch.bfloat16
                )
                index_slots = torch.arange(virtual_capacity // kpool, device="cuda")
                keys = torch.randn(len(index_slots), 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                scales = (index_slots.float() + 1).unsqueeze(1)
                for replicated in (False, True):
                    for rank in range(dcp):
                        with self.subTest(
                            kpool=kpool, dcp=dcp, draft=replicated, rank=rank
                        ):
                            with get_parallel().override(
                                attn_dcp_size=dcp, attn_dcp_rank=rank
                            ):
                                pool = DSATokenToKVPool(
                                    size=size * (dcp if replicated else 1),
                                    page_size=page * (dcp if replicated else 1),
                                    kv_lora_rank=512,
                                    dtype=torch.bfloat16,
                                    qk_rope_head_dim=0,
                                    layer_num=1,
                                    device="cuda",
                                    index_head_dim=128,
                                    enable_memory_saver=False,
                                    kv_cache_dim=512,
                                    index_kpool=kpool,
                                    index_kpool_compress=True,
                                    tail_extra_slots=6,
                                    max_running_requests=2,
                                    dcp_replicated=replicated,
                                )
                                pool.kv_buffer[0].zero_()
                                pool.set_mla_kv_buffer(
                                    None,
                                    slots,
                                    latent,
                                    latent[:, :, :0],
                                    layer_id_override=0,
                                )
                                expected = torch.zeros_like(pool.kv_buffer[0])
                                if replicated:
                                    valid = slots != 0
                                    expected[slots[valid]] = latent[valid]
                                else:
                                    owned = (slots % dcp == rank) & (slots != 0)
                                    expected[slots[owned] // dcp] = latent[owned]
                                torch.testing.assert_close(
                                    pool.kv_buffer[0], expected, atol=0, rtol=0
                                )

                                pool.set_index_k_scale_buffer(
                                    0, index_slots, keys, scales
                                )
                                n = len(index_slots)
                                actual_keys, actual_scales = (
                                    pool.get_index_k_scale_buffer(
                                        0,
                                        torch.tensor(
                                            [n], device="cuda", dtype=torch.int32
                                        ),
                                        torch.arange(
                                            n // 64, device="cuda", dtype=torch.int32
                                        )[None],
                                        n,
                                        n,
                                    )
                                )
                                torch.testing.assert_close(
                                    actual_keys, keys.view(torch.uint8), atol=0, rtol=0
                                )
                                torch.testing.assert_close(
                                    actual_scales.view(torch.float32),
                                    scales,
                                    atol=0,
                                    rtol=0,
                                )
                                del pool


if __name__ == "__main__":
    unittest.main()
