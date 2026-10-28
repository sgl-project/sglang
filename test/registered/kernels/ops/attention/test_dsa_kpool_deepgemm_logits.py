"""SM100 DeepGEMM MQA logits of the kpool indexer: MXFP8 Q/K read from the FP8 index cache."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import IndexKScaleFmt
from sglang.srt.environ import envs
from sglang.srt.layers.attention.deep_gemm_mqa_logits import (
    get_paged_mqa_logits_metadata,
    pack_ue8m0_x4,
)
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
    kpool_softmax_rotate_write_cache,
)
from sglang.srt.layers.attention.dsa.kpool_plan import _compute_pool_schedule_metadata
from sglang.srt.layers.attention.dsa.utils import resolve_paged_mqa_logits_metadata_fn
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.paged_mqa import (
    assert_paged_mqa_matches_ref,
    generate_paged_mqa_test_data,
    ref_fp8_paged_mqa_logits,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

PAGE_SIZE = 64
NUM_HEADS = 64
PAGE_NBYTES = PAGE_SIZE * (INDEX_HEAD_DIM + 4)


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


def _cast_to_fp8_pow2(x: torch.Tensor):
    amax = x.abs().float().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def _indexer(ue8m0_index_cache: bool) -> IndexerKPool:
    indexer = IndexerKPool.__new__(IndexerKPool)
    torch.nn.Module.__init__(indexer)
    indexer.mx_mqa_logits = True
    indexer.ue8m0_index_cache = ue8m0_index_cache
    return indexer


def _ref_logits(q_fp8, k_fp8, k_scale, weights):
    """``[rows, keys]`` with the kernel's BF16 weights; Q scales are in the weights."""
    k = k_fp8.float() * k_scale.view(-1, 1)
    scores = torch.einsum("mhd,nd->mhn", q_fp8.float(), k).relu()
    return torch.einsum("mhn,mh->mn", scores, weights.bfloat16().float())


@unittest.skipUnless(_is_sm100(), "DeepGEMM's MXFP8 MQA logits need SM100")
class TestDsaKpoolDeepGemmLogits(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _q_and_weights(self, num_rows: int):
        q = torch.randn(num_rows, NUM_HEADS, INDEX_HEAD_DIM, device="cuda")
        q_fp8, q_scale = _cast_to_fp8_pow2(q)
        gate = torch.randn(num_rows, NUM_HEADS, device="cuda")
        return q_fp8, gate * q_scale.squeeze(-1) * INDEX_HEAD_DIM**-0.5

    def test_cache_writer_packs_scales_losslessly(self):
        pool = SimpleNamespace(index_page_size=PAGE_SIZE, index_head_dim=INDEX_HEAD_DIM)
        num_slots, kpool = 100, 4
        slot_k = torch.randn(
            num_slots, kpool, INDEX_HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        slot_score = torch.randn_like(slot_k, dtype=torch.float32)
        ape = torch.randn(kpool, INDEX_HEAD_DIM, device="cuda")
        loc = torch.randperm(2 * PAGE_SIZE, device="cuda")[:num_slots]

        caches, compressed = [], []
        for k_scale_fmt in (IndexKScaleFmt.POW2, IndexKScaleFmt.UE8M0):
            kwargs = dict(
                pool=pool,
                slot_k=slot_k,
                slot_score=slot_score,
                ape=ape,
                loc=loc,
                k_scale_fmt=k_scale_fmt,
            )
            cache = torch.zeros(2, PAGE_NBYTES, dtype=torch.uint8, device="cuda")
            kpool_softmax_rotate_write_cache(buf=cache, **kwargs)
            caches.append(cache)
            compressed.append(
                kpool_softmax_rotate_write_cache(
                    buf=cache, return_compressed=True, write_cache=False, **kwargs
                )
            )

        data_bytes = PAGE_SIZE * INDEX_HEAD_DIM
        fp32_cache, ue8m0_cache = caches
        torch.testing.assert_close(
            ue8m0_cache[:, :data_bytes], fp32_cache[:, :data_bytes], atol=0, rtol=0
        )
        fp32_scale = fp32_cache[:, data_bytes:].contiguous().view(torch.float32)
        ue8m0_scale = ue8m0_cache[:, data_bytes:].contiguous().view(torch.int32)
        torch.testing.assert_close(
            ue8m0_scale, pack_ue8m0_x4(fp32_scale), atol=0, rtol=0
        )

        (k_fp32, s_fp32), (k_ue8m0, s_ue8m0) = compressed
        torch.testing.assert_close(
            k_ue8m0.view(torch.uint8), k_fp32.view(torch.uint8), atol=0, rtol=0
        )
        torch.testing.assert_close(
            s_ue8m0.view(torch.int32), pack_ue8m0_x4(s_fp32), atol=0, rtol=0
        )

    def test_ragged_logits_are_compressed(self):
        num_rows, num_keys = 48, 1000
        q_fp8, weights = self._q_and_weights(num_rows)
        k_fp8, k_scale = _cast_to_fp8_pow2(
            torch.randn(num_keys, INDEX_HEAD_DIM, device="cuda")
        )
        k_scale = k_scale.squeeze(-1)
        starts = torch.randint(0, num_keys // 2, (num_rows,), device="cuda").int()
        ends = (
            starts + torch.randint(1, num_keys // 2, (num_rows,), device="cuda")
        ).int()
        max_seqlen_k = int((ends - starts).max())
        ref = _ref_logits(q_fp8, k_fp8, k_scale, weights)

        for ue8m0_index_cache in (False, True):
            cache_scale = (
                pack_ue8m0_x4(k_scale).view(torch.float32)
                if ue8m0_index_cache
                else k_scale
            )
            logits = _indexer(ue8m0_index_cache)._fp8_mqa_logits(
                q_fp8,
                k_fp8,
                cache_scale,
                weights,
                starts,
                ends,
                max_seqlen_k=max_seqlen_k,
            )
            self.assertEqual(logits.dtype, torch.float32)
            self.assertEqual(logits.shape, (num_rows, max_seqlen_k))
            got = torch.cat([logits[i, : ends[i] - starts[i]] for i in range(num_rows)])
            want = torch.cat([ref[i, starts[i] : ends[i]] for i in range(num_rows)])
            self.assertLess(_calc_diff(got, want), 3e-5, f"{ue8m0_index_cache=}")

    def test_pool_schedule_matches_deepgemm_under_torch_mqa_logits(self):
        import deep_gemm

        pool_lens = torch.full((3,), 2048, dtype=torch.int32, device="cuda")
        want = get_paged_mqa_logits_metadata(
            pool_lens.view(-1, 1), PAGE_SIZE, deep_gemm.get_num_sms()
        )
        resolve_paged_mqa_logits_metadata_fn.cache_clear()
        try:
            with envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True):
                got = _compute_pool_schedule_metadata(
                    pool_lens, index_page_size=PAGE_SIZE
                )
        finally:
            resolve_paged_mqa_logits_metadata_fn.cache_clear()
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_paged_logits_read_the_ue8m0_cache(self):
        import deep_gemm

        num_rows, max_seq_len = 16, 2048
        data = generate_paged_mqa_test_data(num_rows, 1, NUM_HEADS, 1024, max_seq_len)
        cache = data["kv_fused"]
        scales = cache.view(cache.shape[0], -1)[:, PAGE_SIZE * INDEX_HEAD_DIM :]
        scales.copy_(pack_ue8m0_x4(data["kv_scales"]).view(torch.uint8).view_as(scales))
        # Exercise non-contiguous physical pages, not just identity mapping.
        data["block_table"] = torch.randperm(cache.shape[0], device="cuda")[
            data["block_table"].long()
        ].int()
        lens = data["context_lens"].unsqueeze(-1)
        logits = _indexer(True)._deep_gemm_paged_mqa_logits(
            data["q_fp8"].squeeze(1),
            cache,
            data["weights"],
            lens,
            data["block_table"],
            get_paged_mqa_logits_metadata(lens, PAGE_SIZE, deep_gemm.get_num_sms()),
            max_seq_len,
        )
        self.assertEqual(logits.dtype, torch.float32)
        ref = ref_fp8_paged_mqa_logits(
            data["q_fp8"],
            data["kv_fp8"],
            data["kv_scales"],
            data["weights"].bfloat16().float(),
            data["context_lens"],
            data["block_table"],
            max_seq_len,
            PAGE_SIZE,
        )
        assert_paged_mqa_matches_ref(
            logits,
            ref,
            data["context_lens"],
            num_rows,
            1,
            max_seq_len,
            max_calc_diff=3e-5,
        )


if __name__ == "__main__":
    unittest.main()
