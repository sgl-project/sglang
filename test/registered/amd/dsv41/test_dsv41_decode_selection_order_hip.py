"""The HIP decode top-k must be ordered by position, not slot: the aiter sparse kernel sums in list order, so a slot-ordered row makes the attention bits depend on which pages a request landed on."""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

PAGE = 256
ROW_BYTES = 584  # DSV4 packed fp8 K row: 448 nope fp8 + 64 rope bf16 + scales
HEAD_DIM = 512
HEADS = 16


def _metadata(low_ratios=(1, 2)):
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DSV4AttnMetadata,
    )

    kw = dict(dtype=torch.int32, device="cuda")
    rows = torch.zeros(2, **kw)
    md = DSV4AttnMetadata(
        page_size=PAGE,
        page_table=torch.ones(2, 4, **kw),
        raw_out_loc=rows + PAGE,
        cuda_int32_kwargs=kw,
        seq_lens_casual=rows + 513,
        positions_casual=rows + 512,
        swa_page_indices=torch.zeros(2, 128, **kw),
        swa_topk_lengths=rows + 128,
        index_topk=512,
        low_ratios=low_ratios,
    )
    md.c4_topk_lengths_clamp1 = rows + 1
    md.c4_topk_lengths_raw = rows + 1
    for ratio in low_ratios:
        setattr(md, f"c{ratio}_topk_lengths_clamp1", rows + 513 // ratio)
    return md


@unittest.skipUnless(is_hip(), "HIP radix backend")
class TestDecodeSelectionOrder(CustomTestCase):
    def test_decode_metadata_carries_raw_indices(self):
        """Decode rows keep the position keys the AOT sort orders by."""
        md = _metadata()
        md.init_flashmla_related(is_prefill=False)
        for ratio in (1, 2, 4):
            raw = md.sparse_raw_indices(ratio)
            self.assertIsNotNone(
                raw, f"ratio {ratio}: decode selection would sort by slot"
            )
            self.assertEqual(raw.shape, md.sparse_page_indices(ratio).shape)

    def test_position_ordered_selection_is_page_invariant(self):
        """Same keys on two page layouts: the position-sorted selection attends bitwise
        the same, and the AOT sort with raw indices produces that order."""
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            topk_transform_paged_sorted,
        )
        from sglang.srt.layers.attention.hip_flash_mla import aiter_sparse_decode_fwd

        torch.manual_seed(0)
        dev = "cuda"
        k_a = torch.randn(PAGE, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        k_b = torch.randn(PAGE, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        k_t = torch.randn(1, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        q = torch.randn(1, 1, HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        sink = torch.zeros(HEADS, device=dev, dtype=torch.float32)
        no_swa = torch.full((1, 1, 128), -1, device=dev, dtype=torch.int32)
        # 513 positions, drop position 356 (in the middle of the second page)
        scores = torch.zeros(1, 1024, device=dev, dtype=torch.float32)
        scores[0, 356] = -1.0
        seq_lens = torch.tensor([513], device=dev, dtype=torch.int32)

        outs = []
        for page_b, page_t in ((3, 4), (4, 3)):
            cache = torch.zeros(8, PAGE * ROW_BYTES, dtype=torch.uint8, device=dev)
            slots = {
                "a": torch.arange(PAGE, 2 * PAGE, device=dev),
                "b": torch.arange(page_b * PAGE, page_b * PAGE + PAGE, device=dev),
                "t": torch.tensor([page_t * PAGE], device=dev),
            }
            for name, k in (("a", k_a), ("b", k_b), ("t", k_t)):
                fused_store_cache(
                    k, cache, slots[name], page_size=PAGE, type="flashmla"
                )
            page_table = torch.tensor(
                [[1, page_b, page_t, 0]], device=dev, dtype=torch.int32
            )
            page_indices = torch.full((1, 512), -1, device=dev, dtype=torch.int32)
            raw_indices = torch.full((1, 512), -1, device=dev, dtype=torch.int32)
            topk_transform_paged_sorted(
                scores, seq_lens, page_table, page_indices, PAGE, raw_indices
            )
            expected_positions = torch.cat(
                [torch.arange(0, 356, device=dev), torch.arange(357, 513, device=dev)]
            ).to(torch.int32)
            torch.testing.assert_close(raw_indices[0], expected_positions)
            out, _ = aiter_sparse_decode_fwd(
                q=q,
                k_cache=cache.view(8, PAGE, 1, ROW_BYTES),
                indices=no_swa,
                attn_sink=sink,
                softmax_scale=HEAD_DIM**-0.5,
                extra_k_cache=cache.view(8, PAGE, 1, ROW_BYTES),
                extra_indices_in_kvcache=page_indices.view(1, 1, 512),
            )
            outs.append(out.clone())
        self.assertTrue(
            torch.equal(outs[0], outs[1]), "attention bits follow the page layout"
        )


if __name__ == "__main__":
    unittest.main()
