import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    CandidateRole,
    apply_decode_candidates,
    mask_topk_scores,
    write_paged_indexer_topk,
)
from sglang.srt.layers.attention.dsv4.indexer_capture import (
    resolve_indexer_capture_options,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIndexerCandidates(CustomTestCase):
    def test_layer_roles_and_capture_variants(self):
        for layer, source, role in (
            (0, -1, CandidateRole.NONE),
            (2, 3, CandidateRole.NONE),
            (3, 3, CandidateRole.PUBLISH),
            (4, 3, CandidateRole.CONSUME),
        ):
            self.assertIs(CandidateRole.for_layer(layer, source), role)
        for variant, all_ratios, bypass in (
            (None, (), False),
            ("candidate_filtered", (), False),
            ("candidate_all", (1, 2), True),
            ("candidate_c2_all", (2,), True),
            ("candidate_unfiltered", (), True),
            ("dense", (), False),
            ("sparse", (), False),
        ):
            for ratio in (1, 2):
                with self.subTest(variant=variant, ratio=ratio):
                    options = resolve_indexer_capture_options(ratio, variant)
                    self.assertEqual(options.select_all, ratio in all_ratios)
                    self.assertEqual(options.bypass_candidates, bypass)

    def test_publish_consume_and_bypass_are_distinct(self):
        scores = torch.tensor([[1.0, 2.0, 9.0, 8.0], [3.0, 4.0, 99.0, 99.0]])
        lengths = torch.tensor([4, 2])
        visible, published = apply_decode_candidates(
            scores,
            lengths,
            candidate_role=CandidateRole.PUBLISH,
            topk_blocks=1,
            block_size=2,
            published=None,
        )
        self.assertTrue(torch.isneginf(visible[1, 2:]).all())
        torch.testing.assert_close(
            published,
            torch.tensor([[False, False, True, True], [True, True, False, False]]),
        )
        consumed, output = apply_decode_candidates(
            scores,
            lengths,
            candidate_role=CandidateRole.CONSUME,
            topk_blocks=1,
            block_size=2,
            published=published,
        )
        self.assertIsNone(output)
        self.assertTrue(torch.isneginf(consumed[~published]).all())
        unchanged, output = apply_decode_candidates(
            scores,
            lengths,
            candidate_role=CandidateRole.NONE,
            topk_blocks=1,
            block_size=2,
            published=None,
        )
        self.assertIs(unchanged, scores)
        self.assertIsNone(output)

    def test_paged_topk_dispatch_and_masked_output(self):
        scores = torch.tensor([[9.0, -torch.inf, 7.0, -torch.inf]])
        lengths = torch.tensor([4], dtype=torch.int32)
        table = torch.tensor([[3, 1]], dtype=torch.int32)
        logical = torch.tensor([[0, 2, 1]], dtype=torch.int32)

        def slots(indices):
            columns = indices.to(torch.long)
            return (table.gather(1, columns // 2) * 2 + columns % 2).to(torch.int32)

        for use_v2, capture_raw, mask in product((False, True), repeat=3):
            with self.subTest(use_v2=use_v2, capture_raw=capture_raw, mask=mask):
                pages = torch.full_like(logical, -2)
                raw = torch.full_like(logical, -2) if capture_raw else None

                def v1(logits, lens, page_table, out, page_size, out_raw):
                    out.copy_(slots(logical))
                    if out_raw is not None:
                        out_raw.copy_(logical)

                def v2(logits, lens, page_table, out, page_size, metadata):
                    out.copy_(logical if page_table is None else slots(logical))

                with (
                    patch(
                        "sglang.srt.layers.attention.dsv4.indexer.topk_transform_paged",
                        side_effect=v1,
                    ) as first,
                    patch(
                        "sglang.srt.layers.attention.dsv4.indexer.topk_transform_paged_v2",
                        side_effect=v2,
                    ) as second,
                ):
                    write_paged_indexer_topk(
                        scores,
                        lengths,
                        table,
                        pages,
                        raw,
                        page_size=2,
                        use_topk_v2=use_v2,
                        topk_metadata=None,
                        mask_topk=mask,
                    )
                self.assertEqual(second.call_count, int(use_v2 and not capture_raw))
                self.assertEqual(
                    first.call_count, int(not (use_v2 and not capture_raw))
                )
                expected_pages = torch.tensor(
                    [[6, 2, -1 if mask else 7]], dtype=torch.int32
                )
                torch.testing.assert_close(pages, expected_pages)
                if raw is not None:
                    torch.testing.assert_close(
                        raw,
                        torch.tensor([[0, 2, -1 if mask else 1]], dtype=torch.int32),
                    )

    def test_ragged_score_mask_uses_logical_offsets(self):
        scores = torch.tensor([[4.0, -torch.inf, 2.0], [1.0, 2.0, 3.0]])
        selected = torch.tensor([[10, 11, 12, 13], [19, 20, 22, -1]], dtype=torch.int32)
        result = mask_topk_scores(scores, selected, torch.tensor([10, 20]))
        torch.testing.assert_close(
            result,
            torch.tensor([[10, -1, 12, -1], [-1, 20, 22, -1]], dtype=torch.int32),
        )

    def test_ragged_publication_releases_batched_mask(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        backend = object.__new__(DeepseekV4AttnBackend)
        backend.candidate_mask = torch.ones(1, 4, dtype=torch.bool)
        backend._publish_or_consume_candidates(
            SimpleNamespace(
                candidate_role=CandidateRole.PUBLISH,
                candidate_topk_blocks=1,
                candidate_block_size=2,
            ),
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([4]),
            [4],
            [1],
            torch.zeros(0, 0, dtype=torch.bool),
        )
        self.assertIsNone(backend.candidate_mask)
        torch.testing.assert_close(
            backend.candidate_masks_by_request[0],
            torch.tensor([[False, False, True, True]]),
        )

    def test_tail_slices_ragged_masks_and_restores_state(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        backend = object.__new__(DeepseekV4AttnBackend)
        masks = [torch.ones(4, 3, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool)]
        batched_mask = torch.ones(2, 3, dtype=torch.bool)
        full = SimpleNamespace(core_attn_metadata=SimpleNamespace(low_ratios=()))
        tail = SimpleNamespace(
            core_attn_metadata=SimpleNamespace(low_ratios=()),
            late_layer_tail=SimpleNamespace(
                cp_metadata=None, extend_seq_lens_cpu=[2, 0]
            ),
        )
        backend.forward_metadata = full
        backend.tail_forward_metadata = tail
        backend.candidate_masks_by_request = masks
        backend.candidate_mask = batched_mask
        backend.token_to_kv_pool = SimpleNamespace(request_window=None)
        batch = SimpleNamespace(attn_cp_metadata=None)
        with (
            patch(
                "sglang.srt.layers.attention.deepseek_v4_backend.get_local_dp_buffer_len",
                return_value=6,
            ),
            patch(
                "sglang.srt.layers.attention.deepseek_v4_backend.set_local_dp_buffer_len"
            ) as restore,
        ):
            saved = backend.enter_late_layer_tail(batch)
            self.assertIs(backend.forward_metadata, tail)
            self.assertEqual(
                [mask.shape[0] for mask in backend.candidate_masks_by_request], [2, 0]
            )
            self.assertIs(backend.candidate_mask, batched_mask)
            backend.exit_late_layer_tail(saved, batch)
            self.assertIs(backend.forward_metadata, full)
            self.assertIs(backend.candidate_masks_by_request, masks)
            self.assertIs(backend.candidate_mask, batched_mask)
            restore.assert_called_once_with(6)


if __name__ == "__main__":
    unittest.main()
