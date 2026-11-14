import unittest

import torch

from sglang.srt.managers.embedding_truncation import (
    embedding_input_token_limit,
    truncate_embedding_input,
)
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.multimodal.modality import Modality
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEmbeddingTruncation(unittest.TestCase):
    def truncate(self, ids, policy="END", max_tokens=None, limit=8, **kwargs):
        return truncate_embedding_input(
            ids,
            truncate=policy,
            max_tokens=max_tokens,
            input_token_limit=limit,
            **kwargs,
        )

    @staticmethod
    def mm(ids, offsets, **kwargs):
        return MultimodalProcessorOutput(
            input_ids=list(ids),
            padded_input_ids=list(ids),
            token_type_ids=torch.zeros(len(ids), dtype=torch.long),
            mm_items=[
                MultimodalDataItem(modality=Modality.IMAGE, offsets=spans)
                for spans in offsets
            ],
            **kwargs,
        )

    def test_effective_context_scheduler_and_prefill_limits(self):
        cases = [
            ({}, 127),
            ({"num_reserved_tokens": 4}, 123),
            ({"max_req_input_len": 100}, 99),
            ({"max_req_input_len": 100, "max_prefill_tokens": 64}, 64),
        ]
        for overrides, expected in cases:
            kwargs = dict(
                context_len=128, num_reserved_tokens=0, max_req_input_len=None
            )
            kwargs.update(overrides)
            with self.subTest(overrides=overrides):
                self.assertEqual(embedding_input_token_limit(**kwargs), expected)
        with self.assertRaisesRegex(ValueError, "no available"):
            embedding_input_token_limit(
                context_len=1, num_reserved_tokens=0, max_req_input_len=None
            )

    def test_direction_and_aligned_token_types(self):
        ids = list(range(12))
        types = list(range(100, 112))
        for policy, expected_ids, expected_types in (
            ("END", ids[:5], types[:5]),
            ("START", ids[-5:], types[-5:]),
        ):
            with self.subTest(policy=policy):
                result, token_types, _ = self.truncate(
                    ids, policy, max_tokens=5, token_type_ids=types
                )
                self.assertEqual(result, expected_ids)
                self.assertEqual(token_types, expected_types)
        self.assertEqual(ids, list(range(12)))
        self.assertEqual(types, list(range(100, 112)))

    def test_exact_boundary_and_one_token_over(self):
        for policy in ("END", "START", "NONE"):
            with self.subTest(policy=policy):
                self.assertEqual(
                    self.truncate(list(range(8)), policy)[0], list(range(8))
                )
        self.assertEqual(self.truncate(list(range(9)))[0], list(range(8)))
        with self.assertRaisesRegex(ValueError, "truncate=NONE"):
            self.truncate(list(range(9)), "NONE")

    def test_special_token_mask_preserves_template_not_literal_special_ids(self):
        # The second 99 is literal text, not a token added by the template.
        ids = [98, 10, 99, 11, 12, 13, 99]
        mask = [1, 0, 0, 0, 0, 0, 1]
        for policy, positions in (("END", [0, 1, 2, 6]), ("START", [0, 4, 5, 6])):
            with self.subTest(policy=policy):
                result, types, _ = self.truncate(
                    ids,
                    policy,
                    max_tokens=4,
                    special_tokens_mask=mask,
                    token_type_ids=list(range(len(ids))),
                )
                self.assertEqual(result, [ids[i] for i in positions])
                self.assertEqual(types, positions)
        self.assertEqual(ids, [98, 10, 99, 11, 12, 13, 99])
        self.assertEqual(mask, [1, 0, 0, 0, 0, 0, 1])

    def test_special_tokens_count_toward_the_entire_budget(self):
        for policy in ("END", "START"):
            with self.subTest(policy=policy):
                self.assertEqual(
                    self.truncate(
                        [98, 10, 11, 99],
                        policy,
                        max_tokens=2,
                        special_tokens_mask=[1, 0, 0, 1],
                    )[0],
                    [98, 99],
                )
                with self.assertRaisesRegex(ValueError, "required special tokens"):
                    self.truncate(
                        [98, 10, 11, 99],
                        policy,
                        max_tokens=1,
                        special_tokens_mask=[1, 0, 0, 1],
                    )
        with self.assertRaisesRegex(ValueError, "truncate=NONE"):
            self.truncate(
                [98, 10, 11, 99], "NONE", max_tokens=3, special_tokens_mask=[1, 0, 0, 1]
            )

    def test_misaligned_special_token_mask_is_rejected(self):
        for mask in ([1, 0], [1, 2, 0, 1]):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(ValueError, "special_tokens_mask"):
                    self.truncate(
                        [98, 10, 11, 99], max_tokens=3, special_tokens_mask=mask
                    )

    def test_special_token_selection_keeps_tensor_metadata_aligned(self):
        types = torch.tensor([[0, 1, 2, 3, 4, 5]])
        ids, selected, _ = self.truncate(
            [98, 10, 11, 12, 13, 99],
            "START",
            max_tokens=4,
            special_tokens_mask=[1, 0, 0, 0, 0, 1],
            token_type_ids=types,
        )
        self.assertEqual(ids, [98, 12, 13, 99])
        self.assertEqual(selected.tolist(), [[0, 3, 4, 5]])
        self.assertEqual(types.tolist(), [[0, 1, 2, 3, 4, 5]])

    def test_none_checks_explicit_and_server_budgets(self):
        with self.assertRaisesRegex(ValueError, "budget of 3"):
            self.truncate([1, 2, 3, 4], "NONE", max_tokens=3)
        self.assertEqual(self.truncate([1, 2, 3], "NONE", max_tokens=20)[0], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "budget of 8"):
            self.truncate(list(range(9)), "NONE", max_tokens=20)

    def test_oversize_explicit_budget_rejected_even_for_short_input(self):
        for policy in ("END", "START"):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, "max_tokens=9.*budget"):
                    self.truncate([1], policy, max_tokens=9)

    def test_invalid_budget(self):
        for value in (-2, True, 1.5, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "integer greater than or equal to -1"
                ):
                    self.truncate([1], max_tokens=value)

    def test_minus_one_uses_server_limit_only_for_end_and_start(self):
        ids = list(range(12))
        for policy, expected in (("END", ids[:8]), ("START", ids[-8:])):
            with self.subTest(policy=policy):
                self.assertEqual(self.truncate(ids, policy, max_tokens=-1)[0], expected)
        for ids in ([], [1], list(range(12))):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, "budget of -1.*truncate=NONE"):
                    self.truncate(ids, "NONE", max_tokens=-1)

    def test_zero_budget_is_not_a_default_or_negative_zero_slice(self):
        for policy in ("END", "START"):
            with self.subTest(policy=policy):
                result, types, _ = self.truncate(
                    [1, 2, 3], policy, max_tokens=0, token_type_ids=[0, 0, 1]
                )
                self.assertEqual(result, [])
                self.assertEqual(types, [])
                self.assertEqual(self.truncate([], policy, max_tokens=0)[0], [])
        self.assertEqual(self.truncate([], "NONE", max_tokens=0)[0], [])
        with self.assertRaisesRegex(ValueError, "budget of 0.*truncate=NONE"):
            self.truncate([1], "NONE", max_tokens=0)

    def test_request_isolation(self):
        ids = list(range(12))
        self.assertEqual(self.truncate(ids, "END", 4)[0], [0, 1, 2, 3])
        self.assertEqual(self.truncate(ids, "START", 3)[0], [9, 10, 11])
        with self.assertRaisesRegex(ValueError, "truncate=NONE"):
            self.truncate(ids, "NONE", 4)
        self.assertEqual(ids, list(range(12)))

    def test_mm_tail_keeps_all_features_and_updates_parallel_ids(self):
        ids = list(range(14))
        mm = self.mm(ids, [[(1, 2), (4, 5)], [(7, 8)]])
        result, types, updated = self.truncate(
            ids, max_tokens=10, limit=20, token_type_ids=[1] * 14, mm_inputs=mm
        )
        self.assertEqual(result, list(range(10)))
        self.assertEqual(types, [1] * 10)
        self.assertEqual(updated.input_ids, result)
        self.assertEqual(updated.padded_input_ids, result)
        self.assertEqual(updated.token_type_ids.tolist(), [0] * 10)
        self.assertIs(updated.mm_items, mm.mm_items)
        self.assertEqual(updated.mm_items[0].offsets, [(1, 2), (4, 5)])
        self.assertEqual(mm.input_ids, ids)
        self.assertEqual(mm.padded_input_ids, ids)
        self.assertEqual(mm.token_type_ids.tolist(), [0] * 14)
        self.assertIsNot(updated, mm)

    def test_mm_rejects_cut_media_removed_media_and_closing_delimiter(self):
        ids = list(range(14))
        for budget in (2, 5, 8, 9):
            with self.subTest(budget=budget):
                mm = self.mm(ids, [[(1, 2), (4, 5)], [(7, 8)]])
                with self.assertRaisesRegex(ValueError, "remove media tokens"):
                    self.truncate(ids, max_tokens=budget, limit=20, mm_inputs=mm)
                self.assertEqual(mm.input_ids, ids)

    def test_mm_start_is_allowed_without_trimming_but_not_with_trimming(self):
        ids = list(range(8))
        mm = self.mm(ids, [[(2, 3)]])
        self.assertEqual(self.truncate(ids, "START", mm_inputs=mm)[0], ids)
        with self.assertRaisesRegex(ValueError, "truncate=START.*multimodal"):
            self.truncate(ids, "START", 7, mm_inputs=mm)

    def test_mm_none_checks_expanded_token_count(self):
        ids = list(range(12))
        mm = self.mm(ids, [[(1, 8)]])
        with self.assertRaisesRegex(ValueError, "Input of 12 tokens"):
            self.truncate(ids, "NONE", mm_inputs=mm)

    def test_mm_incomplete_expansion_rejected_without_mutation(self):
        ids = list(range(8))
        mm = self.mm(ids, [[(1, 3)]])
        mm.padded_input_ids = list(range(20))
        with self.assertRaisesRegex(ValueError, "expansion is complete"):
            self.truncate(ids, "NONE", mm_inputs=mm)
        self.assertEqual(len(mm.padded_input_ids), 20)

    def test_deferred_mm_expansion_rejected_even_when_raw_prompt_fits(self):
        ids = [1, 2]
        mm = self.mm(ids, [[(1, 1)]])
        mm.input_ids = None
        with self.assertRaisesRegex(ValueError, "defers multimodal token expansion"):
            self.truncate(ids, "NONE", mm_inputs=mm)

    def test_mm_unknown_offsets_and_overlapping_offsets_rejected(self):
        ids = list(range(12))
        cases = [
            ([None], "without media token offsets"),
            ([[(1, 3)], [(3, 4)]], "overlapping"),
            ([[(-1, 2)]], "invalid"),
            ([[(4, 3)]], "invalid"),
            ([[(1, 15)]], "invalid"),
        ]
        for offsets, message in cases:
            with self.subTest(offsets=offsets):
                with self.assertRaisesRegex(ValueError, message):
                    self.truncate(ids, mm_inputs=self.mm(ids, offsets))

    def test_mm_positional_metadata_requires_dedicated_adapter(self):
        ids = list(range(12))
        for name in ("mrope_positions", "mrope_position_delta", "vision_position_ids"):
            with self.subTest(name=name):
                mm = self.mm(
                    ids, [[(1, 2)]], **{name: torch.zeros(1, dtype=torch.long)}
                )
                with self.assertRaisesRegex(ValueError, name):
                    self.truncate(ids, mm_inputs=mm)

    def test_mm_boundary_tokens_cannot_be_removed(self):
        ids = list(range(12))
        for name in (
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "slice_start_id",
            "slice_end_id",
            "video_token_id",
            "audio_token_id",
            "audio_start_id",
            "audio_end_id",
        ):
            with self.subTest(field=name):
                mm = self.mm(ids, [[(1, 2)]], **{name: 10})
                with self.assertRaisesRegex(ValueError, "boundary token"):
                    self.truncate(ids, mm_inputs=mm)

    def test_mm_optional_metadata_stays_none(self):
        ids = list(range(12))
        mm = MultimodalProcessorOutput(
            input_ids=ids,
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE, offsets=[(1, 2)])],
            im_end_id=3,
            media_nums_per_sample=[1],
        )
        result, _, updated = self.truncate(ids, mm_inputs=mm)
        self.assertEqual(result, ids[:8])
        self.assertIsInstance(updated, MultimodalProcessorOutput)
        self.assertIsNone(updated.padded_input_ids)
        self.assertIsNone(updated.token_type_ids)
        self.assertEqual(updated.im_end_id, 3)
        self.assertEqual(updated.media_nums_per_sample, [1])
        self.assertIs(updated.mm_items, mm.mm_items)
        self.assertEqual(mm.input_ids, ids)

    def test_mm_tensor_metadata_alignment_and_isolation(self):
        ids = list(range(12))
        mm = self.mm(ids, [[(1, 2)]])
        original = torch.arange(24).reshape(2, 12)
        mm.token_type_ids = original
        _, _, updated = self.truncate(ids, mm_inputs=mm)
        torch.testing.assert_close(updated.token_type_ids, original[:, :8])
        self.assertIs(mm.token_type_ids, original)
        self.assertEqual(mm.token_type_ids.shape, (2, 12))
        for invalid in (torch.tensor(1), torch.arange(4)):
            with self.subTest(shape=invalid.shape):
                mm.token_type_ids = invalid
                with self.assertRaisesRegex(ValueError, "misaligned"):
                    self.truncate(ids, mm_inputs=mm)
                self.assertIs(mm.token_type_ids, invalid)
                self.assertEqual(mm.input_ids, ids)

    def test_misaligned_token_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "misaligned token_type_ids"):
            self.truncate(list(range(12)), token_type_ids=[0, 1])


if __name__ == "__main__":
    unittest.main()
