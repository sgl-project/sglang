# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from sglang.omni.bridges.sensenova_u1.context import (
    U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    build_u1_native_generated_image_commit_prepared_input,
    build_u1_native_interleave_text_uncondition_marker_prepared_input,
)
from sglang.srt.omni_session.runtime_protocol import OmniSessionHandle


class TestSenseNovaU1Context(unittest.TestCase):
    def test_marker_prepare_accepts_raw_session_handle(self):
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=8,
            context_version=1,
        )

        prepared = build_u1_native_interleave_text_uncondition_marker_prepared_input(
            tokenizer=_Tokenizer(),
            session=session,
            logical_position=8,
        )

        self.assertEqual(
            f"s0:{U1_INTERLEAVE_TEXT_UNCONDITION_ROLE}",
            prepared.condition_path_session_id,
        )
        self.assertEqual(
            "s0",
            prepared.policy_metadata["omni_model_state_updates"]["u1"]["session_id"],
        )

    def test_generated_image_commit_accepts_precomputed_embeddings(self):
        embeddings = torch.randn(4, 8)
        prepared = build_u1_native_generated_image_commit_prepared_input(
            tokenizer=_Tokenizer(),
            image={
                "precomputed_embeddings": embeddings,
                "grid_hw": torch.tensor([[4, 4]]),
                "pad_hash": 77,
            },
            patch_size=2,
            downsample_ratio=0.5,
        )

        item = prepared.mm_inputs.mm_items[0]
        self.assertIsNone(item.feature)
        self.assertEqual(77, item.hash)
        self.assertIsNotNone(item.pad_value)
        self.assertEqual((4, 8), tuple(item.precomputed_embeddings.shape))


class _Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 123


if __name__ == "__main__":
    unittest.main()
