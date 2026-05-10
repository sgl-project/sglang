# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.omni.bridges.sensenova_u1.context import (
    U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
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


class _Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 123


if __name__ == "__main__":
    unittest.main()
