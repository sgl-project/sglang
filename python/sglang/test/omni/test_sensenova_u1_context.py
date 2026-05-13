# SPDX-License-Identifier: Apache-2.0

import unittest
from base64 import b64encode
from io import BytesIO

import torch
from PIL import Image

from sglang.omni.bridges.sensenova_u1.bridge import U1OmniSessionModelPolicy
from sglang.omni.bridges.sensenova_u1.context import (
    U1_SPECIAL_TOKENS,
    U1SpecialTokens,
    U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    build_u1_native_interleave_prepared_input,
    build_u1_interleave_prompt,
    build_u1_native_generated_image_commit_prepared_input,
    build_u1_native_interleave_text_uncondition_marker_prepared_input,
    build_u1_t2i_plain_query,
    build_u1_t2i_prompt,
    build_u1_vlm_prompt,
    load_u1_native_image,
)
from sglang.srt.omni_session.runtime import (
    OmniInterleavedMessage,
    OmniSegmentState,
    OmniTextDecodeResult,
)
from sglang.srt.omni_session.runtime_types import OmniSessionHandle


class TestSenseNovaU1Context(unittest.TestCase):
    def test_prompt_builders_match_official_chatml_separator(self):
        prompts = [
            build_u1_interleave_prompt(prompt="draw a cube"),
            build_u1_t2i_prompt(prompt="draw a cube"),
            build_u1_t2i_plain_query(prompt="draw a cube"),
            build_u1_vlm_prompt(question="what is this?"),
        ]

        for prompt in prompts:
            self.assertIn("<|im_end|>\n<|im_start|>", prompt)

    def test_interleave_prompt_defaults_to_official_protocol_system_message(self):
        prompt = build_u1_interleave_prompt(prompt="draw a cube")

        self.assertTrue(
            prompt.startswith(
                "<|im_start|>system\nYou are a multimodal assistant"
            )
        )

    def test_u1_special_tokens_are_model_local_contract(self):
        self.assertIsInstance(U1_SPECIAL_TOKENS, U1SpecialTokens)
        self.assertEqual("<img>", U1_SPECIAL_TOKENS.img_start)
        self.assertEqual("</img>", U1_SPECIAL_TOKENS.img_end)
        self.assertEqual("<IMG_CONTEXT>", U1_SPECIAL_TOKENS.img_context)
        self.assertEqual("<image>", U1_SPECIAL_TOKENS.image_placeholder)

    def test_interleave_think_mode_matches_official_default_query_shape(self):
        prompt = build_u1_interleave_prompt(prompt="hi", think_mode=True)

        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>\n\n</think>\n\n", prompt)

    def test_interleave_non_think_mode_injects_official_empty_think_stub(self):
        prompt = build_u1_interleave_prompt(prompt="hi", think_mode=False)

        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_native_interleave_prepared_input_respects_think_mode(self):
        messages = [OmniInterleavedMessage(type="text", content="hi")]

        think_prepared = build_u1_native_interleave_prepared_input(
            tokenizer=_Tokenizer(),
            messages=messages,
            think_mode=True,
        )
        non_think_prepared = build_u1_native_interleave_prepared_input(
            tokenizer=_Tokenizer(),
            messages=messages,
            think_mode=False,
        )

        self.assertTrue(think_prepared.input_text.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>\n\n</think>\n\n", think_prepared.input_text)
        self.assertIn("<think>\n\n</think>\n\n", non_think_prepared.input_text)

    def test_policy_interleave_think_flag_reaches_prepared_prompt(self):
        policy = U1OmniSessionModelPolicy(native_tokenizer=_Tokenizer())
        policy.native_generation_mode = "interleave"
        messages = [OmniInterleavedMessage(type="text", content="hi")]

        policy.native_interleave_think_mode = True
        think_prepared = policy.prepare_srt_ar_interleaved_inputs(
            session=None,
            messages=messages,
            state=OmniSegmentState.AR_PREFILL,
        )
        policy.native_interleave_think_mode = False
        non_think_prepared = policy.prepare_srt_ar_interleaved_inputs(
            session=None,
            messages=messages,
            state=OmniSegmentState.AR_PREFILL,
        )

        self.assertIsNotNone(think_prepared)
        self.assertIsNotNone(non_think_prepared)
        self.assertNotIn("<think>\n\n</think>\n\n", think_prepared[0].input_text)
        self.assertIn("<think>\n\n</think>\n\n", non_think_prepared[0].input_text)

    def test_load_native_image_accepts_data_url(self):
        image = Image.new("RGB", (4, 4), (255, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        payload = "data:image/png;base64," + b64encode(buffer.getvalue()).decode()

        pixel_values, grid_hw = load_u1_native_image(
            payload,
            patch_size=2,
            downsample_ratio=0.5,
            min_pixels=16,
            max_pixels=16,
        )

        self.assertEqual((4, 12), tuple(pixel_values.shape))
        self.assertEqual([[2, 2]], grid_hw.tolist())

    def test_policy_owns_u1_image_boundary_token(self):
        policy = U1OmniSessionModelPolicy(native_tokenizer=_Tokenizer())

        self.assertTrue(policy.is_image_generation_boundary_token(123))
        self.assertFalse(policy.is_image_generation_boundary_token(124))

    def test_interleave_decode_positions_follow_official_next_output_semantics(self):
        policy = U1OmniSessionModelPolicy(native_tokenizer=_Tokenizer())
        runtime = _DecodeRuntime(output_ids=[123])
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=480,
            context_version=1,
        )

        result = policy._decode_native_interleave_next_segment(
            runtime=runtime,
            session=session,
            u1_state={
                "native_interleave_prompt": True,
                "generation_position_start": 225,
            },
        )

        self.assertEqual("image_marker", result.type)
        self.assertEqual([224], runtime.decode_positions)
        self.assertEqual((123, 225), runtime.committed_tokens[-1])
        self.assertEqual(
            226,
            runtime.model_state_updates[-1]["u1"]["generation_position_start"],
        )

    def test_interleave_decode_text_then_image_marker_positions(self):
        policy = U1OmniSessionModelPolicy(native_tokenizer=_Tokenizer())
        runtime = _DecodeRuntime(output_ids=[42, 123])
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=480,
            context_version=1,
        )

        result = policy._decode_native_interleave_next_segment(
            runtime=runtime,
            session=session,
            u1_state={
                "native_interleave_prompt": True,
                "generation_position_start": 225,
            },
        )

        self.assertEqual("text", result.type)
        self.assertEqual((42,), result.token_ids)
        self.assertEqual([224, 225], runtime.decode_positions)
        self.assertEqual((123, 226), runtime.committed_tokens[-1])
        self.assertTrue(
            runtime.model_state_updates[-1]["u1"]["interleave_pending_image_marker"]
        )
        self.assertEqual(
            227,
            runtime.model_state_updates[-1]["u1"]["generation_position_start"],
        )

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
    eos_token_id = 999

    def __call__(self, prompt, **kwargs):
        self.last_prompt = prompt
        return {"input_ids": torch.tensor([[ord(ch) for ch in prompt]])}

    def convert_tokens_to_ids(self, token):
        if token == "<img>":
            return 123
        if token == "<|im_end|>":
            return 999
        return 123

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(str(token_id) for token_id in token_ids)


class _DecodeRuntime:
    srt_ar_decode_max_new_tokens = 8
    srt_request_executor = object()

    def __init__(self, output_ids):
        self.output_ids = list(output_ids)
        self.decode_positions = []
        self.committed_tokens = []
        self.model_state_updates = []

    def decode_one_step(
        self,
        session,
        *,
        max_new_tokens,
        decode_position_id,
        greedy,
        model_state_updates,
    ):
        self.decode_positions.append(decode_position_id)
        token_id = self.output_ids.pop(0)
        return OmniTextDecodeResult(
            session=session,
            input_ids=(),
            output_ids=(token_id,),
            position_ids=(),
            text=str(token_id),
        )

    def commit_ar_decode_input_token(
        self,
        session,
        *,
        token_id,
        position_id,
        model_state_updates,
    ):
        self.committed_tokens.append((token_id, position_id))
        self.model_state_updates.append(model_state_updates)
        return session

    def get_condition_path_handle(self, session, role):
        return None

    def get_condition_path_model_state(self, session, role):
        return {}


if __name__ == "__main__":
    unittest.main()
