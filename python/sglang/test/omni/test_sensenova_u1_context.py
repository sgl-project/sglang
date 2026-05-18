# SPDX-License-Identifier: Apache-2.0

import unittest
from base64 import b64encode
from io import BytesIO

import torch
from PIL import Image

from sglang.omni.core.interleaved import (
    INTERLEAVED_BOUNDARY_MODALITY_KEY,
    INTERLEAVED_BOUNDARY_POSITION_ID_KEY,
    INTERLEAVED_BOUNDARY_TOKEN_ID_KEY,
    INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY,
    STREAMED_TEXT_METADATA_KEY,
    TEXT_ROLE_METADATA_KEY,
    TEXT_ROLE_THINK,
)
from sglang.omni.model_adapters.sensenova_u1.context import (
    U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    U1_SPECIAL_TOKENS,
    U1SpecialTokens,
    build_u1_interleave_prompt,
    build_u1_native_edit_img_condition_prepared_input,
    build_u1_native_edit_prepared_input,
    build_u1_native_generated_image_commit_prepared_input,
    build_u1_native_interleave_prepared_input,
    build_u1_native_interleave_text_uncondition_marker_prepared_input,
    build_u1_native_t2i_prepared_input,
    build_u1_t2i_plain_query,
    build_u1_t2i_prompt,
    build_u1_vlm_prompt,
    load_u1_native_image,
)
from sglang.omni.model_adapters.sensenova_u1.session_adapter import (
    U1OmniSessionModelHooks,
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
            prompt.startswith("<|im_start|>system\nYou are a multimodal assistant")
        )

    def test_interleave_system_message_hides_planning_from_final_answer(self):
        prompt = build_u1_interleave_prompt(prompt="hi", think_mode=True)

        self.assertIn("final answer must not include hidden reasoning", prompt)
        self.assertIn("planning steps", prompt)
        self.assertIn("explicit image prompts", prompt)

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

        self.assertTrue(
            prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        )

    def test_t2i_think_mode_matches_official_open_think_prefix(self):
        prompt = build_u1_t2i_prompt(prompt="draw a cube", think_mode=True)

        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertNotIn("</think>\n\n<img>", prompt)

    def test_t2i_non_think_mode_injects_official_empty_think_and_image_marker(self):
        prompt = build_u1_t2i_prompt(prompt="draw a cube", think_mode=False)

        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n<img>"))

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

    def test_hooks_interleave_think_flag_reaches_prepared_prompt(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_Tokenizer())
        hooks.native_generation_mode = "interleave"
        messages = [OmniInterleavedMessage(type="text", content="hi")]

        hooks.native_interleave_think_mode = True
        think_prepared = hooks.prepare_srt_ar_interleaved_inputs(
            session=None,
            messages=messages,
            state=OmniSegmentState.AR_PREFILL,
        )
        hooks.native_interleave_think_mode = False
        non_think_prepared = hooks.prepare_srt_ar_interleaved_inputs(
            session=None,
            messages=messages,
            state=OmniSegmentState.AR_PREFILL,
        )

        self.assertIsNotNone(think_prepared)
        self.assertIsNotNone(non_think_prepared)
        self.assertNotIn("<think>\n\n</think>\n\n", think_prepared[0].input_text)
        self.assertIn("<think>\n\n</think>\n\n", non_think_prepared[0].input_text)
        self.assertTrue(
            think_prepared[-1].policy_metadata["omni_model_state_updates"]["u1"][
                "interleave_think_mode"
            ]
        )
        self.assertFalse(
            think_prepared[-1].policy_metadata["omni_model_state_updates"]["u1"][
                "interleave_thinking_done"
            ]
        )

    def test_hooks_t2i_think_flag_reaches_prepared_prompt(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_Tokenizer())
        hooks.native_generation_mode = "t2i"
        messages = [OmniInterleavedMessage(type="text", content="hi")]

        hooks.native_interleave_think_mode = True
        think_prepared = hooks.prepare_srt_ar_interleaved_inputs(
            session=None,
            messages=messages,
            state=OmniSegmentState.AR_PREFILL,
        )
        hooks.native_interleave_think_mode = False
        non_think_prepared = hooks.prepare_srt_ar_interleaved_inputs(
            session=None,
            messages=messages,
            state=OmniSegmentState.AR_PREFILL,
        )

        self.assertIsNotNone(think_prepared)
        self.assertIsNotNone(non_think_prepared)
        self.assertTrue(think_prepared[-1].input_text.endswith("<think>\n"))
        self.assertFalse(
            think_prepared[-1].policy_metadata["omni_model_state_updates"]["u1"][
                "open_image_marker"
            ]
        )
        self.assertTrue(non_think_prepared[-1].input_text.endswith("</think>\n\n<img>"))

    def test_t2i_prepared_input_records_think_mode(self):
        prepared = build_u1_native_t2i_prepared_input(
            tokenizer=_Tokenizer(),
            messages=[OmniInterleavedMessage(type="text", content="hi")],
            think_mode=True,
        )

        self.assertTrue(prepared.policy_metadata["u1"]["think_mode"])
        self.assertTrue(
            prepared.policy_metadata["omni_model_state_updates"]["u1"]["t2i_think_mode"]
        )

    def test_edit_prepared_input_records_think_mode_without_open_image_marker(self):
        prepared = build_u1_native_edit_prepared_input(
            tokenizer=_Tokenizer(),
            messages=[
                OmniInterleavedMessage(
                    type="image",
                    content={
                        "pixel_values": torch.zeros(4, 12),
                        "grid_hw": torch.tensor([[4, 4]]),
                    },
                ),
                OmniInterleavedMessage(type="text", content="make it red"),
            ],
            think_mode=True,
        )
        u1_updates = prepared.policy_metadata["omni_model_state_updates"]["u1"]

        self.assertTrue(prepared.input_text.endswith("<think>\n"))
        self.assertIsNotNone(prepared.position_ids)
        self.assertEqual(len(prepared.input_ids), len(prepared.position_ids))
        self.assertEqual(3, len(prepared.position_ids[0]))
        self.assertEqual(
            u1_updates["generation_position_start"],
            max(position[0] for position in prepared.position_ids) + 1,
        )
        self.assertTrue(u1_updates["edit_think_mode"])
        self.assertFalse(u1_updates["open_image_marker"])
        self.assertGreater(u1_updates["generation_position_start"], 0)

    def test_edit_image_condition_path_uses_vlm_position_ids(self):
        prepared = build_u1_native_edit_img_condition_prepared_input(
            tokenizer=_Tokenizer(),
            messages=[
                OmniInterleavedMessage(
                    type="image",
                    content={
                        "pixel_values": torch.zeros(4, 12),
                        "grid_hw": torch.tensor([[4, 4]]),
                    },
                )
            ],
        )
        u1_metadata = prepared.policy_metadata["u1"]

        self.assertIsNotNone(prepared.position_ids)
        self.assertEqual(len(prepared.input_ids), len(prepared.position_ids))
        self.assertEqual(3, len(prepared.position_ids[0]))
        self.assertEqual(
            u1_metadata["generation_position_start"],
            max(position[0] for position in prepared.position_ids) + 1,
        )

    def test_t2i_think_decode_appends_image_marker_after_think(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_Tokenizer())
        runtime = _NativeThinkRuntime(output_ids=[42, 124])
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=32,
            context_version=1,
        )

        result = hooks.decode_vlm_text(
            runtime=runtime,
            session=session,
            max_new_tokens=8,
        )

        self.assertEqual((42, 124), result.next_token_ids)
        self.assertEqual([10, 10, 123], runtime.appended_token_ids)
        self.assertEqual([31, 32], runtime.decode_positions)
        self.assertEqual([34, 35, 36], runtime.appended_position_ids)
        self.assertTrue(runtime.model_state_updates[-1]["u1"]["open_image_marker"])

    def test_t2i_think_decode_streams_tokens_before_image_marker(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_Tokenizer())
        runtime = _NativeThinkRuntime(output_ids=[42, 124])
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=32,
            context_version=1,
        )
        stream_sink = _TextDeltaSink()

        result = hooks.decode_vlm_text(
            runtime=runtime,
            session=session,
            max_new_tokens=8,
            stream_sink=stream_sink,
        )

        self.assertTrue(result.streamed_text)
        self.assertEqual(["42", "124"], stream_sink.deltas)
        self.assertEqual(
            [{TEXT_ROLE_METADATA_KEY: TEXT_ROLE_THINK}] * 2,
            stream_sink.metadata,
        )
        self.assertEqual([10, 10, 123], runtime.appended_token_ids)

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

    def test_hooks_owns_u1_image_boundary_token(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_Tokenizer())

        self.assertTrue(hooks.is_image_generation_boundary_token(123))
        self.assertFalse(hooks.is_image_generation_boundary_token(124))

    def test_interleave_decode_positions_follow_official_next_output_semantics(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_Tokenizer())
        runtime = _DecodeRuntime(output_ids=[123])
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=480,
            context_version=1,
        )

        result = hooks._decode_native_interleave_next_segment(
            runtime=runtime,
            session=session,
            u1_state={
                "native_interleave_prompt": True,
                "generation_position_start": 225,
            },
        )

        self.assertEqual("image_marker", result.type)
        self.assertEqual(
            {
                INTERLEAVED_BOUNDARY_MODALITY_KEY: "image",
                INTERLEAVED_BOUNDARY_TOKEN_ID_KEY: 123,
                INTERLEAVED_BOUNDARY_POSITION_ID_KEY: 225,
            },
            result.metadata,
        )
        self.assertEqual([224], runtime.decode_positions)
        self.assertEqual((123, 225), runtime.committed_tokens[-1])
        self.assertEqual(
            226,
            runtime.model_state_updates[-1]["u1"]["generation_position_start"],
        )

    def test_interleave_decode_text_then_image_marker_positions(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_Tokenizer())
        runtime = _DecodeRuntime(output_ids=[42, 123])
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=480,
            context_version=1,
        )

        result = hooks._decode_native_interleave_next_segment(
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
            {
                INTERLEAVED_BOUNDARY_MODALITY_KEY: "image",
                INTERLEAVED_BOUNDARY_TOKEN_ID_KEY: 123,
                INTERLEAVED_BOUNDARY_POSITION_ID_KEY: 226,
            },
            runtime.model_state_updates[-1]["u1"][
                INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY
            ],
        )
        self.assertEqual(
            227,
            runtime.model_state_updates[-1]["u1"]["generation_position_start"],
        )

    def test_interleave_think_role_survives_image_handoff_until_think_end(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_Tokenizer())
        runtime = _DecodeRuntime(output_ids=[42, 124])
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=480,
            context_version=1,
        )
        stream_sink = _TextDeltaSink()

        result = hooks._decode_native_interleave_next_segment(
            runtime=runtime,
            session=session,
            u1_state={
                "native_interleave_prompt": True,
                "interleave_think_mode": True,
                "interleave_thinking_done": False,
                "generation_position_start": 225,
            },
            stream_sink=stream_sink,
        )

        self.assertEqual("text", result.type)
        self.assertEqual((42, 124), result.token_ids)
        self.assertEqual(TEXT_ROLE_THINK, result.metadata[TEXT_ROLE_METADATA_KEY])
        self.assertTrue(result.metadata[STREAMED_TEXT_METADATA_KEY])
        self.assertEqual(
            [{TEXT_ROLE_METADATA_KEY: TEXT_ROLE_THINK}] * 2,
            stream_sink.metadata,
        )
        self.assertTrue(
            runtime.model_state_updates[-1]["u1"]["interleave_thinking_done"]
        )

    def test_interleave_text_after_think_end_is_user_visible(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_Tokenizer())
        runtime = _DecodeRuntime(output_ids=[42, 999])
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=480,
            context_version=1,
        )

        result = hooks._decode_native_interleave_next_segment(
            runtime=runtime,
            session=session,
            u1_state={
                "native_interleave_prompt": True,
                "interleave_think_mode": True,
                "interleave_thinking_done": True,
                "generation_position_start": 225,
            },
        )

        self.assertEqual("text", result.type)
        self.assertEqual((42,), result.token_ids)
        self.assertNotIn(TEXT_ROLE_METADATA_KEY, result.metadata)

    def test_interleave_planning_after_think_end_stays_hidden(self):
        hooks = U1OmniSessionModelHooks(native_tokenizer=_PlanningTokenizer())
        runtime = _DecodeRuntime(output_ids=[42, 999])
        session = OmniSessionHandle(
            session_id="s0",
            anchor_request_id="r0",
            context_length=480,
            context_version=1,
        )
        stream_sink = _TextDeltaSink()

        result = hooks._decode_native_interleave_next_segment(
            runtime=runtime,
            session=session,
            u1_state={
                "native_interleave_prompt": True,
                "interleave_think_mode": True,
                "interleave_thinking_done": True,
                "generation_position_start": 225,
                "last_generated_image_commit": True,
            },
            stream_sink=stream_sink,
        )

        self.assertEqual("done", result.type)
        self.assertEqual([], stream_sink.deltas)

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
        special_tokens = {
            "<IMG_CONTEXT>": 125,
            "<|im_end|>": 999,
            "</think>": 124,
            "</img>": 126,
            "<img>": 123,
        }
        input_ids = []
        index = 0
        while index < len(prompt):
            for token, token_id in special_tokens.items():
                if prompt.startswith(token, index):
                    input_ids.append(token_id)
                    index += len(token)
                    break
            else:
                input_ids.append(ord(prompt[index]))
                index += 1
        return {"input_ids": torch.tensor([input_ids])}

    def convert_tokens_to_ids(self, token):
        if token == "<img>":
            return 123
        if token == "</think>":
            return 124
        if token == "<IMG_CONTEXT>":
            return 125
        if token == "</img>":
            return 126
        if token == "<|im_end|>":
            return 999
        return 123

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(str(token_id) for token_id in token_ids)


class _PlanningTokenizer(_Tokenizer):
    def decode(self, token_ids, skip_special_tokens=True):
        if list(token_ids) == [42]:
            return "The user has initiated a conversation. I will craft a response."
        return super().decode(token_ids, skip_special_tokens=skip_special_tokens)


class _DecodeRuntime:
    srt_ar_decode_max_new_tokens = 8
    srt_request_executor = object()

    def __init__(self, output_ids):
        self.output_ids = list(output_ids)
        self.decode_positions = []
        self.committed_tokens = []
        self.model_state_updates = []

    def decode(
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

    def merge_model_state_updates(self, session, *, namespace, updates):
        self.model_state_updates.append({namespace: updates})

    def get_condition_path_handle(self, session, role):
        return None

    def get_condition_path_model_state(self, session, role):
        return {}


class _NativeThinkRuntime:
    def __init__(self, output_ids):
        self.output_ids = list(output_ids)
        self.decode_positions = []
        self.appended_token_ids = []
        self.appended_position_ids = []
        self.model_state_updates = []

    def get_model_state(self, session, *, namespace):
        return {"t2i_think_mode": True}

    def decode(
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

    def append_ar_input_tokens(
        self, session, *, token_ids, position_ids, model_state_updates
    ):
        self.appended_token_ids.extend(token_ids)
        self.appended_position_ids.extend(position_ids)
        self.model_state_updates.append(model_state_updates)
        return session


class _TextDeltaSink:
    def __init__(self):
        self.deltas = []
        self.metadata = []

    def text_delta(self, delta, *, token_id=None, metadata=None):
        self.deltas.append(delta)
        self.metadata.append(metadata)


if __name__ == "__main__":
    unittest.main()
