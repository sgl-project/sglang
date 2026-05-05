# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from transformers import AutoConfig

from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.multimodal_gen.runtime.pipelines.ug import (
    _build_ug_g_segment_executor,
    _load_ug_bridge,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ug_bagel import (
    BAGELLatentFlowGSegmentExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ug_u1 import (
    U1PixelFlowGSegmentExecutor,
    _u1_guidance_branch,
    _u1_patch_grid,
    _u1_timesteps,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.ug import (
    UGContextStage,
    UGDecodeStage,
    UGGSegmentStage,
)
from sglang.srt.session.session_controller import SessionController
from sglang.srt.configs.model_config import is_multimodal_model
from sglang.srt.configs.neo_chat import NEOChatConfig, NEOVisionConfig
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.models.neo_chat import (
    NEOChatModel,
    NEOQwen3ForCausalLM,
    NEOQwen3Model,
    NEOVisionModel,
    build_abs_positions_from_grid_hw,
    build_u1_block_causal_allowed_mask,
    build_u1_vlm_input_info,
    build_u1_vlm_thw_indexes,
    map_u1_language_model_weight_name,
    _u1_decode_thw_positions_from_mm_inputs,
)
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.ug.adapter import UGModelRunnerAdapter
from sglang.srt.ug.context import UGContextBundle, UGContextHandle, UGSessionHandle
from sglang.srt.ug.runtime import (
    UGInterleavedMessage,
    UGSegmentState,
    UGSessionRuntime,
)
from sglang.srt.ug.u1 import (
    U1UGModelAdapter,
    U1VLMBackendResult,
    build_u1_vlm_prompt,
)


class TestU1UGBackendShell(unittest.TestCase):
    def test_neo_chat_config_wraps_qwen3_text_and_vision_configs(self):
        config = NEOChatConfig(
            vision_config={
                "architectures": ["NEOVisionModel"],
                "patch_size": 16,
                "hidden_size": 1024,
                "llm_hidden_size": 4096,
            },
            llm_config={
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 128,
                "rope_theta_hw": 10000.0,
            },
            downsample_ratio=0.5,
        )

        self.assertIsInstance(config.vision_config, NEOVisionConfig)
        self.assertEqual(config.model_type, "neo_chat")
        self.assertEqual(config.llm_config.architectures, ["Qwen3ForCausalLM"])
        self.assertEqual(config.llm_config.rope_theta_hw, 10000.0)
        self.assertIs(config.get_text_config(), config.llm_config)
        self.assertEqual(config.hidden_size, 32)

    def test_auto_config_can_build_neo_chat_without_remote_code(self):
        config = AutoConfig.for_model(
            "neo_chat",
            vision_config={"architectures": ["NEOVisionModel"]},
            llm_config={
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "vocab_size": 64,
            },
        )

        self.assertIsInstance(config, NEOChatConfig)
        self.assertEqual(config.llm_config.architectures, ["Qwen3ForCausalLM"])

    def test_neo_chat_is_registered_as_multimodal_srt_model(self):
        model_cls, arch = ModelRegistry.resolve_model_cls(["NEOChatModel"])

        self.assertIs(model_cls, NEOChatModel)
        self.assertEqual(arch, "NEOChatModel")
        self.assertTrue(is_multimodal_model(["NEOChatModel"]))

    def test_u1_abs_positions_match_row_major_patch_grid(self):
        abs_x, abs_y = build_abs_positions_from_grid_hw(
            torch.tensor([[2, 3], [1, 2]], dtype=torch.long)
        )

        self.assertEqual(abs_x.tolist(), [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(abs_y.tolist(), [0, 0, 0, 1, 1, 1, 0, 0])

    def test_u1_vlm_thw_indexes_match_official_context_semantics(self):
        input_ids = torch.tensor(
            [101, 151670, 151669, 151669, 151669, 151669, 151671, 102],
            dtype=torch.long,
        )

        indexes = build_u1_vlm_thw_indexes(
            input_ids,
            grid_hw=torch.tensor([[4, 4]], dtype=torch.long),
            downsample_ratio=0.5,
        )

        self.assertEqual(indexes.shape, (3, 8))
        self.assertEqual(indexes[0].tolist(), [0, 1, 2, 2, 2, 2, 3, 4])
        self.assertEqual(indexes[1].tolist(), [0, 0, 0, 0, 1, 1, 0, 0])
        self.assertEqual(indexes[2].tolist(), [0, 0, 0, 1, 0, 1, 0, 0])

    def test_u1_vlm_input_info_counts_context_tokens(self):
        info = build_u1_vlm_input_info(
            [151670, 151669, 151669, 151669, 151669, 151671],
            grid_hw=[[4, 4]],
        )

        self.assertEqual(info.image_context_token_count, 4)
        self.assertEqual(info.image_token_count, 5)
        self.assertEqual(tuple(info.thw_indexes.shape), (3, 6))

    def test_u1_native_vision_model_returns_dense_image_embeddings(self):
        config = NEOVisionConfig(
            patch_size=2,
            hidden_size=8,
            llm_hidden_size=16,
            downsample_ratio=0.5,
            max_position_embeddings_vision=16,
        )
        vision_model = NEOVisionModel(config)
        pixel_values = torch.arange(8 * 3 * 2 * 2, dtype=torch.float32).view(8, -1)
        grid_hw = torch.tensor([[2, 2], [2, 2]], dtype=torch.long)

        image_embeds = vision_model(pixel_values=pixel_values, grid_hw=grid_hw).last_hidden_state

        self.assertEqual(tuple(image_embeds.shape), (2, 16))

    def test_neo_chat_get_image_feature_uses_u1_vision_grid_metadata(self):
        model = _tiny_neo_chat_model_without_language_model()
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).view(4, -1),
            model_specific_data={"image_grid_hws": torch.tensor([[2, 2]])},
        )

        image_embeds = model.get_image_feature([item])

        self.assertEqual(tuple(image_embeds.shape), (1, 16))

    def test_neo_chat_pad_input_ids_replaces_img_context_tokens(self):
        mm_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            pad_value=999001,
            feature=torch.zeros(8, 12),
            model_specific_data={"image_grid_hws": torch.tensor([[2, 4]])},
            offsets=[(1, 2)],
        )
        mm_inputs = MultimodalInputs(mm_items=[mm_item])
        model = _tiny_neo_chat_model_without_language_model()

        padded = model.pad_input_ids([151670, 151669, 151669, 151671], mm_inputs)

        self.assertEqual(padded, [151670, 999001, 999001, 151671])
        self.assertEqual(mm_inputs.im_token_id, 151669)
        self.assertEqual(mm_inputs.mrope_positions[0].tolist(), [0, 1, 1, 2])
        self.assertEqual(mm_inputs.mrope_positions[1].tolist(), [0, 0, 0, 0])
        self.assertEqual(mm_inputs.mrope_positions[2].tolist(), [0, 0, 1, 0])

    def test_u1_decode_positions_continue_from_prompt_thw_index(self):
        mm_inputs = MultimodalInputs(mm_items=[])
        mm_inputs.mrope_positions = torch.tensor(
            [[0, 1, 1, 2], [0, 0, 0, 0], [0, 0, 1, 0]],
            dtype=torch.long,
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            mm_inputs=[mm_inputs],
            seq_lens_cpu=torch.tensor([6], dtype=torch.long),
        )

        positions = _u1_decode_thw_positions_from_mm_inputs(
            forward_batch=forward_batch,
            device=torch.device("cpu"),
        )

        self.assertEqual(positions.tolist(), [[4], [0], [0]])

    def test_u1_block_causal_mask_matches_same_t_bidirectional_rows(self):
        mask = build_u1_block_causal_allowed_mask(
            torch.tensor([0, 1, 2, 2, 2, 2, 3], dtype=torch.long)
        )

        self.assertTrue(mask[2, 5])
        self.assertTrue(mask[5, 2])
        self.assertFalse(mask[1, 2])
        self.assertTrue(mask[6, 5])

    def test_neo_chat_uses_u1_specific_qwen3_language_path(self):
        with patch("sglang.srt.models.neo_chat.NEOQwen3ForCausalLM") as language_cls:
            language_cls.return_value.model = object()
            model = NEOChatModel(_tiny_neo_chat_config())

        language_cls.assert_called_once()
        self.assertIs(model.language_model, language_cls.return_value)
        self.assertIs(model.model, language_cls.return_value.model)
        self.assertTrue(issubclass(NEOQwen3ForCausalLM, torch.nn.Module))
        self.assertTrue(issubclass(NEOQwen3Model, torch.nn.Module))

    def test_u1_language_weight_mapper_keeps_only_qwen3_u_path(self):
        self.assertEqual(
            map_u1_language_model_weight_name("language_model.model.layers.0.weight"),
            "model.layers.0.weight",
        )
        self.assertEqual(
            map_u1_language_model_weight_name("language_model.lm_head.weight"),
            "lm_head.weight",
        )
        self.assertIsNone(
            map_u1_language_model_weight_name(
                "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight"
            )
        )
        self.assertIsNone(map_u1_language_model_weight_name("vision_model.patch.weight"))
        self.assertIsNone(map_u1_language_model_weight_name("fm_modules.fm_head.weight"))

    def test_u1_adapter_declares_pixel_flow_kind(self):
        adapter = U1UGModelAdapter()

        self.assertEqual(adapter.g_kind, "pixel_flow")

    def test_u1_shell_does_not_expose_bagel_latent_api(self):
        adapter = U1UGModelAdapter()

        self.assertFalse(hasattr(adapter, "prepare_latents_from_session"))
        self.assertFalse(hasattr(adapter, "predict_velocity_from_session"))
        self.assertFalse(hasattr(adapter, "decode_latents_to_image"))

    def test_unwired_generation_methods_fail_clearly_until_wired(self):
        adapter = U1UGModelAdapter()

        with self.assertRaisesRegex(NotImplementedError, "SenseNova U1"):
            adapter.decode_vlm_text(runtime=None, session=None, max_new_tokens=1)

    def test_u1_vlm_backend_generates_real_text_through_adapter(self):
        adapter = U1UGModelAdapter(vlm_backend=FakeU1VLMBackend("real answer"))
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )
        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="image", content="/tmp/u1-input.png"),
                UGInterleavedMessage(type="text", content="what is in this image?"),
            ],
            session_id="u1-vlm-session",
        )

        result = adapter.decode_vlm_text(
            runtime=runtime,
            session=handle,
            max_new_tokens=4,
        )

        self.assertEqual(result.text, "real answer")
        self.assertEqual(adapter.vlm_backend.calls[0]["max_new_tokens"], 4)
        self.assertEqual(
            [message.type for message in adapter.vlm_backend.calls[0]["messages"]],
            ["image", "text"],
        )

    def test_u1_bridge_vlm_uses_real_backend_text(self):
        adapter = U1UGModelAdapter(vlm_backend=FakeU1VLMBackend("bridge answer"))
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )
        bridge = _load_u1_bridge(runtime)

        result = bridge.generate_vlm_text(
            messages=[
                UGInterleavedMessage(type="image", content="/tmp/u1-input.png"),
                UGInterleavedMessage(type="text", content="describe"),
            ],
            max_new_tokens=3,
        )

        self.assertEqual(result.text, "bridge answer")
        self.assertEqual(runtime.get_debug_counters(result.session)["prefill_count"], 1)
        bridge.runtime.close_session(result.session)

    def test_u1_vlm_without_external_backend_uses_srt_decode(self):
        adapter = U1UGModelAdapter()
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=FakeSRTDecodeExecutor([910, 911]),
            tokenizer=FakeTokenizer(),
        )
        bridge = _load_u1_bridge(runtime)

        result = bridge.generate_vlm_text(
            messages=[
                UGInterleavedMessage(type="image", content="/tmp/u1-input.png"),
                UGInterleavedMessage(type="text", content="describe"),
            ],
            max_new_tokens=2,
        )

        counters = runtime.get_debug_counters(result.session)
        self.assertEqual(result.text, "native:910 911")
        self.assertEqual(result.next_token_ids, (910, 911))
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["srt_u_decode_request_count"], 1)
        self.assertEqual(counters["srt_request_count"], 3)
        bridge.runtime.close_session(result.session)

    def test_u1_native_tokenizer_builds_single_real_vlm_prefill(self):
        adapter = U1UGModelAdapter(native_tokenizer=FakeU1Tokenizer())
        messages = [
            UGInterleavedMessage(
                type="image",
                content={
                    "pixel_values": torch.zeros(4, 12),
                    "grid_hw": torch.tensor([[4, 4]], dtype=torch.long),
                },
            ),
            UGInterleavedMessage(type="text", content="what is here?"),
        ]

        prepared = adapter.prepare_srt_u_interleaved_inputs(
            session=None,
            messages=messages,
            state=UGSegmentState.U_PREFILL,
        )

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].messages, messages)
        self.assertIsNotNone(prepared[0].mm_inputs)
        self.assertIn(151669, prepared[0].input_ids)
        self.assertEqual(prepared[0].mm_inputs.mm_items[0].offsets, [(1, 4)])
        self.assertEqual(
            prepared[0].adapter_metadata["u1"]["source"],
            "native_vlm_input",
        )

    def test_u1_vlm_prompt_matches_neo_chat_template(self):
        self.assertEqual(
            build_u1_vlm_prompt(question="What is in this image?"),
            "<|im_start|>user\n<image>\nWhat is in this image?"
            "<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_u1_bridge_native_vlm_prefill_uses_model_pad_once(self):
        executor = FakePadAndDecodeExecutor([910])
        adapter = U1UGModelAdapter(native_tokenizer=FakeU1Tokenizer())
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=executor,
            tokenizer=FakeTokenizer(),
        )
        bridge = _load_u1_bridge(runtime)

        result = bridge.generate_vlm_text(
            messages=[
                UGInterleavedMessage(
                    type="image",
                    content={
                        "pixel_values": torch.zeros(4, 12),
                        "grid_hw": torch.tensor([[4, 4]], dtype=torch.long),
                    },
                ),
                UGInterleavedMessage(type="text", content="what is here?"),
            ],
            max_new_tokens=1,
        )

        counters = runtime.get_debug_counters(result.session)
        self.assertEqual(result.text, "native:910")
        self.assertEqual(counters["srt_request_count"], 2)
        self.assertEqual(executor.pad_call_count, 1)
        self.assertEqual(len(executor.origin_input_ids_by_request), 2)
        prefill_ids = executor.origin_input_ids_by_request[0]
        self.assertIn(999001, prefill_ids)
        self.assertNotIn(151669, prefill_ids)
        bridge.runtime.close_session(result.session)

    def test_u1_text_prepared_input_carries_causal_rows(self):
        adapter = U1UGModelAdapter()

        prepared = adapter.prepare_srt_u_message_inputs(
            session=None,
            message=UGInterleavedMessage(type="text", content="hello U1"),
            state=UGSegmentState.U_PREFILL,
        )

        self.assertEqual(len(prepared), 1)
        metadata = prepared[0].adapter_metadata["u1"]
        self.assertEqual(metadata["segment_type"], "text")
        self.assertEqual(metadata["attention_rows"][0]["attention"], "causal")
        self.assertEqual(prepared[0].mot_text_token_indices, metadata["token_indices"])
        self.assertFalse(prepared[0].non_causal_query_attention)

    def test_u1_image_prepared_input_carries_hybrid_rows(self):
        adapter = U1UGModelAdapter()

        prepared = adapter.prepare_srt_u_message_inputs(
            session=None,
            message=UGInterleavedMessage(type="image", content=object()),
            state=UGSegmentState.U_PREFILL,
        )

        self.assertEqual(len(prepared), 1)
        metadata = prepared[0].adapter_metadata["u1"]
        self.assertEqual(metadata["segment_type"], "image")
        self.assertEqual(metadata["attention_rows"][0]["attention"], "hybrid")
        self.assertEqual(prepared[0].mot_image_token_indices, metadata["token_indices"])
        self.assertTrue(prepared[0].non_causal_query_attention)
        self.assertFalse(metadata["generated_image_commit"])

    def test_u1_generated_image_commit_uses_u_path_metadata(self):
        adapter = U1UGModelAdapter()

        prepared = adapter.prepare_srt_u_message_inputs(
            session=None,
            message=UGInterleavedMessage(type="image", content=object()),
            state=UGSegmentState.APPEND_IMAGE,
        )

        metadata = prepared[0].adapter_metadata["u1"]
        self.assertTrue(metadata["generated_image_commit"])
        self.assertEqual(metadata["source"], "generated_image")

    def test_u1_runtime_prefill_records_u_context_state_without_kv_details(self):
        adapter = U1UGModelAdapter()
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )

        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="text", content="describe this"),
                UGInterleavedMessage(type="image", content=object()),
            ],
            session_id="u1-session",
        )

        counters = runtime.get_debug_counters(handle)
        u1_state = counters["ug_model_state"]["u1"]
        self.assertEqual(handle.session_id, "u1-session")
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(u1_state["segments"][0]["segment_type"], "text")
        self.assertEqual(u1_state["segments"][1]["segment_type"], "image")
        self.assertEqual(
            u1_state["segments"][1]["attention_rows"][0]["attention"], "hybrid"
        )
        self.assertFalse(_has_kv_allocator_detail(u1_state))
        self.assertGreaterEqual(len(adapter.observed_u_forwards), 2)
        self.assertFalse(_has_kv_allocator_detail(adapter.observed_u_forwards[-1]))

    def test_u1_runtime_append_generated_image_records_commit_intent(self):
        adapter = U1UGModelAdapter()
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a cup")],
            session_id="u1-commit-session",
        )
        g_handle = runtime.begin_g_denoise(handle)

        updated = runtime.append_generated_image(g_handle, image=object())

        counters = runtime.get_debug_counters(updated)
        u1_state = counters["ug_model_state"]["u1"]
        self.assertEqual(updated.session_id, handle.session_id)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertTrue(u1_state["segments"][-1]["generated_image_commit"])
        self.assertEqual(u1_state["segments"][-1]["source"], "generated_image")
        self.assertFalse(_has_kv_allocator_detail(u1_state))

    def test_u1_g_executor_declares_pixel_flow_requirement(self):
        executor = U1PixelFlowGSegmentExecutor()

        self.assertEqual(executor.required_g_kind, "pixel_flow")

    def test_u1_patch_grid_ceil_aligns_to_patch_size(self):
        self.assertEqual(_u1_patch_grid(height=33, width=65, patch_size=16), (3, 5))

    def test_u1_timesteps_use_num_inference_steps(self):
        timesteps = _u1_timesteps(num_inference_steps=4, timestep_shift=2.0)

        self.assertEqual(len(timesteps), 4)
        self.assertGreater(timesteps[0], timesteps[-1])

    def test_u1_guidance_branch_selection(self):
        self.assertEqual(_u1_guidance_branch(_sampling()), "none")
        self.assertEqual(
            _u1_guidance_branch(_sampling(cfg_text_scale=2.0)),
            "text",
        )
        self.assertEqual(
            _u1_guidance_branch(_sampling(cfg_img_scale=2.0)),
            "image",
        )
        self.assertEqual(
            _u1_guidance_branch(_sampling(cfg_text_scale=2.0, cfg_img_scale=2.0)),
            "text_image",
        )

    def test_u1_g_executor_is_accepted_by_pixel_flow_bridge(self):
        bridge = PixelFlowBridge()
        server_args = _make_ug_server_args()

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base."
            "get_global_server_args",
            return_value=server_args,
        ):
            UGGSegmentStage(bridge, U1PixelFlowGSegmentExecutor())

    def test_u1_g_executor_rejects_latent_flow_bridge(self):
        bridge = LatentFlowBridge()
        server_args = _make_ug_server_args()

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base."
            "get_global_server_args",
            return_value=server_args,
        ):
            with self.assertRaisesRegex(ValueError, "pixel_flow"):
                UGGSegmentStage(bridge, U1PixelFlowGSegmentExecutor())

    def test_u1_g_executor_returns_image_and_metadata(self):
        executor = U1PixelFlowGSegmentExecutor()
        batch = Req(
            sampling_params=_sampling(
                height=8,
                width=8,
                num_inference_steps=3,
                cfg_text_scale=2.0,
            ),
            prompt="draw a cup",
            seed=7,
        )

        result = executor(
            bridge=PixelFlowBridge(),
            contexts=_make_contexts(),
            batch=batch,
            server_args=_make_ug_server_args(),
        )

        self.assertEqual(result.type, "image")
        self.assertEqual(result.image.size, (8, 8))
        self.assertEqual(result.metadata["g_kind"], "pixel_flow")
        self.assertEqual(result.metadata["grid"], (1, 1))
        self.assertEqual(result.metadata["timesteps"], 3)
        self.assertEqual(result.metadata["guidance"], "text")
        self.assertFalse(result.metadata["temporary_g_kv"])

    def test_u1_common_stages_run_text_image_text_light_smoke(self):
        bridge = _load_ug_bridge("sensenova/SenseNova-U1-8B-MoT")
        server_args = _make_ug_server_args()
        batch = Req(
            sampling_params=_sampling(
                height=8,
                width=8,
                num_inference_steps=2,
            ),
            prompt="draw a cup",
            seed=11,
            extra={"ug_request_metadata": {"mode": "interleave"}},
        )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base."
            "get_global_server_args",
            return_value=server_args,
        ):
            batch = UGContextStage(bridge).forward(batch, server_args)
            session = batch.extra["ug_contexts"].full.session
            session_id = session.session_id
            batch = UGGSegmentStage(bridge, U1PixelFlowGSegmentExecutor()).forward(
                batch,
                server_args,
            )
            self.assertEqual(
                bridge.runtime.get_debug_counters(session_id)["append_image_count"],
                0,
            )
            batch = UGDecodeStage(bridge).forward(batch, server_args)

        outputs = batch.extra["ug_output_segments"]
        self.assertEqual([segment["type"] for segment in outputs], ["image", "text"])
        self.assertIn("u1_pixel_flow", outputs[1]["text"])
        self.assertEqual(
            bridge.runtime.get_debug_counters(session_id)["append_image_count"],
            1,
        )
        bridge.release(batch.extra["ug_contexts"])

    def test_pipeline_dispatch_selects_u1_executor_for_pixel_flow(self):
        executor = _build_ug_g_segment_executor(PixelFlowBridge())

        self.assertIsInstance(executor, U1PixelFlowGSegmentExecutor)

    def test_pipeline_dispatch_selects_bagel_executor_for_latent_flow(self):
        executor = _build_ug_g_segment_executor(LatentFlowBridge())

        self.assertIsInstance(executor, BAGELLatentFlowGSegmentExecutor)

    def test_load_u1_bridge_shell_from_model_path(self):
        bridge = _load_ug_bridge("sensenova/SenseNova-U1-8B-MoT")

        self.assertEqual(bridge.g_kind, "pixel_flow")
        self.assertFalse(hasattr(bridge, "prepare_g_latents"))
        contexts = bridge.prepare_u_context(prompt="draw a cup", image=None)
        self.assertEqual(contexts.full.session.context_version, 1)
        bridge.release(contexts)

    def test_load_u1_bridge_accepts_scheduler_for_native_srt_decode(self):
        scheduler = FakeScheduler()

        bridge = _load_ug_bridge(
            "sensenova/SenseNova-U1-8B-MoT",
            scheduler=scheduler,
            srt_u_decode_max_new_tokens=2,
        )

        self.assertEqual(bridge.g_kind, "pixel_flow")
        self.assertIs(bridge.runtime.srt_request_executor.scheduler, scheduler)
        self.assertEqual(bridge.runtime.srt_u_decode_max_new_tokens, 2)


class PixelFlowBridge:
    g_kind = "pixel_flow"


class LatentFlowBridge:
    g_kind = "latent_flow"


class FakeTreeCache:
    def __init__(self):
        self.released_sessions = []

    def release_session(self, session_id):
        self.released_sessions.append(session_id)


class FakeSRTDecodeExecutor:
    finish_request_after_execute = True

    def __init__(self, output_ids):
        self.output_ids = list(output_ids)
        self.requests = []

    def execute_ug_request(self, *, record, req, state):
        del record, state
        self.requests.append(req.rid)
        if req.sampling_params.max_new_tokens > 0:
            req.output_ids = self.output_ids[: req.sampling_params.max_new_tokens]


class FakePadAndDecodeExecutor(FakeSRTDecodeExecutor):
    def __init__(self, output_ids):
        super().__init__(output_ids)
        self.pad_call_count = 0
        self.origin_input_ids_by_request = []

    def pad_input_ids(self, input_ids, mm_inputs):
        self.pad_call_count += 1
        del mm_inputs
        return [999001 if token_id == 151669 else token_id for token_id in input_ids]

    def execute_ug_request(self, *, record, req, state):
        self.origin_input_ids_by_request.append(list(req.origin_input_ids))
        super().execute_ug_request(record=record, req=req, state=state)


class FakeTokenizer:
    bos_token_id = 1

    def decode(self, token_ids):
        return "native:" + " ".join(str(token_id) for token_id in token_ids)


class FakeU1Tokenizer:
    def convert_tokens_to_ids(self, token):
        return {
            "<IMG_CONTEXT>": 151669,
            "<img>": 151670,
            "</img>": 151671,
        }[token]

    def __call__(self, text, return_tensors=None):
        del return_tensors
        ids = []
        i = 0
        while i < len(text):
            if text.startswith("<IMG_CONTEXT>", i):
                ids.append(151669)
                i += len("<IMG_CONTEXT>")
            elif text.startswith("<img>", i):
                ids.append(151670)
                i += len("<img>")
            elif text.startswith("</img>", i):
                ids.append(151671)
                i += len("</img>")
            else:
                i += 1
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class FakeScheduler:
    def __init__(self):
        self.session_controller = SessionController(FakeTreeCache())
        self.tokenizer = FakeTokenizer()
        self.model_config = FakeModelConfig()
        self.model_worker = type(
            "Worker",
            (),
            {"model_runner": type("ModelRunner", (), {})()},
        )()


class FakeModelConfig:
    vocab_size = 32000


class FakeU1VLMBackend:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_text(self, *, messages, max_new_tokens):
        self.calls.append(
            {
                "messages": list(messages),
                "max_new_tokens": max_new_tokens,
            }
        )
        return U1VLMBackendResult(text=self.text)


def _load_u1_bridge(runtime):
    from sglang.srt.ug.u1 import U1SRTBackedUGMiddleBridge

    return U1SRTBackedUGMiddleBridge(runtime)


def _make_contexts():
    session = UGSessionHandle(
        session_id="u1-test-session",
        anchor_request_id="u1-test-session:0",
        context_length=1,
        context_version=1,
    )
    return UGContextBundle(
        full=UGContextHandle("u1-test-session:0", 1, session=session),
        text_cfg=UGContextHandle("u1-test-session:text", 0, session=session),
        image_cfg=UGContextHandle("u1-test-session:image", 0, session=session),
    )


def _make_ug_server_args():
    return FakeServerArgs(
        pipeline_config=UGPipelineConfig(default_height=4, default_width=4),
        num_gpus=1,
        enable_cfg_parallel=False,
        disagg_mode=False,
        comfyui_mode=True,
    )


def _sampling(**kwargs):
    values = {
        "height": 8,
        "width": 8,
        "num_inference_steps": 2,
        "cfg_text_scale": 1.0,
        "cfg_img_scale": 1.0,
    }
    values.update(kwargs)
    return UGSamplingParams(**values)


def _tiny_neo_chat_config():
    return NEOChatConfig(
        vision_config={
            "architectures": ["NEOVisionModel"],
            "patch_size": 2,
            "hidden_size": 8,
            "llm_hidden_size": 16,
            "downsample_ratio": 0.5,
            "max_position_embeddings_vision": 16,
        },
        llm_config={
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "vocab_size": 128,
            "head_dim": 8,
        },
        downsample_ratio=0.5,
    )


def _tiny_neo_chat_model_without_language_model():
    model = NEOChatModel.__new__(NEOChatModel)
    torch.nn.Module.__init__(model)
    model.config = _tiny_neo_chat_config()
    model.img_context_token_id = model.config.img_context_token_id
    model.img_start_token_id = model.config.img_start_token_id
    model.img_end_token_id = model.config.img_end_token_id
    model.downsample_ratio = model.config.downsample_ratio
    model.vision_model = NEOVisionModel(model.config.vision_config)
    return model


class FakeServerArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _has_kv_allocator_detail(value):
    forbidden = ("allocator", "page", "slot")
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if "kv" in key_text or any(word in key_text for word in forbidden):
                return True
            if _has_kv_allocator_detail(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_kv_allocator_detail(item) for item in value)
    return False


if __name__ == "__main__":
    unittest.main()
