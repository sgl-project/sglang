# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning import (
    LLaDAImageTextConditioningStage,
    LLaDAImageTextEncoderRunner,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class _FakeTextRunner:
    def __init__(self):
        self.prompts = None
        self.component_names = []

    def encode(self, prompts, max_sequence_length, component_context=None):
        self.prompts = (prompts, max_sequence_length)
        if component_context is not None:
            for component_name in ("queryformer", "text_projection"):
                with component_context(component_name=component_name, module=object()):
                    self.component_names.append(component_name)
        return [
            torch.full((index + 2, 4), float(index + 1))
            for index in range(len(prompts))
        ]


class TestLLaDAImageTextConditioning(unittest.TestCase):
    def setUp(self):
        self.runner = _FakeTextRunner()
        with patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()):
            self.stage = LLaDAImageTextConditioningStage(self.runner)

    def test_repeats_positive_and_negative_text_for_each_output(self):
        batch = SimpleNamespace(
            prompt="a red car",
            negative_prompt=None,
            guidance_scale=5.0,
            num_outputs_per_prompt=2,
            max_sequence_length=128,
        )

        result = self.stage.forward(batch, server_args=SimpleNamespace())

        self.assertTrue(result.do_classifier_free_guidance)
        self.assertEqual(len(result.prompt_embeds), 2)
        self.assertEqual(len(result.negative_prompt_embeds), 2)
        self.assertEqual(len(result.prompt_attention_mask), 2)
        self.assertEqual(len(result.negative_attention_mask), 2)
        self.assertTrue(torch.equal(result.prompt_embeds[0], result.prompt_embeds[1]))
        self.assertTrue(
            torch.equal(
                result.negative_prompt_embeds[0], result.negative_prompt_embeds[1]
            )
        )

    def test_guidance_disabled_has_no_negative_condition(self):
        batch = SimpleNamespace(
            prompt="a red car",
            negative_prompt=None,
            guidance_scale=1.0,
            num_outputs_per_prompt=2,
            max_sequence_length=128,
        )

        result = self.stage.forward(batch, server_args=SimpleNamespace())

        self.assertFalse(result.do_classifier_free_guidance)
        self.assertEqual(len(result.prompt_embeds), 2)
        self.assertEqual(result.negative_prompt_embeds, [])
        self.assertEqual(result.negative_attention_mask, [])

    def test_conditioning_mask_guard_fails_closed(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning import (
            ensure_conditioning_mask_active,
        )

        ensure_conditioning_mask_active(SimpleNamespace(conditioning_mask_active=True))
        with self.assertRaisesRegex(RuntimeError, "without its block attention mask"):
            ensure_conditioning_mask_active(
                SimpleNamespace(conditioning_mask_active=False)
            )

    def test_runtime_lifecycle_apis_reject_partial_coverage(self):
        from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
        from sglang.multimodal_gen.runtime.managers.memory_managers.memory_occupation_controller import (
            MemoryOccupationController,
        )
        from sglang.multimodal_gen.runtime.post_training.weights_updater import (
            WeightsUpdater,
        )

        self.assertTrue(PipelineConfig().supports_memory_release())
        self.assertTrue(PipelineConfig().supports_hot_weight_updates())
        config = LLaDAImagePipelineConfig()
        self.assertFalse(config.supports_memory_release())
        self.assertFalse(config.supports_hot_weight_updates())

        pipeline = SimpleNamespace(server_args=SimpleNamespace(pipeline_config=config))
        controller = MemoryOccupationController(
            pipeline=pipeline, rank=0, use_fsdp_inference=False
        )
        result = controller.release_memory_occupation()
        self.assertFalse(result["success"])
        self.assertIn("does not support memory release", result["message"])
        self.assertFalse(controller.is_sleeping())

        updater = object.__new__(WeightsUpdater)
        updater.pipeline = pipeline
        ok, message = updater.update_weights_from_disk("/unused/path")
        self.assertFalse(ok)
        self.assertIn("old checkpoint", message)
        ok, message = updater.update_weights_from_tensor(named_tensors=[])
        self.assertFalse(ok)
        self.assertIn("old checkpoint", message)

    def test_forward_batch_declares_conditioning_text_lens_field(self):
        import dataclasses

        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        field_names = {field.name for field in dataclasses.fields(ForwardBatch)}
        # Batch rebuilds use dataclasses.replace, which drops dynamic attrs.
        self.assertIn("llada_image_conditioning_text_lens_cpu", field_names)

    def test_stage_declares_auxiliary_component_residency(self):
        uses = self.stage.component_uses(SimpleNamespace(), "conditioning")

        self.assertEqual(
            [use.component_name for use in uses],
            ["queryformer", "text_projection"],
        )

        batch = SimpleNamespace(
            prompt="a red car",
            negative_prompt=None,
            guidance_scale=1.0,
            num_outputs_per_prompt=1,
            max_sequence_length=128,
        )
        self.stage.forward(batch, server_args=SimpleNamespace())

        self.assertEqual(
            self.runner.component_names, ["queryformer", "text_projection"]
        )

    def test_text_runner_uses_sp_group_as_text_encoder_tp_group(self):
        resolved_page_size = 16
        fake_worker = SimpleNamespace(
            model_runner=SimpleNamespace(page_size=resolved_page_size),
            model_config=object(),
            get_memory_pool=lambda: (object(), object()),
            alloc_memory_pool=lambda: None,
            init_attention_backends=lambda: None,
            init_cuda_graphs=lambda: None,
        )
        srt_args_module = "sglang.srt.server_args.ServerArgs"
        worker_module = "sglang.srt.managers.tp_worker.TpModelWorker"
        with (
            patch(
                worker_module,
                return_value=fake_worker,
            ) as worker_cls,
            patch("sglang.srt.runtime_context.publish"),
            patch(
                "sglang.srt.mem_cache.cache_init_params.CacheInitParams",
                return_value=object(),
            ) as cache_init_params_cls,
            patch(
                "sglang.srt.mem_cache.chunk_cache.ChunkCache",
                return_value=object(),
            ),
            patch(
                srt_args_module,
                side_effect=lambda **kwargs: SimpleNamespace(page_size=None, **kwargs),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning.get_local_torch_device",
                return_value=torch.device("cpu"),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning.get_sp_parallel_rank",
                return_value=1,
            ),
        ):
            runner = LLaDAImageTextEncoderRunner(
                model_root="/unused/model",
                queryformer=object(),
                text_projection=object(),
                tokenizer=object(),
                server_args=SimpleNamespace(
                    sp_degree=2,
                    nccl_port=29500,
                    trust_remote_code=True,
                    revision=None,
                    pipeline_config=SimpleNamespace(
                        text_encoder_mem_fraction_static=0.1
                    ),
                ),
            )
            untrusted_runner = LLaDAImageTextEncoderRunner(
                model_root="/unused/model",
                queryformer=object(),
                text_projection=object(),
                tokenizer=object(),
                server_args=SimpleNamespace(
                    sp_degree=2,
                    nccl_port=29500,
                    trust_remote_code=False,
                    revision="pinned-rev",
                    pipeline_config=SimpleNamespace(
                        text_encoder_mem_fraction_static=0.1
                    ),
                ),
            )

        self.assertIs(runner.worker, fake_worker)
        self.assertIsNone(runner.server_args.page_size)
        self.assertEqual(runner.page_size, resolved_page_size)
        self.assertEqual(
            [call.kwargs["page_size"] for call in cache_init_params_cls.call_args_list],
            [resolved_page_size, resolved_page_size],
        )
        self.assertEqual(runner.server_args.tp_size, 2)
        self.assertEqual(runner.server_args.dp_size, 2)
        self.assertTrue(runner.server_args.enable_dp_attention)
        self.assertTrue(runner.server_args.enable_dp_lm_head)
        self.assertEqual(runner.server_args.attn_cp_size, 1)
        self.assertEqual(runner.server_args.ep_size, 1)
        self.assertEqual(runner.server_args.moe_dp_size, 2)
        self.assertEqual(runner.server_args.moe_dense_tp_size, 1)
        self.assertEqual(runner.server_args.moe_a2a_backend, "none")
        self.assertEqual(runner.server_args.max_running_requests, 4)
        parallel_state = worker_cls.call_args_list[0].kwargs["ps"]
        self.assertEqual(parallel_state.tp_rank, 1)
        self.assertEqual(parallel_state.tp_size, 2)
        self.assertEqual(parallel_state.dp_rank, 1)
        self.assertEqual(parallel_state.dp_size, 2)
        self.assertEqual(parallel_state.attn_tp_rank, 0)
        self.assertEqual(parallel_state.attn_tp_size, 1)
        self.assertEqual(parallel_state.attn_cp_rank, 0)
        self.assertEqual(parallel_state.attn_cp_size, 1)
        self.assertEqual(parallel_state.attn_dp_rank, 1)
        self.assertEqual(parallel_state.attn_dp_size, 2)
        self.assertEqual(parallel_state.moe_ep_rank, 0)
        self.assertEqual(parallel_state.moe_ep_size, 1)
        self.assertEqual(parallel_state.moe_dp_rank, 1)
        self.assertEqual(parallel_state.moe_dp_size, 2)
        self.assertIs(runner.server_args.trust_remote_code, True)
        self.assertIs(untrusted_runner.server_args.trust_remote_code, False)
        self.assertEqual(untrusted_runner.server_args.revision, "pinned-rev")

    def test_text_runner_scopes_singleton_attention_group_and_restores(self):
        import sglang.multimodal_gen.runtime.distributed.parallel_state as mm_parallel_state
        import sglang.srt.distributed.parallel_state as srt_parallel_state

        diffusion_group = object()
        encoder_group = SimpleNamespace(world_size=2, rank_in_group=1)
        encoder_attention_group = object()
        runner = object.__new__(LLaDAImageTextEncoderRunner)
        runner.encoder_tp_group = encoder_group
        runner.encoder_attn_tp_group = encoder_attention_group
        observed = {}

        def fail_encode(*_args, **_kwargs):
            observed["tp"] = srt_parallel_state._TP
            observed["attn_tp"] = srt_parallel_state._ATTN_TP
            raise RuntimeError("encode failed")

        runner._encode_impl = fail_encode
        with (
            patch.object(mm_parallel_state, "_TP", diffusion_group),
            patch.object(srt_parallel_state, "_TP", diffusion_group),
            patch.object(srt_parallel_state, "_ATTN_TP", diffusion_group),
        ):
            with self.assertRaisesRegex(RuntimeError, "encode failed"):
                runner.encode(["hello"], max_sequence_length=16)

            self.assertIs(mm_parallel_state._TP, diffusion_group)
            self.assertIs(srt_parallel_state._TP, diffusion_group)
            self.assertIs(srt_parallel_state._ATTN_TP, diffusion_group)

        self.assertIs(observed["tp"], encoder_group)
        self.assertIs(observed["attn_tp"], encoder_attention_group)

    def test_text_runner_uses_resolved_page_size_for_prefill(self):
        class StopAfterPrefillAdder(Exception):
            pass

        runner = object.__new__(LLaDAImageTextEncoderRunner)
        runner.tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=[[1, 2]])
        runner.queryformer = SimpleNamespace(config=SimpleNamespace(num_queries=2))
        runner.worker = SimpleNamespace(model_config=SimpleNamespace(vocab_size=128))
        runner.server_args = SimpleNamespace(
            page_size=None,
            chunked_prefill_size=-1,
            max_prefill_tokens=8192,
        )
        runner.page_size = 64
        runner.tree_cache = object()
        runner.token_to_kv_pool_allocator = object()
        with (
            patch(
                "sglang.srt.sampling.sampling_params.SamplingParams",
                return_value=SimpleNamespace(normalize=lambda _: None),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.Req",
                return_value=SimpleNamespace(),
            ),
            patch(
                "sglang.srt.managers.schedule_policy.PrefillAdder",
                side_effect=StopAfterPrefillAdder,
            ) as prefill_adder_cls,
        ):
            with self.assertRaises(StopAfterPrefillAdder):
                runner._encode_impl(["hello"], max_sequence_length=16)

        self.assertEqual(prefill_adder_cls.call_args.args[0], runner.page_size)

    def test_text_dp_metadata_covers_each_replica(self):
        from sglang.srt.managers.schedule_policy import AddReqResult

        class StopAfterMetadata(Exception):
            pass

        class FakeReq:
            def __init__(self, *_args, **_kwargs):
                self.kv = None
                self.req_pool_idx = None

            def init_next_round_input(self, _tree_cache):
                return None

        class FakePrefillAdder:
            def __init__(self, *_args, **_kwargs):
                self.can_run_list = []

            def add_one_req(self, req, **_kwargs):
                self.can_run_list.append(req)
                return AddReqResult.CONTINUE

        def stop_after_metadata():
            raise StopAfterMetadata

        batch = SimpleNamespace(prepare_for_extend=stop_after_metadata)
        runner = object.__new__(LLaDAImageTextEncoderRunner)
        runner.tokenizer = lambda *_args, **_kwargs: SimpleNamespace(
            input_ids=[[1, 2], [3]]
        )
        runner.queryformer = SimpleNamespace(config=SimpleNamespace(num_queries=2))
        runner.worker = SimpleNamespace(model_config=SimpleNamespace(vocab_size=128))
        runner.server_args = SimpleNamespace(
            chunked_prefill_size=-1,
            max_prefill_tokens=8192,
            dp_size=2,
        )
        runner.text_dp_attention = True
        runner.page_size = 1
        runner.tree_cache = object()
        runner.token_to_kv_pool_allocator = object()
        runner.req_to_token_pool = object()

        with (
            patch(
                "sglang.srt.sampling.sampling_params.SamplingParams",
                return_value=SimpleNamespace(normalize=lambda _config: None),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.Req",
                side_effect=FakeReq,
            ),
            patch(
                "sglang.srt.managers.schedule_policy.PrefillAdder",
                side_effect=FakePrefillAdder,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.ScheduleBatch.init_new",
                return_value=batch,
            ),
        ):
            with self.assertRaises(StopAfterMetadata):
                runner._encode_impl(
                    ["positive", "negative"],
                    max_sequence_length=16,
                )

        self.assertEqual(batch.global_num_tokens, [7, 7])
        self.assertEqual(batch.global_num_tokens_for_logprob, [2, 2])

    def test_text_runner_restores_diffusion_groups_when_worker_init_fails(self):
        import sglang.srt.distributed.parallel_state as srt_parallel_state

        diffusion_group = object()
        encoder_group = object()

        def fail_after_installing_encoder_groups(**_kwargs):
            srt_parallel_state._TP = encoder_group
            srt_parallel_state._ATTN_TP = encoder_group
            return SimpleNamespace(
                alloc_memory_pool=lambda: (_ for _ in ()).throw(
                    RuntimeError("allocation failed")
                )
            )

        with (
            patch(
                "sglang.srt.managers.tp_worker.TpModelWorker",
                side_effect=fail_after_installing_encoder_groups,
            ),
            patch("sglang.srt.runtime_context.publish"),
            patch(
                "sglang.srt.server_args.ServerArgs",
                side_effect=lambda **kwargs: SimpleNamespace(page_size=1, **kwargs),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning.get_local_torch_device",
                return_value=torch.device("cpu"),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning.get_sp_parallel_rank",
                return_value=0,
            ),
            patch.object(srt_parallel_state, "_TP", diffusion_group),
            patch.object(srt_parallel_state, "_ATTN_TP", diffusion_group),
        ):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                LLaDAImageTextEncoderRunner(
                    model_root="/unused/model",
                    queryformer=object(),
                    text_projection=object(),
                    tokenizer=object(),
                    server_args=SimpleNamespace(
                        sp_degree=1,
                        nccl_port=29500,
                        trust_remote_code=True,
                        revision=None,
                        pipeline_config=SimpleNamespace(
                            text_encoder_mem_fraction_static=0.1
                        ),
                    ),
                )

            self.assertIs(srt_parallel_state._TP, diffusion_group)
            self.assertIs(srt_parallel_state._ATTN_TP, diffusion_group)


class TestLLaDAImageConditionKwargs(unittest.TestCase):
    def setUp(self):
        self.config = LLaDAImagePipelineConfig()
        self.semantic = [
            torch.full((3, 5), 1.0),
            torch.full((3, 5), 2.0),
        ]
        self.source = [
            torch.full((8, 1, 2, 3), 3.0),
            torch.full((8, 1, 2, 3), 4.0),
        ]
        self.batch = SimpleNamespace(
            batch_size=2,
            image_embeds=self.semantic,
            source_latents=self.source,
            do_classifier_free_guidance=True,
        )

    def test_cfg_uses_semantics_only_on_positive_branch(self):
        positive = self.config.prepare_pos_cond_kwargs(
            self.batch, torch.device("cpu"), rotary_emb=None, dtype=torch.float64
        )
        negative = self.config.prepare_neg_cond_kwargs(
            self.batch, torch.device("cpu"), rotary_emb=None, dtype=torch.float64
        )
        policy = CFGPolicy().build(
            self.batch,
            {"encoder_hidden_states_image": self.semantic},
            positive,
            negative,
        )

        positive_kwargs = policy.branches[0].kwargs
        negative_kwargs = policy.branches[1].kwargs
        self.assertEqual(
            [tuple(x.shape) for x in positive_kwargs["encoder_hidden_states_image"]],
            [(3, 5)] * 2,
        )
        self.assertEqual(
            [tuple(x.shape) for x in negative_kwargs["encoder_hidden_states_image"]],
            [(0, 5)] * 2,
        )
        self.assertTrue(
            all(x.dtype == torch.float64 for x in positive_kwargs["source_latents"])
        )
        for positive_source, negative_source in zip(
            positive_kwargs["source_latents"],
            negative_kwargs["source_latents"],
            strict=True,
        ):
            torch.testing.assert_close(positive_source, negative_source)

    def test_rejects_condition_batch_length_mismatch(self):
        self.batch.image_embeds = self.semantic[:1]

        with self.assertRaisesRegex(
            ValueError, "image_embeds has 1 entries, expected 2"
        ):
            self.config.prepare_pos_cond_kwargs(
                self.batch,
                torch.device("cpu"),
                rotary_emb=None,
                dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
