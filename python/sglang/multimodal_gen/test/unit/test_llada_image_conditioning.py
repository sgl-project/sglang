# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.conditioning import (
    LLaDAImageTextConditioningStage,
    LLaDAImageTextEncoderRunner,
)
from sglang.srt.runtime_context import ParallelContext, RuntimeContext

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
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.conditioning import (
            ensure_conditioning_mask_active,
        )

        ensure_conditioning_mask_active(SimpleNamespace(conditioning_mask_active=True))
        with self.assertRaisesRegex(RuntimeError, "without its block attention mask"):
            ensure_conditioning_mask_active(
                SimpleNamespace(conditioning_mask_active=False)
            )

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

    def test_text_runner_scopes_singleton_attention_group_and_restores(self):
        import sglang.multimodal_gen.runtime.distributed.parallel_state as mm_parallel_state
        import sglang.srt.distributed.parallel_state as srt_parallel_state

        diffusion_group = object()
        encoder_group = SimpleNamespace(world_size=1, rank_in_group=0)
        encoder_attention_group = object()
        runner = object.__new__(LLaDAImageTextEncoderRunner)
        runner.runtime_context = RuntimeContext(ParallelContext())
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
            patch(
                "sglang.srt.runtime_context.create_context",
                side_effect=lambda *_args, **_kwargs: RuntimeContext(ParallelContext()),
            ),
            patch(
                "sglang.srt.server_args.ServerArgs",
                side_effect=lambda **kwargs: SimpleNamespace(page_size=1, **kwargs),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.conditioning.get_local_torch_device",
                return_value=torch.device("cpu"),
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


class TestLLaDAImageRuntimeContext(unittest.TestCase):
    def test_encoder_preserves_diffusion_runtime(self):
        """The embedded encoder retains its context inside the diffusion TP scope."""
        import sglang.multimodal_gen.runtime.distributed.parallel_state as mm_state
        import sglang.srt.distributed.parallel_state as srt_state
        from sglang.srt import runtime_context as rc
        from sglang.srt.server_args import ServerArgs as SRTServerArgs

        saved = rc.snapshot_context()
        self.addCleanup(rc.restore_context, saved)
        conditioning = LLaDAImageTextEncoderRunner.__module__
        for fail_init in (False, True):
            with self.subTest(fail_init=fail_init):
                rc.reset_context()
                diffusion_args = SRTServerArgs(model_path="dummy", tp_size=1)
                diffusion_context = rc.publish(
                    diffusion_args, role="diffusion_gpu_worker"
                )
                diffusion_context.override("diffusion.setup", grammar_backend="none")
                diffusion_bags = diffusion_context._config_bags
                diffusion_log = diffusion_context.overrides_log()
                diffusion_buffer = rc.get_buffer("conditioning", object)
                diffusion_group = SimpleNamespace(world_size=1, rank_in_group=0)
                encoder_group = SimpleNamespace(world_size=1, rank_in_group=0)
                encoder_attention_group = SimpleNamespace(world_size=1, rank_in_group=0)
                observed = {}

                def make_args(**kwargs):
                    kwargs["model_path"] = "dummy"
                    return SRTServerArgs(**kwargs)

                def check_encoder_context():
                    context = rc.assert_published(observed["args"], role="scheduler")
                    self.assertEqual(rc.get_parallel().tp_size, 1)
                    self.assertEqual(rc.get_parallel().attn_tp_size, 1)
                    self.assertEqual(rc.get_parallel().attn_dp_size, 1)
                    self.assertEqual(rc.get_parallel().moe_tp_size, 1)
                    self.assertEqual(rc.get_schedule().max_running_requests, 2)
                    return context

                def allocate():
                    context = check_encoder_context()
                    context.override("encoder.setup", page_size=16)
                    rc.get_buffer("conditioning", object)
                    if fail_init:
                        raise RuntimeError("allocation failed")

                def make_worker(**kwargs):
                    observed["args"] = kwargs["server_args"]
                    observed["context"] = check_encoder_context()
                    srt_state._TP = encoder_group
                    srt_state._ATTN_TP = encoder_attention_group
                    return SimpleNamespace(
                        model_runner=SimpleNamespace(page_size=16),
                        get_memory_pool=lambda: (object(), object()),
                        alloc_memory_pool=allocate,
                        init_attention_backends=check_encoder_context,
                        init_cuda_graphs=check_encoder_context,
                    )

                def make_cache(_params):
                    self.assertEqual(rc.get_schedule().page_size, 16)
                    return object()

                def check_diffusion_context():
                    self.assertIs(rc.get_context(), diffusion_context)
                    self.assertIs(rc.get_server_args(), diffusion_args)
                    self.assertEqual(rc.publish_role(), "diffusion_gpu_worker")
                    self.assertIs(rc.get_context()._config_bags, diffusion_bags)
                    self.assertEqual(rc.get_context().overrides_log(), diffusion_log)
                    self.assertEqual(rc.get_exec().kernel.grammar_backend, "none")
                    self.assertEqual(rc.get_parallel().tp_size, 1)
                    self.assertEqual(rc.get_parallel().attn_tp_size, 1)
                    self.assertEqual(rc.get_parallel().attn_dp_size, 1)
                    self.assertEqual(rc.get_parallel().moe_tp_size, 1)
                    self.assertIs(
                        rc.get_buffer("conditioning", object), diffusion_buffer
                    )
                    self.assertIs(mm_state._TP, diffusion_group)
                    self.assertIs(srt_state._TP, diffusion_group)
                    self.assertIs(srt_state._ATTN_TP, diffusion_group)

                with (
                    patch("sglang.srt.server_args.ServerArgs", side_effect=make_args),
                    patch(
                        "sglang.srt.managers.tp_worker.TpModelWorker",
                        side_effect=make_worker,
                    ),
                    patch(
                        "sglang.srt.mem_cache.chunk_cache.ChunkCache",
                        side_effect=make_cache,
                    ),
                    patch(
                        f"{conditioning}.get_local_torch_device",
                        return_value=torch.device("cpu"),
                    ),
                    patch.object(mm_state, "_TP", diffusion_group),
                    patch.object(srt_state, "_TP", diffusion_group),
                    patch.object(srt_state, "_ATTN_TP", diffusion_group),
                ):
                    kwargs = dict(
                        model_root="/unused/model",
                        queryformer=object(),
                        text_projection=object(),
                        tokenizer=object(),
                        server_args=SimpleNamespace(
                            sp_degree=1,
                            nccl_port=29500,
                            trust_remote_code=False,
                            revision=None,
                            component_paths={},
                            pipeline_config=SimpleNamespace(
                                text_encoder_mem_fraction_static=0.1
                            ),
                        ),
                    )
                    if fail_init:
                        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                            LLaDAImageTextEncoderRunner(**kwargs)
                        check_diffusion_context()
                        continue
                    runner = LLaDAImageTextEncoderRunner(**kwargs)
                    check_diffusion_context()
                    self.assertIsNot(observed["context"], diffusion_context)

                    def encode(*_args, **_kwargs):
                        self.assertIs(check_encoder_context(), observed["context"])
                        self.assertEqual(rc.get_schedule().page_size, 16)
                        self.assertEqual(len(rc.get_context().overrides_log()), 1)
                        self.assertIsNot(
                            rc.get_buffer("conditioning", object), diffusion_buffer
                        )
                        self.assertIs(srt_state._ATTN_TP, encoder_attention_group)
                        if observed["fail_encode"]:
                            raise RuntimeError("encode failed")
                        return ["encoded"]

                    runner._encode_impl = encode
                    for fail_encode in (False, True, False):
                        observed["fail_encode"] = fail_encode
                        if fail_encode:
                            with self.assertRaisesRegex(RuntimeError, "encode failed"):
                                runner.encode(["hello"], max_sequence_length=16)
                        else:
                            self.assertEqual(
                                runner.encode(["hello"], max_sequence_length=16),
                                ["encoded"],
                            )
                        check_diffusion_context()


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
