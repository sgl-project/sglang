# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.conditioning import (
    LLaDAImageTextConditioningStage,
    LLaDAImageTextEncoderRunner,
    ensure_conditioning_mask_active,
)


@pytest.mark.parametrize("guidance", [1.0, 5.0])
def test_text_conditioning_repeats_outputs_and_masks(guidance):
    outputs = [torch.ones(2, 4), torch.full((3, 4), 2.0)]
    runner = SimpleNamespace(encode=Mock(return_value=outputs))
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
        return_value=SimpleNamespace(),
    ):
        stage = LLaDAImageTextConditioningStage(runner)
    batch = SimpleNamespace(
        prompt="a red car",
        negative_prompt=None,
        guidance_scale=guidance,
        num_outputs_per_prompt=2,
        max_sequence_length=128,
    )
    stage.forward(batch, SimpleNamespace())
    assert batch.do_classifier_free_guidance == (guidance > 1)
    assert len(runner.encode.call_args.args[0]) == (2 if guidance > 1 else 1)
    for prefix, output in zip(("prompt", "negative"), outputs, strict=True):
        embeddings = getattr(
            batch, "prompt_embeds" if prefix == "prompt" else "negative_prompt_embeds"
        )
        masks = getattr(batch, f"{prefix}_attention_mask")
        if prefix == "negative" and guidance == 1:
            assert embeddings == masks == []
            continue
        assert len(embeddings) == len(masks) == 2
        for embedding, mask in zip(embeddings, masks, strict=True):
            torch.testing.assert_close(embedding, output)
            torch.testing.assert_close(
                mask, torch.ones(output.shape[0], dtype=torch.bool)
            )


def test_conditioning_mask_guard():
    ensure_conditioning_mask_active(SimpleNamespace(conditioning_mask_active=True))
    with pytest.raises(RuntimeError, match="without its block attention mask"):
        ensure_conditioning_mask_active(SimpleNamespace(conditioning_mask_active=False))


def test_prefill_uses_resolved_page_size():
    runner = object.__new__(LLaDAImageTextEncoderRunner)
    runner.tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=[[1, 2]])
    runner.queryformer = SimpleNamespace(config=SimpleNamespace(num_queries=2))
    runner.worker = SimpleNamespace(model_config=SimpleNamespace(vocab_size=128))
    runner.server_args = SimpleNamespace(
        page_size=None, chunked_prefill_size=-1, max_prefill_tokens=8192
    )
    runner.page_size = 64
    runner.tree_cache = runner.token_to_kv_pool_allocator = object()
    with (
        patch(
            "sglang.srt.sampling.sampling_params.SamplingParams",
            return_value=SimpleNamespace(normalize=lambda _: None),
        ),
        patch("sglang.srt.managers.schedule_batch.Req", return_value=SimpleNamespace()),
        patch(
            "sglang.srt.managers.schedule_policy.PrefillAdder",
            side_effect=RuntimeError("stop at prefill"),
        ) as adder,
        pytest.raises(RuntimeError, match="stop at prefill"),
    ):
        runner._encode_impl(["hello"], max_sequence_length=16)
    assert adder.call_args.args[0] == 64


@pytest.mark.parametrize("fail_init", [False, True])
def test_encoder_context_restores_after_init_and_encode(fail_init):
    import sglang.multimodal_gen.runtime.distributed.parallel_state as mm_state
    import sglang.srt.distributed.parallel_state as srt_state
    from sglang.srt import runtime_context as rc
    from sglang.srt.server_args import ServerArgs as SRTServerArgs

    saved = rc.snapshot_context()
    rc.reset_context()
    try:
        outer_args = SRTServerArgs(model_path="dummy", tp_size=1)
        outer = rc.publish(outer_args, role="diffusion_gpu_worker")
        outer.override("diffusion.setup", grammar_backend="none")
        outer_bags, outer_log = outer._config_bags, outer.overrides_log()
        outer_buffer = rc.get_buffer("conditioning", object)
        diffusion_group = SimpleNamespace(world_size=1, rank_in_group=0)
        encoder_group = SimpleNamespace(world_size=1, rank_in_group=0)
        attention_group = object()
        observed = {}

        def check_encoder():
            context = rc.assert_published(observed["args"], role="scheduler")
            assert rc.get_schedule().max_running_requests == 2
            assert rc.get_parallel().tp_size == 1
            return context

        def allocate():
            check_encoder().override("encoder.setup", page_size=16)
            observed["buffer"] = rc.get_buffer("conditioning", object)
            if fail_init:
                raise RuntimeError("allocation failed")

        def make_worker(server_args, **kwargs):
            observed["args"] = server_args
            observed["context"] = check_encoder()
            srt_state._TP, srt_state._ATTN_TP = encoder_group, attention_group
            return SimpleNamespace(
                model_runner=SimpleNamespace(page_size=16),
                get_memory_pool=lambda: (object(), object()),
                alloc_memory_pool=allocate,
                init_attention_backends=check_encoder,
                init_cuda_graphs=check_encoder,
            )

        def make_cache(params):
            assert params.page_size == rc.get_schedule().page_size == 16
            return object()

        def check_diffusion():
            assert rc.get_context() is outer and rc.get_server_args() is outer_args
            assert rc.publish_role() == "diffusion_gpu_worker"
            assert (
                outer._config_bags is outer_bags and outer.overrides_log() == outer_log
            )
            assert rc.get_exec().kernel.grammar_backend == "none"
            assert rc.get_buffer("conditioning", object) is outer_buffer
            assert all(
                group is diffusion_group
                for group in (mm_state._TP, srt_state._TP, srt_state._ATTN_TP)
            )

        with (
            patch(
                "sglang.srt.server_args.ServerArgs",
                side_effect=lambda **kwargs: SRTServerArgs(
                    **dict(kwargs, model_path="dummy")
                ),
            ),
            patch(
                "sglang.srt.managers.tp_worker.TpModelWorker", side_effect=make_worker
            ),
            patch(
                "sglang.srt.mem_cache.chunk_cache.ChunkCache", side_effect=make_cache
            ),
            patch(
                f"{LLaDAImageTextEncoderRunner.__module__}.get_local_torch_device",
                return_value=torch.device("cpu"),
            ),
            patch.object(mm_state, "_TP", diffusion_group),
            patch.object(srt_state, "_TP", diffusion_group),
            patch.object(srt_state, "_ATTN_TP", diffusion_group),
        ):
            args = SimpleNamespace(
                nccl_port=29500,
                trust_remote_code=False,
                revision=None,
                pipeline_config=SimpleNamespace(text_encoder_mem_fraction_static=0.1),
            )
            if fail_init:
                with pytest.raises(RuntimeError, match="allocation failed"):
                    LLaDAImageTextEncoderRunner(
                        "/unused", object(), object(), object(), args
                    )
                check_diffusion()
                return
            runner = LLaDAImageTextEncoderRunner(
                "/unused", object(), object(), object(), args
            )
            check_diffusion()
            assert observed["context"] is not outer

            def encode(*args, **kwargs):
                assert check_encoder() is observed["context"]
                assert rc.get_schedule().page_size == 16
                assert rc.get_buffer("conditioning", object) is observed["buffer"]
                assert observed["buffer"] is not outer_buffer
                assert (
                    srt_state._TP is encoder_group
                    and srt_state._ATTN_TP is attention_group
                )
                if observed["fail_encode"]:
                    raise RuntimeError("encode failed")
                return ["encoded"]

            runner._encode_impl = encode
            for fail_encode in (False, True, False):
                observed["fail_encode"] = fail_encode
                if fail_encode:
                    with pytest.raises(RuntimeError, match="encode failed"):
                        runner.encode(["hello"], 16)
                else:
                    assert runner.encode(["hello"], 16) == ["encoded"]
                check_diffusion()
    finally:
        rc.restore_context(saved)


def test_edit_cfg_preserves_source_and_removes_negative_semantics():
    config = LLaDAImagePipelineConfig()
    batch = SimpleNamespace(
        batch_size=2,
        image_embeds=[torch.ones(3, 5)] * 2,
        source_latents=[torch.ones(8, 1, 2, 3)] * 2,
        do_classifier_free_guidance=True,
    )
    positive = config.prepare_pos_cond_kwargs(batch, "cpu", None, torch.float64)
    negative = config.prepare_neg_cond_kwargs(batch, "cpu", None, torch.float64)
    policy = CFGPolicy().build(
        batch, {"encoder_hidden_states_image": batch.image_embeds}, positive, negative
    )
    for branch, shape in zip(policy.branches, [(3, 5), (0, 5)], strict=True):
        assert [
            tuple(value.shape) for value in branch.kwargs["encoder_hidden_states_image"]
        ] == [shape] * 2
        for actual, source in zip(
            branch.kwargs["source_latents"], batch.source_latents, strict=True
        ):
            torch.testing.assert_close(actual, source.double())
    batch.image_embeds = batch.image_embeds[:1]
    with pytest.raises(ValueError, match="image_embeds has 1 entries, expected 2"):
        config.prepare_pos_cond_kwargs(batch, "cpu", None, torch.float32)
