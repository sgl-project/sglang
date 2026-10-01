import os
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import BatchEncoding

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.configs.sample.longlive2 import LongLive2SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.cache import conditioning
from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentResidencyManager,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentOffloadStrategy,
)
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CAUSAL_BLOCK_PROMPTS_KEY,
    CAUSAL_SCENE_CUT_MASK_KEY,
    CAUSAL_SHOT_INDICES_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longlive2 import (
    LongLive2TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ming_image import (
    MingImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.text_encoding import (
    RealtimeTextEncodingStage,
    RealtimeTextState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class FullHiddenStateEncoder(TextEncoder):
    uses_sglang_forward_context = False

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        states = tuple(input_ids[..., None].float() + layer for layer in range(32))
        return BaseEncoderOutput(
            last_hidden_state=states[-1],
            hidden_states=states,
            pooler_output=states[-1][:, 0].clone(),
        )


class LibraryEncoder(torch.nn.Module):
    __init__ = FullHiddenStateEncoder.__init__
    forward = FullHiddenStateEncoder.forward


def make_text_config(output_type="tensor"):
    def postprocess(output, text_inputs, return_attention_mask=False):
        embedding = output.hidden_states[9]
        mask = text_inputs["attention_mask"].bool()
        if output_type == "tuple":
            return embedding, mask
        if output_type == "structured":
            return TextConditioningOutput(embedding, mask, [2])
        return embedding

    def tokenize(texts, _tokenizer, _kwargs):
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[len(text), 2] for text in texts]),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }
        )

    return SimpleNamespace(
        text_encoder_configs=[SimpleNamespace(tokenizer_kwargs={})],
        preprocess_text_funcs=[None],
        postprocess_text_funcs=[postprocess],
        text_encoder_extra_args=[],
        is_flux_v1=lambda: False,
        tokenize_prompt=tokenize,
        get_text_encoder_attention_mask=lambda inputs, _: inputs["attention_mask"],
        get_text_encoder_pooler_output=lambda outputs, _: outputs.pooler_output,
        build_text_conditioning_mask=lambda inputs, mask, embeds, _: mask.bool(),
        seq_lens_from_text_conditioning_mask=lambda mask: mask.sum(-1).tolist(),
    )


@pytest.mark.parametrize("output_type", ["tensor", "tuple", "structured"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
@torch.no_grad()
def test_cache_stores_only_consumed_text_conditioning(output_type, device, monkeypatch):
    encoder = FullHiddenStateEncoder().to(device).eval()
    fingerprint = conditioning._fingerprint

    def fingerprint_host_inputs(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu", "token keys must not read GPU inputs"
        return fingerprint(value)

    monkeypatch.setattr(conditioning, "_fingerprint", fingerprint_host_inputs)

    config = make_text_config(output_type)
    args = make_server_args(pipeline_config=config)

    def make_stage():
        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock()):
            stage = TextEncodingStage(text_encoders=[encoder], tokenizers=[object()])
        stage._begin_text_encoder_use = MagicMock()
        stage._text_encode_dp_group = MagicMock(return_value=None)
        return stage

    stage = make_stage()
    cache = ConditioningCache(4096)
    with cache.scope():
        first = stage.encode_text(
            "hello", args, device=device, return_attention_mask=True
        )
        expected = first[0][0].clone()
        expected_pooled = first[2][0].clone()
        first[0][0].zero_()
        first[2][0].zero_()
        restored = stage.encode_text(
            "hello", args, device=device, return_attention_mask=True
        )
        torch.testing.assert_close(restored[0][0], expected, rtol=0, atol=0)
        torch.testing.assert_close(restored[2][0], expected_pooled, rtol=0, atol=0)
        assert restored[4] == [[2]]
        assert encoder.calls == 1
        stage._begin_text_encoder_use.assert_called_once_with(0)
        assert cache.stats()["entries"] == 1
        assert cache.bytes < 32  # two embeddings, one pooled value, optional mask
        stage.encode_text("changed", args, device=device, return_attention_mask=True)
        assert encoder.calls == 2
        # The same encoder can serve different pipeline postprocessing contracts.
        make_stage().encode_text(
            "hello", args, device=device, return_attention_mask=True
        )
        assert encoder.calls == 3

    # exercise the real negative-stage boundary with only one entry of capacity
    cache = ConditioningCache(cache.bytes // 3)
    stage.encode_text = partial(stage.encode_text, device=device)
    calls_before = encoder.calls
    with cache.scope(refresh=True):
        stage.encode_text("hello", args, return_attention_mask=True)
        stage.get_or_compute_negative_text_embedding(make_req(), args, [0])
    with cache.scope():
        stage.encode_text("changed positive", args, return_attention_mask=True)
        stage.get_or_compute_negative_text_embedding(make_req(), args, [0])
    assert encoder.calls == calls_before + 3
    assert cache.hits == 1
    assert cache.bytes <= cache.max_bytes


class DummyTextEncodingStage(TextEncodingStage):
    def __init__(self):
        with patch(_GLOBAL_ARGS_PATCH) as mock_global_args:
            mock_global_args.return_value = MagicMock()
            super().__init__(text_encoders=[], tokenizers=[])
        self.calls = 0

    def encode_text(self, *args, **kwargs):
        self.calls += 1
        text = args[0]
        batch_size = len(text) if isinstance(text, list) else 1
        embeds = torch.full((batch_size, 1, 1), float(self.calls))
        mask = torch.ones((batch_size, 1), dtype=torch.int64)
        return [embeds], [mask], [], [mask], [[1] * batch_size]


def make_req(**kwargs):
    defaults = {
        "prompt": "hello",
        "negative_prompt": "bad quality",
        "do_classifier_free_guidance": True,
        "prompt_template": {"template": "{}"},
        "max_sequence_length": 1024,
        "is_warmup": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_server_args(**kwargs):
    defaults = {
        "pipeline_class_name": "LTX2TwoStagePipeline",
        "model_path": "dummy-model",
        "backend": "auto",
        "model_id": None,
        "pipeline_config": SimpleNamespace(text_encoder_configs=[]),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_group_executor(
    stage_type, device, capacity, *, fsdp=False, native=True, tokenizer=None
):
    encoder = (
        (FullHiddenStateEncoder() if native else LibraryEncoder()).to(device).eval()
    )
    config = make_text_config()
    config.supports_auto_residency = False
    config.vae_config = SimpleNamespace(use_temporal_scaling_frames=False)
    config.dit_config = SimpleNamespace(
        arch_config=SimpleNamespace(num_frames_per_block=1)
    )
    args = make_server_args(
        pipeline_config=config,
        component_precisions={},
        comfyui_mode=True,
        enable_layerwise_nvtx_marker=False,
        use_fsdp_inference=fsdp,
        disable_conditioning_cache=capacity == 0,
        conditioning_cache_max_size_mb=capacity / 1024**2,
        should_cpu_offload_component=lambda _: False,
    )
    with patch(_GLOBAL_ARGS_PATCH, return_value=args):
        stage = stage_type(
            [encoder], [tokenizer if tokenizer is not None else object()]
        )
    stage._text_encode_dp_group = Mock(return_value=None)
    stage.encode_text = partial(stage.encode_text, device=device)
    pipeline = SimpleNamespace(
        modules={"text_encoder": encoder},
        _stage_name_mapping={"text": stage},
        component_residency_strategies={},
    )
    executor = SyncExecutor(args)
    manager = ComponentResidencyManager(pipeline, args)
    strategy = Mock()
    strategy.prefetch_for_use.return_value = False
    manager.strategy_for = Mock(return_value=strategy)
    executor.component_residency_manager = manager
    return executor, stage, encoder, args


@pytest.mark.parametrize("capacity", [0, 4096])
@torch.no_grad()
def test_grouped_realtime_text_updates_each_session(capacity):
    executor, stage, encoder, args = make_group_executor(
        RealtimeTextEncodingStage, "cpu", capacity
    )
    sessions = [RealtimeSession(), RealtimeSession()]
    requests = [
        Req(
            sampling_params=SamplingParams(prompt="hello", num_inference_steps=4),
            do_classifier_free_guidance=False,
            session=session,
        )
        for session in sessions
    ]

    outputs = executor.execute_group([stage], requests, args)

    assert encoder.calls == 1
    for session, output in zip(sessions, outputs):
        state = session.get_or_create_state(RealtimeTextState)
        if capacity:
            assert state.cache_key is not None
            assert state.prompt_embeds[0] is output.prompt_embeds[0]
            assert state.prompt_seq_lens[0] is not output.prompt_seq_lens[0]
        else:
            assert state.cache_key is None


@pytest.mark.parametrize("fsdp, native", [(True, False), (True, True), (False, False)])
@torch.no_grad()
def test_one_unique_stage_skips_group_only_cache_lookup(fsdp, native, monkeypatch):
    executor, stage, encoder, args = make_group_executor(
        TextEncodingStage, "cpu", 4096, fsdp=fsdp, native=native
    )
    fingerprint = Mock(wraps=conditioning._fingerprint)
    monkeypatch.setattr(conditioning, "_fingerprint", fingerprint)
    capture_query = Mock(wraps=torch.cuda.is_current_stream_capturing)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capture_query)
    stage.forward = Mock(wraps=stage.forward)
    requests = [
        Req(
            sampling_params=SamplingParams(prompt="hello", num_inference_steps=4),
            do_classifier_free_guidance=False,
        )
        for _ in range(3)
    ]

    outputs = executor.execute_group([stage], requests, args)

    assert stage.forward.call_count == 1
    assert encoder.calls == 1
    assert outputs[0].prompt_embeds[0] is outputs[2].prompt_embeds[0]
    fingerprint.assert_not_called()
    capture_query.assert_not_called()


@pytest.mark.parametrize("capacity", [0, 1, 4096])
@pytest.mark.parametrize("fsdp", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
@torch.no_grad()
def test_grouped_conditioning_reuses_positive_and_negative_independently(
    capacity, device, fsdp, native, warmup
):
    executor, stage, encoder, args = make_group_executor(
        TextEncodingStage, device, capacity, fsdp=fsdp, native=native
    )
    stage.forward = Mock(wraps=stage.forward)

    def requests():
        return [
            Req(
                sampling_params=SamplingParams(
                    prompt=prompt, negative_prompt="bad quality", num_inference_steps=4
                ),
                do_classifier_free_guidance=True,
                is_warmup=warmup,
            )
            for prompt in ("hello", "different", "hello")
        ]

    first = executor.execute_group([stage], requests(), args)
    assert stage.forward.call_count == 2
    assert encoder.calls == 3  # two positives, one shared negative
    assert executor.conditioning_cache.group_hits == 1
    assert first[0].prompt_embeds[0] is first[2].prompt_embeds[0]
    assert first[0].negative_prompt_embeds[0] is first[1].negative_prompt_embeds[0]
    assert first[0].prompt_seq_lens is not first[2].prompt_seq_lens
    assert first[0].prompt_seq_lens[0] is not first[2].prompt_seq_lens[0]
    expected = first[0].prompt_embeds[0].clone()
    first[0].prompt_embeds[0].zero_()
    first[0].prompt_seq_lens[0][0] = 0
    assert executor.conditioning_cache._group_entries.get() is None

    second = executor.execute_group([stage], requests(), args)
    assert stage.forward.call_count == 4
    persistent = capacity == 4096 and not fsdp and not warmup
    expected_calls = (3 if native else 5) if persistent else 6
    assert encoder.calls == expected_calls
    torch.testing.assert_close(second[0].prompt_embeds[0], expected, rtol=0, atol=0)
    assert second[0].prompt_seq_lens == [[2]]
    assert executor.conditioning_cache.bytes <= capacity
    if fsdp:
        assert executor.conditioning_cache.bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("native", [False, True])
@torch.no_grad()
def test_negative_conditioning_keeps_private_device_snapshot(native):
    encoder = (FullHiddenStateEncoder() if native else LibraryEncoder()).cuda().eval()
    config = make_text_config("structured")
    config.postprocess_text_funcs[0] = Mock(wraps=config.postprocess_text_funcs[0])
    args = make_server_args(pipeline_config=config)
    with patch(_GLOBAL_ARGS_PATCH, return_value=args):
        stage = TextEncodingStage([encoder], [object()])
    stage._text_encode_dp_group = Mock(return_value=None)
    stage._begin_text_encoder_use = Mock()
    stage.encode_text = partial(stage.encode_text, device="cuda")
    cache = ConditioningCache(4096)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    expected = torch.tensor([[[20.0], [11.0]]])

    with cache.scope(), torch.cuda.stream(producer):
        torch.cuda._sleep(250_000_000)
        first = stage.get_or_compute_negative_text_embedding(make_req(), args, [0])
        first[0][0].zero_()
        first[1][0].zero_()
        first[4][0][0] = 0
        entry = next(iter(cache._entries.values()))
        assert entry.device_resident
        assert entry.output[0].device.type == "cuda"
        assert entry.output[0].data_ptr() != first[0][0].data_ptr()
        assert cache.stats()["device_bytes"] == cache.bytes <= 4096

    with cache.scope(), torch.cuda.stream(consumer):
        for _ in range(2):
            hit = stage.get_or_compute_negative_text_embedding(make_req(), args, [0])
            torch.testing.assert_close(hit[0][0].cpu(), expected, rtol=0, atol=0)
            assert hit[1][0].tolist() == [[1, 1]]
            assert hit[4] == [[2]]
            hit[0][0].zero_()
        assert encoder.calls == config.postprocess_text_funcs[0].call_count == 1
        stage._begin_text_encoder_use.assert_called_once_with(0)
        conditioning.invalidate_conditioning_caches([encoder])
        assert cache.bytes == cache.stats()["device_bytes"] == 0
        stage.get_or_compute_negative_text_embedding(make_req(), args, [0])
        assert encoder.calls == 2
        with cache.scope(refresh=True):
            stage.get_or_compute_negative_text_embedding(make_req(), args, [0])
        assert encoder.calls == 3
        stage.get_or_compute_negative_text_embedding(
            make_req(negative_prompt="changed negative"), args, [0]
        )
        assert encoder.calls == 4


@torch.no_grad()
def test_batched_request_reuses_conditioning_without_changing_output_seeds():
    executor, stage, encoder, args = make_group_executor(TextEncodingStage, "cpu", 0)
    for _ in range(2):
        request = Req(
            sampling_params=SamplingParams(
                prompt="hello",
                negative_prompt="hello",
                num_outputs_per_prompt=3,
                seed=[7, 8, 9],
            ),
            do_classifier_free_guidance=True,
        )
        output = executor.execute([stage], request, args)
        assert output.seed == [7, 8, 9]
        assert output.num_outputs_per_prompt == 3
        assert output.prompt_embeds[0] is output.negative_prompt_embeds[0]
        assert executor.conditioning_cache._group_entries.get() is None
    assert encoder.calls == 2
    assert executor.conditioning_cache.group_hits == 2


@torch.no_grad()
def test_grouped_longlive_conditioning_keeps_per_request_shot_metadata():
    executor, stage, encoder, args = make_group_executor(
        LongLive2TextEncodingStage, "cpu", 0
    )
    requests = [
        Req(
            sampling_params=LongLive2SamplingParams(
                prompt="hello",
                shot_prompts=["first", "second"],
                shot_durations=[1, 1],
                num_frames=2,
            ),
            do_classifier_free_guidance=False,
        )
        for _ in range(2)
    ]
    outputs = executor.execute_group([stage], requests, args)
    assert encoder.calls == 1
    for key in (
        CAUSAL_BLOCK_PROMPTS_KEY,
        CAUSAL_SCENE_CUT_MASK_KEY,
        CAUSAL_SHOT_INDICES_KEY,
    ):
        assert key in outputs[0].extra and key in outputs[1].extra
        assert outputs[0].extra[key] == outputs[1].extra[key]
        assert outputs[0].extra[key] is not outputs[1].extra[key]


@pytest.mark.parametrize("fsdp", [False, True])
@pytest.mark.parametrize(
    "stage_type", [MingImageEncodingStage, LingBotVideoTextEncodingStage]
)
@torch.no_grad()
def test_custom_text_stages_preserve_group_reuse_and_request_metadata(fsdp, stage_type):
    tokenizer = Mock(name_or_path="fixture", eos_token="<eos>")
    tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [len(text), 2]}
    tokenizer.convert_tokens_to_ids.side_effect = lambda text: (
        3 if text == "<image>" else 4
    )
    factory = stage_type
    if stage_type is LingBotVideoTextEncodingStage:
        factory = partial(stage_type, transformer=torch.nn.Linear(1, 1))

        def tokenize(**kwargs):
            texts = kwargs["text"]
            ids = (
                [[1]]
                if isinstance(texts, str)
                else [[1, len(text), 2] for text in texts]
            )
            return BatchEncoding(
                {
                    "input_ids": torch.tensor(ids),
                    "attention_mask": torch.ones_like(torch.tensor(ids)),
                }
            )

        tokenizer.side_effect = tokenize

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "ming_image.Qwen2VLImageProcessorPil.from_pretrained"
    ):
        executor, stage, encoder, args = make_group_executor(
            factory, "cpu", 0, fsdp=fsdp, tokenizer=tokenizer
        )
    encoder.dtype = torch.bfloat16
    encoder.config = SimpleNamespace(projection_config={"img_gen_scales": [1]})
    encoder.image_token = 99
    args.pipeline_config.dit_config.arch_config.alignment_padding_mode = "zero_masked"
    args.pipeline_config.dit_config.arch_config.multi_frame_output = False

    def requests():
        return [
            Req(
                sampling_params=SamplingParams(
                    prompt=prompt, negative_prompt="", height=64, width=64
                ),
                do_classifier_free_guidance=False,
            )
            for prompt in ("hello", "different", "hello")
        ]

    with patch(
        f"{stage_type.__module__}.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        for run in range(2):
            outputs = executor.execute_group([stage], requests(), args)
            assert encoder.calls == 2 * (run + 1)
            assert outputs[0].prompt_embeds[0] is outputs[2].prompt_embeds[0]
            assert outputs[0].prompt_embeds is not outputs[2].prompt_embeds
            if stage_type is MingImageEncodingStage:
                assert all(output.extra["ming_frames"] == 1 for output in outputs)
                assert (
                    outputs[0].extra["ming_direct"] is outputs[2].extra["ming_direct"]
                )
    assert executor.conditioning_cache._group_entries.get() is None


def get_negative_embedding_twice(stage, server_args, first_req, second_req=None):
    stage.get_or_compute_negative_text_embedding(first_req, server_args, [0])
    stage.get_or_compute_negative_text_embedding(
        second_req if second_req is not None else make_req(), server_args, [0]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("capacity", [0, 1, 4096])
@torch.no_grad()
def test_lingbot_cache_miss_prepares_offloaded_encoder(capacity, monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.managers.memory_managers."
        "component_residency_strategies.get_local_torch_device",
        lambda: device,
    )
    executor, stage, encoder, args = make_group_executor(
        partial(LingBotVideoTextEncodingStage, transformer=torch.nn.Linear(1, 1)),
        "cpu",
        capacity,
    )
    encoder.embedding = torch.nn.Embedding(8, 2)

    def encode(input_ids, **kwargs):
        encoder.calls += 1
        states = encoder.embedding(input_ids)
        return BaseEncoderOutput(last_hidden_state=states, hidden_states=(states,))

    encoder.forward = encode
    stage._crop_start = 0
    stage._build_prompt_inputs = lambda prompt, images=None: BatchEncoding(
        {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
    )
    manager = executor.component_residency_manager
    strategy = ComponentOffloadStrategy()
    strategy.prepare_for_use = Mock(wraps=strategy.prepare_for_use)
    manager.strategy_for = Mock(return_value=strategy)
    stage.set_component_residency_manager(manager)
    cache = ConditioningCache(capacity)
    results = []
    for _ in range(2):
        batch = make_req()
        manager.begin_request([stage], batch, args)
        manager.before_stage(stage, 0, batch, args)
        manager.begin_stage()
        with cache.scope():
            results.append(stage._encode_prompt("hello", device, torch.float32))
        manager.end_stage()
        manager.finish_request()
        assert encoder.embedding.weight.device.type == "cpu"
        assert results[-1][0].device == device
    expected_calls = 1 if capacity == 4096 else 2
    assert encoder.calls == strategy.prepare_for_use.call_count == expected_calls
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)


def test_negative_text_encoding_has_no_separate_gpu_cache():
    stage = DummyTextEncodingStage()
    server_args = make_server_args()
    get_negative_embedding_twice(stage, server_args, make_req())
    assert stage.calls == 2


def test_component_uses_exact_encoder_precision():
    with patch(_GLOBAL_ARGS_PATCH) as mock_global_args:
        mock_global_args.return_value = MagicMock()
        stage = TextEncodingStage(text_encoders=[object(), object()], tokenizers=[])
    server_args = make_server_args(
        component_precisions={"text_encoder_2": "fp32"},
        pipeline_config=SimpleNamespace(
            text_encoder_configs=[], text_encoder_precisions=["bf16", "bf16"]
        ),
    )

    uses = stage.component_uses(server_args)

    assert [(use.component_name, use.target_dtype) for use in uses] == [
        ("text_encoder", None),
        ("text_encoder_2", torch.float32),
    ]


def test_negative_text_encoding_warmup_does_not_seed_a_private_cache():
    stage = DummyTextEncodingStage()
    get_negative_embedding_twice(stage, make_server_args(), make_req(is_warmup=True))
    assert stage.calls == 2


@pytest.mark.parametrize("encoder_count", [1, 2])
@torch.no_grad()
def test_cache_hit_skips_encoder_residency_and_input_preparation(encoder_count):
    encoders = [FullHiddenStateEncoder().eval() for _ in range(encoder_count)]
    config = make_text_config()
    config.text_encoder_configs *= encoder_count
    config.preprocess_text_funcs *= encoder_count
    config.postprocess_text_funcs *= encoder_count
    config.supports_auto_residency = False
    config.get_text_encoder_attention_mask = Mock(
        wraps=config.get_text_encoder_attention_mask
    )
    config.seq_lens_from_text_conditioning_mask = Mock(
        wraps=config.seq_lens_from_text_conditioning_mask
    )
    args = make_server_args(
        pipeline_config=config,
        component_precisions={},
        enable_layerwise_nvtx_marker=False,
    )
    with patch(_GLOBAL_ARGS_PATCH, return_value=args):
        stage = TextEncodingStage(encoders, [object()] * encoder_count)
    stage._text_encode_dp_group = Mock(return_value=None)
    pipeline = SimpleNamespace(
        modules={
            "text_encoder" if i == 0 else f"text_encoder_{i + 1}": encoder
            for i, encoder in enumerate(encoders)
        },
        _stage_name_mapping={"text": stage},
        component_residency_strategies={},
    )
    manager = ComponentResidencyManager(pipeline, args)
    strategy = Mock()
    strategy.prefetch_for_use.return_value = False
    manager.strategy_for = Mock(return_value=strategy)
    stage.set_component_residency_manager(manager)
    cache = ConditioningCache(4096)

    def request(prompt, *, refresh=False, enabled=True, dtype=None):
        batch = make_req(is_warmup=refresh)
        manager.begin_request([stage], batch, args)
        manager.before_stage(stage, 0, batch, args)
        manager.begin_stage()
        with (cache if enabled else ConditioningCache(0)).scope(refresh=refresh):
            result = stage.encode_text(
                prompt,
                args,
                encoder_index=list(range(encoder_count)),
                device="cpu",
                dtype=dtype,
                return_attention_mask=True,
            )
        manager.end_stage()
        manager.finish_request()
        return result

    first = request("hello")
    expected = first[0][0].clone()
    first[0][0].zero_()
    first[1][0].zero_()
    first[3][0].zero_()
    first[4][0][0] = 0
    strategy.reset_mock()
    restored = request("hello")
    strategy.prepare_for_use.assert_not_called()
    strategy.wait_for_use.assert_not_called()
    assert config.get_text_encoder_attention_mask.call_count == encoder_count
    assert config.seq_lens_from_text_conditioning_mask.call_count == encoder_count
    torch.testing.assert_close(restored[0][0], expected, rtol=0, atol=0)
    assert restored[1][0].tolist() == [[1, 1]]
    assert restored[3][0].tolist() == [[True, True]]
    assert restored[4][0] == [2]

    # dtype is part of the consumed conditioning contract
    cast = request("hello", dtype=torch.float64)
    assert cast[0][0].dtype == torch.float64
    assert encoders[0].calls == 2
    request("hello", refresh=True)
    request("changed")
    request("hello", enabled=False)
    assert all(encoder.calls == 5 for encoder in encoders)
    assert strategy.prepare_for_use.call_count == 4 * encoder_count


class TextEncodingDPGroup:
    world_size = 2

    def __init__(self, rank):
        self.rank_in_group = rank
        self.gathers = 0
        self.cpu_group = dist.group.WORLD

    def all_reduce(self, tensor):
        dist.all_reduce(tensor)
        return tensor

    def all_gather(self, tensor, dim):
        self.gathers += 1
        outputs = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(outputs, tensor)
        return torch.cat(outputs, dim=dim)


def run_text_encoding_dp(rank, rendezvous, grouped, negative):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=30),
    )
    try:
        encoder = FullHiddenStateEncoder().eval()
        args = make_server_args(pipeline_config=make_text_config())
        with patch(_GLOBAL_ARGS_PATCH, return_value=args):
            stage = TextEncodingStage([encoder], [object()])
        group = TextEncodingDPGroup(rank)
        stage._text_encode_dp_group = Mock(return_value=group)
        cache = ConditioningCache(4096)
        stage_module = TextEncodingStage.__module__
        with (
            torch.no_grad(),
            cache.scope(),
            cache.group_scope(enabled=grouped),
            conditioning.prefer_conditioning_cache() if negative else nullcontext(),
            patch(f"{stage_module}.model_parallel_is_initialized", return_value=True),
            patch(f"{stage_module}.get_replica_group", return_value=group),
        ):
            for attempt in range(3):
                if attempt == 1 and rank == 0:
                    cache.clear()
                before = group.gathers
                outputs = stage.encode_text(
                    ["a", "bb", "ccc"], args, device="cpu", return_attention_mask=True
                )
                if (grouped or negative) and attempt == 2:
                    assert group.gathers == before
                else:
                    assert group.gathers > before
                assert outputs[0][0][:, 0, 0].tolist() == [10, 11, 12]
                assert outputs[4] == [[2, 2, 2]]
        assert encoder.calls == (2 if negative or rank == 0 else 1)
        assert cache.hits == (1 if negative or rank == 0 else 2)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("negative", [False, True])
def test_batch_dp_requires_consensus_to_skip_gather(tmp_path, grouped, negative):
    mp.spawn(
        run_text_encoding_dp,
        args=(f"file://{tmp_path / 'dp-rendezvous'}", grouped, negative),
        nprocs=2,
        join=True,
    )


class ShardedTextEncoder(FullHiddenStateEncoder):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.projection = torch.nn.Linear(1, 8, bias=False)
        torch.nn.init.ones_(self.projection.weight)
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        hidden = self.projection(input_ids[..., None].float())
        return BaseEncoderOutput(last_hidden_state=hidden, hidden_states=(hidden,) * 32)


def run_fsdp_conditioning(rank, rendezvous):
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=rendezvous,
        backend="nccl",
        timeout=60,
    )
    initialize_model_parallel(sequence_parallel_degree=2, ulysses_degree=2)
    try:
        encoder = ShardedTextEncoder().cuda().eval()
        fully_shard(
            encoder, mesh=init_device_mesh("cuda", (2,)), reshard_after_forward=True
        )
        assert encoder.projection.weight.to_local().numel() == 4
        args = make_server_args(pipeline_config=make_text_config())
        with patch(_GLOBAL_ARGS_PATCH, return_value=args):
            stage = TextEncodingStage([encoder], [object()])
        stage._text_encode_dp_group = Mock(return_value=None)
        cache = ConditioningCache(4096)
        with torch.no_grad(), cache.scope(cross_request=False):
            for _ in range(2):
                with cache.group_scope():
                    for attempt in range(3):
                        if rank == 0 and attempt == 1:
                            cache.clear()
                        output = stage.encode_text("hello", args, device=f"cuda:{rank}")
                        torch.testing.assert_close(
                            output[0][0][0, :, 0],
                            torch.tensor([5.0, 2.0], device=f"cuda:{rank}"),
                            rtol=0,
                            atol=0,
                        )
                assert cache._group_entries.get() is None
        assert encoder.calls == 4
        assert cache.group_hits == 2
        assert cache.bytes == 0
        assert encoder.projection.weight.to_local().numel() == 4
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_fsdp_group_reuse_agrees_across_shards_and_expires(tmp_path):
    mp.spawn(
        run_fsdp_conditioning,
        args=(f"file://{tmp_path / 'fsdp-rendezvous'}",),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_cache_hit_releases_weights_retained_after_warmup(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.managers.memory_managers."
        "component_residency_strategies.get_local_torch_device",
        lambda: torch.device("cuda"),
    )
    encoder = FullHiddenStateEncoder().eval()
    config = make_text_config()
    config.supports_auto_residency = False
    args = make_server_args(
        pipeline_config=config,
        component_precisions={},
        enable_layerwise_nvtx_marker=False,
    )
    with patch(_GLOBAL_ARGS_PATCH, return_value=args):
        stage = TextEncodingStage([encoder], [object()])
    stage._text_encode_dp_group = Mock(return_value=None)
    pipeline = SimpleNamespace(
        modules={"text_encoder": encoder},
        _stage_name_mapping={"text": stage},
        component_residency_strategies={},
    )
    manager = ComponentResidencyManager(pipeline, args)
    strategy = ComponentOffloadStrategy()
    manager.strategy_for = Mock(return_value=strategy)
    stage.set_component_residency_manager(manager)
    cache = ConditioningCache(4096)
    for warmup in [True, False]:
        batch = make_req(is_warmup=warmup)
        manager.begin_request([stage], batch, args)
        manager.before_stage(stage, 0, batch, args)
        manager.begin_stage()
        with cache.scope(refresh=warmup):
            result = stage.encode_text(
                "hello", args, device="cuda", return_attention_mask=True
            )
        manager.end_stage()
        manager.finish_request()
        assert encoder.weight.device.type == ("cuda" if warmup else "cpu")
        assert result[0][0].device.type == "cuda"
    assert encoder.calls == 1
    assert cache.hits == 1
