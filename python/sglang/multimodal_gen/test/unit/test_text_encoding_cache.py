from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import BatchEncoding

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.runtime.cache import conditioning
from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentResidencyManager,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentOffloadStrategy,
)
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)

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


def get_negative_embedding_twice(stage, server_args, first_req, second_req=None):
    stage.get_or_compute_negative_text_embedding(first_req, server_args, [0])
    stage.get_or_compute_negative_text_embedding(
        second_req if second_req is not None else make_req(), server_args, [0]
    )


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

    def all_reduce(self, tensor):
        dist.all_reduce(tensor)
        return tensor

    def all_gather(self, tensor, dim):
        self.gathers += 1
        outputs = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(outputs, tensor)
        return torch.cat(outputs, dim=dim)


def run_text_encoding_dp(rank, rendezvous):
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
        with torch.no_grad(), cache.scope():
            for attempt in range(3):
                if attempt == 1 and rank == 0:
                    cache.clear()
                before = group.gathers
                outputs = stage.encode_text(
                    ["a", "bb", "ccc"], args, device="cpu", return_attention_mask=True
                )
                assert group.gathers > before
                assert outputs[0][0][:, 0, 0].tolist() == [10, 11, 12]
                assert outputs[4] == [[2, 2, 2]]
                outputs[0][0].zero_()
        assert encoder.calls == (2 if rank == 0 else 1)
        assert cache.hits == (1 if rank == 0 else 2)
    finally:
        dist.destroy_process_group()


def test_batch_dp_keeps_gathering_on_rank_local_encoder_hits(tmp_path):
    mp.spawn(
        run_text_encoding_dp,
        args=(f"file://{tmp_path / 'dp-rendezvous'}",),
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
