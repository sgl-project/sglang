# SPDX-License-Identifier: Apache-2.0
import asyncio
import itertools
import json
import sys
import time
import types
from collections import deque
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.cache_utils import DynamicCache

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.sensenova_u1 import (
    SenseNovaU1PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1SamplingParams,
)
from sglang.multimodal_gen.configs.sensenova_u1 import (
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
    SenseNovaGuidanceProfile,
    derive_guidance_profile,
    has_sensenova_u1_explicit_size,
    resolve_sensenova_u1_edit_auto_size,
)
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_model_info,
    get_non_diffusers_pipeline_name,
    is_registered_diffusion_model_path,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_vit import (
    NEOVisionConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.conversation import (
    get_conv_template,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
    _copy_right_aligned_prefix_bnsd,
    _randn_with_seed,
    prepare_flash_kv_cache,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    _flash_or_sdpa,
    _sdpa_attn_func,
    create_block_causal_mask,
    make_qwen3_rms_norm,
    npu_fia_available,
    position_ids_from_indexes,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1GenerationStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    should_include_warmup_image,
)
from sglang.srt.layers.layernorm import RMSNorm


class _FakeSenseNovaModel:
    def __init__(self):
        self.call_kwargs = None
        self.t2i_calls = []
        self.it2i_calls = []

    def t2i_generate(self, tokenizer, prompt, **kwargs):
        self.call_kwargs = {"tokenizer": tokenizer, "prompt": prompt, **kwargs}
        self.t2i_calls.append(self.call_kwargs)
        sample = torch.tensor(
            [
                [
                    [[-1.0, 0.0], [0.5, 1.0]],
                    [[-1.0, 0.0], [0.5, 1.0]],
                    [[-1.0, 0.0], [0.5, 1.0]],
                ]
            ]
        )
        return sample.repeat(kwargs["batch_size"], 1, 1, 1)

    def it2i_generate(self, tokenizer, prompt, images, **kwargs):
        self.call_kwargs = {
            "tokenizer": tokenizer,
            "prompt": prompt,
            "images": images,
            **kwargs,
        }
        self.it2i_calls.append(self.call_kwargs)
        sample = torch.tensor(
            [
                [
                    [[-1.0, 0.0], [0.5, 1.0]],
                    [[-1.0, 0.0], [0.5, 1.0]],
                    [[-1.0, 0.0], [0.5, 1.0]],
                ]
            ]
        )
        return sample.repeat(kwargs["batch_size"], 1, 1, 1)


class _FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2

    def __call__(self, text, return_tensors):
        del return_tensors
        token_count = len(text.split()) + 1
        return {"input_ids": torch.arange(1, token_count + 1).unsqueeze(0)}


def _install_sensenova_cache_dit_stub(
    monkeypatch,
    *,
    enable_error: Exception | None = None,
    disable_error: Exception | None = None,
):
    calls = {"enable": [], "disable": [], "refresh": []}
    module = types.ModuleType(
        "sglang.multimodal_gen.runtime.cache.cache_dit_integration"
    )

    class CacheDitConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.CacheDitConfig = CacheDitConfig
    module.cache_dit_overrides_key = lambda overrides: tuple(sorted(overrides.items()))
    module.resolve_cache_dit_request_overrides = lambda raw: dict(raw or {})
    module.CACHE_DIT_DBCACHE_KEYS = frozenset(
        {
            "Fn_compute_blocks",
            "Bn_compute_blocks",
            "max_warmup_steps",
            "residual_diff_threshold",
            "max_continuous_cached_steps",
        }
    )
    module.cache_dit_env_defaults = lambda: {
        "Fn_compute_blocks": envs.SGLANG_CACHE_DIT_FN,
        "Bn_compute_blocks": envs.SGLANG_CACHE_DIT_BN,
        "max_warmup_steps": envs.SGLANG_CACHE_DIT_WARMUP,
        "residual_diff_threshold": envs.SGLANG_CACHE_DIT_RDT,
        "max_continuous_cached_steps": envs.SGLANG_CACHE_DIT_MC,
    }

    def enable_cache_on_transformer(transformer, config, **kwargs):
        calls["enable"].append((transformer, config, kwargs))
        if enable_error is not None:
            # Model the important part of a real mid-mount failure: the
            # transformer has already been mutated before enable raises.
            transformer._partial_cache_dit_hook = True
            raise enable_error
        return transformer

    def disable_cache_on_transformer(transformer):
        calls["disable"].append(transformer)
        if disable_error is not None:
            raise disable_error
        if hasattr(transformer, "_partial_cache_dit_hook"):
            del transformer._partial_cache_dit_hook
        return transformer

    module.enable_cache_on_transformer = enable_cache_on_transformer
    module.disable_cache_on_transformer = disable_cache_on_transformer

    def refresh_context_on_transformer(transformer, steps, *, config=None):
        calls["refresh"].append((transformer, steps, config))

    module.refresh_context_on_transformer = refresh_context_on_transformer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def _cache_dit_server_args(*, enable_breakable_cuda_graph: bool = False):
    return SimpleNamespace(
        enable_breakable_cuda_graph=enable_breakable_cuda_graph,
    )


def _cache_dit_batch(
    *,
    num_inference_steps: int = 8,
    guidance_scale: float = 1.0,
    enable_cache_dit: bool | None = True,
    cache_dit_params: dict | None = None,
    extra: dict | None = None,
):
    """Request double carrying the fields the generation stage reads directly."""
    return SimpleNamespace(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        sampling_params=SimpleNamespace(
            enable_cache_dit=enable_cache_dit,
            cache_dit_params=cache_dit_params,
        ),
        extra={} if extra is None else extra,
    )


def _t2i_guidance_profile(cfg_scale: float) -> SenseNovaGuidanceProfile:
    return derive_guidance_profile(
        is_edit=False,
        cfg_scale=cfg_scale,
        img_cfg_scale=1.0,
    )


@pytest.mark.parametrize(
    ("is_edit", "cfg_scale", "img_cfg_scale", "expected_profile"),
    [
        (False, 0.5, 1.0, SenseNovaGuidanceProfile.CONDITION),
        (False, 4.0, 1.0, SenseNovaGuidanceProfile.CONDITION_UNCONDITIONAL),
        (True, 0.5, 1.0, SenseNovaGuidanceProfile.CONDITION_IMAGE),
        (True, 1.0, 1.0, SenseNovaGuidanceProfile.CONDITION),
        (True, 4.0, 1.0, SenseNovaGuidanceProfile.CONDITION_IMAGE),
        (True, 4.0, 4.0, SenseNovaGuidanceProfile.CONDITION_UNCONDITIONAL),
        (
            True,
            4.0,
            2.0,
            SenseNovaGuidanceProfile.CONDITION_IMAGE_UNCONDITIONAL,
        ),
    ],
)
def test_sensenova_u1_guidance_profile_matches_generation_schedule(
    is_edit, cfg_scale, img_cfg_scale, expected_profile
):
    profile = derive_guidance_profile(
        is_edit=is_edit,
        cfg_scale=cfg_scale,
        img_cfg_scale=img_cfg_scale,
    )
    assert profile is expected_profile
    assert profile.branch_count == len(expected_profile.value)


@pytest.mark.parametrize(
    ("profile", "cfg_interval", "expected_enabled"),
    [
        (SenseNovaGuidanceProfile.CONDITION_IMAGE, (0.0, 1.0), True),
        (SenseNovaGuidanceProfile.CONDITION_UNCONDITIONAL, (0.0, 1.0), True),
        (SenseNovaGuidanceProfile.CONDITION_IMAGE_UNCONDITIONAL, (0.0, 1.0), False),
        (SenseNovaGuidanceProfile.CONDITION_IMAGE, (0.2, 0.8), False),
        (SenseNovaGuidanceProfile.CONDITION_IMAGE, (0.0, 0.5), False),
    ],
)
def test_sensenova_u1_it2i_cache_dit_fails_closed_for_unsupported_schedules(
    monkeypatch, profile, cfg_interval, expected_enabled
):
    calls = _install_sensenova_cache_dit_stub(monkeypatch)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(attention_type="full_attention")],
        config=SimpleNamespace(num_hidden_layers=1),
    )
    stage = SenseNovaU1GenerationStage(
        model=SimpleNamespace(language_model=SimpleNamespace(model=transformer)),
        tokenizer="tok",
    )

    stage._maybe_enable_cache_dit(
        _cache_dit_batch(),
        _cache_dit_server_args(),
        guidance_profile=profile,
        cfg_interval=cfg_interval,
    )

    assert bool(calls["enable"]) is expected_enabled
    if expected_enabled:
        assert calls["enable"][0][2]["has_separate_cfg"] is profile.has_separate_cfg


def test_sensenova_u1_cache_dit_requires_an_explicit_guidance_profile():
    stage = SenseNovaU1GenerationStage(model=_FakeSenseNovaModel(), tokenizer="tok")

    with pytest.raises(TypeError, match="guidance_profile"):
        stage._maybe_enable_cache_dit(
            _cache_dit_batch(),
            _cache_dit_server_args(),
            cfg_interval=(0.0, 1.0),
        )


class _CacheDitRecordingBlock(torch.nn.Module):
    def __init__(self, transform=None):
        super().__init__()
        self.calls = []
        self.attention_type = "full_attention"
        self.transform = transform or (lambda hidden_states: hidden_states + 1)

    def forward(self, hidden_states, *, sensenova_marker=None, **kwargs):
        self.calls.append((sensenova_marker, kwargs))
        return self.transform(hidden_states)


class _CacheDitSenseNovaTransformer(torch.nn.Module):
    """Small SenseNova-shaped transformer for the real cache-dit wrapper test."""

    def __init__(self, layers=None):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            layers
            if layers is not None
            else [_CacheDitRecordingBlock(), _CacheDitRecordingBlock()]
        )
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.used_native_layers = []

    def forward(
        self,
        hidden_states,
        *,
        image_gen_indicators=None,
        update_cache=True,
        sensenova_marker=None,
    ):
        exist_non_image_gen_tokens = image_gen_indicators is None or bool(
            (~image_gen_indicators).any().item()
        )
        exist_image_gen_tokens = image_gen_indicators is not None and bool(
            image_gen_indicators.any().item()
        )
        layers = self.layers
        native_layers = getattr(self, "_sensenova_cache_dit_native_layers", None)
        if native_layers is not None and (
            update_cache or exist_non_image_gen_tokens or not exist_image_gen_tokens
        ):
            layers = native_layers
        self.used_native_layers.append(layers is native_layers)

        for layer in layers:
            hidden_states = layer(
                hidden_states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                update_cache=update_cache,
                sensenova_marker=sensenova_marker,
            )
        return hidden_states


_CacheDitQwen3Model = type(
    "Qwen3Model",
    (_CacheDitSenseNovaTransformer,),
    {
        "__module__": (
            "sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3"
        )
    },
)


class _RecordingTraceContext:
    tracing_enable = True

    def __init__(self):
        self.finish_count = 0
        self.started_slices = []
        self.finished_slices = []

    def trace_req_finish(self):
        self.finish_count += 1

    def trace_slice_start(self, name, level=0):
        self.started_slices.append((name, level))

    def trace_slice_end(self, name, level=0, **kwargs):
        self.finished_slices.append((name, level))


class _SequentialTestExecutor(PipelineExecutor):
    def __init__(self, server_args, *, fail=False, fail_request_ids=None):
        super().__init__(server_args)
        self.fail = fail
        self.fail_request_ids = set(fail_request_ids or [])
        self.executed_requests = []

    def execute_group(self, stages, batches, server_args):
        for batch in batches:
            batch.metrics.record_stage("InputValidationStage", 0.125)
            batch.metrics.record_memory_snapshot(
                "after_validation",
                MemorySnapshot(
                    allocated_mb=100.0,
                    reserved_mb=200.0,
                    peak_allocated_mb=300.0,
                    peak_reserved_mb=400.0,
                ),
            )
        return batches

    def execute(self, stages, batch, server_args):
        self.executed_requests.append(batch)
        if self.fail or batch.request_id in self.fail_request_ids:
            raise RuntimeError(f"generation failed for {batch.request_id}")
        return OutputBatch(
            output_file_paths=[batch.output_file_name],
            metrics=batch.metrics,
        )


class _SequentialTestPipeline:
    def __init__(self, server_args, *, fail=False, fail_request_ids=None):
        self.input_stage = InputValidationStage()
        self.executor = _SequentialTestExecutor(
            server_args,
            fail=fail,
            fail_request_ids=fail_request_ids,
        )

    def forward_batch_sequentially(self, batches, server_args):
        return self.executor.execute_group_sequentially(
            [self.input_stage, object()], batches, server_args
        )


class _WorkerBackedSchedulerClient:
    def __init__(self, worker):
        self.worker = worker

    async def forward(self, batches):
        return next(self.worker.execute_forward_sequentially(batches))


@pytest.mark.parametrize(
    ("template_name", "expected_system_message"),
    [
        (
            "Hermes-2",
            "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, \u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b\u3002",
        ),
        (
            "internlm2-chat",
            "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, \u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b\u3002",
        ),
        (
            "phi3-chat",
            "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, \u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b\u3002",
        ),
        (
            "internvl2_5",
            "\u4f60\u662f\u4e66\u751f\xb7\u4e07\u8c61\uff0c\u82f1\u6587\u540d\u662fInternVL\uff0c\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u3001\u6e05\u534e\u5927\u5b66\u53ca\u591a\u5bb6\u5408\u4f5c\u5355\u4f4d\u8054\u5408\u5f00\u53d1\u7684\u591a\u6a21\u6001\u5927\u8bed\u8a00\u6a21\u578b\u3002",
        ),
    ],
)
def test_sensenova_u1_conversation_preserves_upstream_system_prompt(
    template_name, expected_system_message
):
    assert get_conv_template(template_name).system_message == expected_system_message


def _force_generator_fallback(monkeypatch, device_type):
    original_generator = torch.Generator

    def unsupported_device_generator(device="cpu"):
        if torch.device(device).type == device_type:
            raise RuntimeError(f"Generator is unsupported on {device_type}")
        return original_generator(device)

    monkeypatch.setattr(torch, "Generator", unsupported_device_generator)


def test_sensenova_u1_randn_fallback_preserves_cpu_rng(monkeypatch):
    _force_generator_fallback(monkeypatch, "cpu")
    rng_state = torch.get_rng_state().clone()

    first = _randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)
    second = _randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_sensenova_u1_randn_supports_per_sample_seeds():
    actual = _randn_with_seed(
        (2, 3, 4), device=torch.device("cpu"), dtype=torch.float32, seed=[7, 19]
    )
    expected = torch.cat(
        [
            _randn_with_seed(
                (1, 3, 4),
                device=torch.device("cpu"),
                dtype=torch.float32,
                seed=seed,
            )
            for seed in (7, 19)
        ]
    )

    assert torch.equal(actual, expected)


def test_sensenova_u1_builds_padded_batched_text_inputs():
    model = SimpleNamespace(device=torch.device("cpu"))

    input_ids, indexes, attention_mask, valid_mask, prefix_lengths = (
        NEOChatModel._build_t2i_text_inputs(
            model, _FakeTokenizer(), ["short", "a much longer prompt"]
        )
    )

    assert input_ids.shape == (2, 5)
    assert indexes.shape == (2, 3, 5)
    assert prefix_lengths.tolist() == [2, 5]
    assert valid_mask.tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
    ]
    mask = attention_mask["full_attention"]
    assert mask.shape == (2, 1, 5, 5)
    assert torch.isneginf(mask[0, :, :, 2:]).all()
    assert torch.isfinite(mask[0, :, :, :2]).any()


def test_sensenova_u1_position_indexes_support_batched_inputs():
    indexes = torch.tensor(
        [
            [[0, 1], [0, 0], [0, 0]],
            [[4, 4], [0, 1], [0, 0]],
        ]
    )

    assert torch.equal(position_ids_from_indexes(indexes, 0), indexes[:, 0])
    assert torch.equal(
        position_ids_from_indexes(indexes[0], 1), indexes[0, 1].unsqueeze(0)
    )


def test_sensenova_u1_singleton_text_matches_valid_batched_tokens():
    model = SimpleNamespace(device=torch.device("cpu"))
    tokenizer = _FakeTokenizer()
    batched = NEOChatModel._build_t2i_text_inputs(
        model, tokenizer, ["short", "a much longer prompt"]
    )
    for i, prompt in enumerate(["short", "a much longer prompt"]):
        single = NEOChatModel._build_t2i_text_inputs(model, tokenizer, prompt)
        length = single[0].shape[1]
        assert torch.equal(batched[0][i, :length], single[0][0])
        assert torch.equal(batched[1][i, :, :length], single[1])
        assert torch.equal(
            batched[2]["full_attention"][i, :, :length, :length],
            single[2]["full_attention"][0],
        )


def test_sensenova_u1_block_causal_mask_rejects_padded_keys():
    indexes = torch.tensor([[0, 1, 2], [0, 1, 2]])
    valid = torch.tensor([[True, True, False], [True, True, True]])

    mask = create_block_causal_mask(indexes, valid)

    assert mask.shape == (2, 1, 3, 3)
    assert torch.isneginf(mask[0, :, :, 2]).all()
    assert mask[1, 0, 2, 2] == 0


def test_sensenova_u1_builds_per_sample_image_indexes():
    indexes = NEOChatModel._build_t2i_image_indexes(
        SimpleNamespace(),
        token_h=2,
        token_w=2,
        text_len=torch.tensor([2, 5]),
        device=torch.device("cpu"),
    )

    assert indexes.shape == (2, 3, 4)
    assert indexes[:, 0].tolist() == [[2, 2, 2, 2], [5, 5, 5, 5]]
    assert indexes[:, 1].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert indexes[:, 2].tolist() == [[0, 1, 0, 1], [0, 1, 0, 1]]


def test_sensenova_u1_compacts_variable_length_kv_before_attention():
    generator = torch.Generator().manual_seed(29)
    q = torch.randn(2, 3, 4, 8, generator=generator)
    k = torch.randn(2, 8, 2, 8, generator=generator)
    v = torch.randn(2, 8, 2, 8, generator=generator)
    actual = _flash_or_sdpa(
        q,
        k,
        v,
        actual_seq_lengths_kv=[5, 8],
    )

    expected_short = _sdpa_attn_func(
        q[:1],
        torch.cat((k[:1, :2], k[:1, 5:]), dim=1),
        torch.cat((v[:1, :2], v[:1, 5:]), dim=1),
    )
    expected_long = _sdpa_attn_func(q[1:], k[1:], v[1:])
    torch.testing.assert_close(actual, torch.cat((expected_short, expected_long)))


def test_sensenova_u1_sdpa_masks_padded_prefix_keys():
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 0.0]], [[0.0, 4.0]], [[100.0, 100.0]]]])
    key_mask = torch.tensor([[[[True, True, False]]]])

    actual = _sdpa_attn_func(q, k, v, attention_mask=key_mask)
    expected = _sdpa_attn_func(q, k[:, :2], v[:, :2])

    torch.testing.assert_close(actual, expected)


def test_sensenova_u1_right_aligns_bnsd_prefix_for_npu_fia():
    source = torch.tensor(
        [
            [[[1], [2], [99], [99], [99]]],
            [[[3], [4], [5], [6], [7]]],
        ]
    )
    destination = torch.zeros(2, 1, 8, 1, dtype=source.dtype)

    _copy_right_aligned_prefix_bnsd(destination, source, [2, 5])

    assert destination[:, 0, :5, 0].tolist() == [
        [0, 0, 0, 1, 2],
        [3, 4, 5, 6, 7],
    ]
    assert destination[:, :, 5:].eq(0).all()


@pytest.mark.parametrize("available", [False, True])
def test_sensenova_u1_npu_fia_checks_operator_availability(monkeypatch, available):
    namespace = SimpleNamespace()
    if available:
        namespace.npu_fused_infer_attention_score = object()
    monkeypatch.setattr(torch.ops, "npu", namespace, raising=False)

    assert npu_fia_available() is available


def test_sensenova_u1_shared_rmsnorm_uses_framework_dispatch():
    norm = make_qwen3_rms_norm(64, eps=1e-6)

    assert isinstance(norm, RMSNorm)
    assert norm.cast_x_before_out_mul
    assert norm._forward_method != norm.forward_native


@torch.no_grad()
def test_sensenova_u1_fused_dense_mlp_matches_original(monkeypatch):
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="silu",
    )
    with torch.random.fork_rng():
        torch.manual_seed(37)
        mlp = Qwen3MLP(config).eval()
        hidden_states = torch.randn(2, 5, config.hidden_size)
        expected = mlp(hidden_states)

    monkeypatch.setattr(mlp, "_use_npu_fused_mlp", lambda _x: True)
    monkeypatch.setattr(
        torch.ops,
        "npu",
        SimpleNamespace(
            npu_swiglu=lambda x, dim=-1: (
                F.silu(x.chunk(2, dim=dim)[0]) * x.chunk(2, dim=dim)[1]
            )
        ),
        raising=False,
    )
    actual = mlp(hidden_states)

    torch.testing.assert_close(actual, expected)
    assert set(mlp.state_dict()) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }
    assert (
        mlp.gate_proj.weight.untyped_storage().data_ptr()
        == mlp.up_proj.weight.untyped_storage().data_ptr()
    )


def test_sensenova_u1_batched_gqa_matches_unpadded_singletons():
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(2, 3, 4, 8, generator=generator)
    k = torch.randn(2, 8, 2, 8, generator=generator)
    v = torch.randn(2, 8, 2, 8, generator=generator)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, 2:5] = False
    attention_mask = valid[:, None, None, :].expand(-1, -1, q.shape[1], -1)

    actual = _sdpa_attn_func(q, k, v, attention_mask=attention_mask)

    for i in range(2):
        expected = _sdpa_attn_func(
            q[i : i + 1], k[i : i + 1, valid[i]], v[i : i + 1, valid[i]]
        )
        torch.testing.assert_close(actual[i : i + 1], expected)


@torch.no_grad()
def test_sensenova_u1_prefix_and_denoise_attention_match_singletons():
    config = NEOLLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(23)
        attention = Qwen3Attention(config, layer_idx=0).eval()
    generator = torch.Generator().manual_seed(31)
    text = torch.randn(2, 5, 64, generator=generator)
    image = torch.randn(2, 3, 64, generator=generator)
    helper = SimpleNamespace(device=torch.device("cpu"))

    def run(prefix, lengths, image_states):
        batch_size, width, _ = prefix.shape
        positions = torch.arange(width).expand(batch_size, -1)
        indexes = torch.stack(
            [positions, torch.zeros_like(positions), torch.zeros_like(positions)], dim=1
        )
        valid = positions < torch.tensor(lengths)[:, None]
        cache = DynamicCache(config=config)
        attention.forward_und(
            prefix, indexes, create_block_causal_mask(positions, valid), cache
        )
        prefix_keys = cache.layers[0].keys.clone()
        prepare_flash_kv_cache(
            cache,
            current_len=3,
            batch_size=batch_size,
            prefix_lengths=torch.tensor(lengths),
        )
        image_indexes = NEOChatModel._build_t2i_image_indexes(
            helper, 1, 3, torch.tensor(lengths), torch.device("cpu")
        )
        outputs = []
        for _ in range(2):
            image_states, _ = attention.forward_gen(
                image_states,
                image_indexes,
                None,
                cache,
                update_cache=False,
            )
            outputs.append(image_states)
        torch.testing.assert_close(cache.layers[0].keys, prefix_keys)
        return prefix_keys, outputs

    keys, batched = run(text, [2, 5], image)
    for i, length in enumerate([2, 5]):
        single_keys, single = run(text[i : i + 1, :length], [length], image[i : i + 1])
        torch.testing.assert_close(keys[i : i + 1, :, :length], single_keys)
        for step in range(2):
            torch.testing.assert_close(
                batched[step][i : i + 1], single[step], atol=1e-5, rtol=1e-4
            )


def test_sensenova_u1_randn_fallback_preserves_device_rng(monkeypatch):
    device_type = current_platform.device_type
    if not device_type or device_type == "cpu":
        pytest.skip("No accelerator is available")

    device = torch.device(device_type, 0)
    device_module = torch.get_device_module(device)
    if not device_module.is_available():
        pytest.skip(f"{device_type} is not available")

    _force_generator_fallback(monkeypatch, device_type)
    cpu_rng_state = torch.get_rng_state().clone()
    device_rng_state = device_module.get_rng_state(device).clone()

    first = _randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)
    second = _randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), cpu_rng_state)
    assert torch.equal(device_module.get_rng_state(device), device_rng_state)


def test_sensenova_u1_registry_resolves_local_and_hf_paths(tmp_path):
    _get_config_info.cache_clear()
    get_model_info.cache_clear()

    local_path = tmp_path / "checkpoint-revision-abc123"
    local_path.mkdir()
    (local_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NEOChatModel"],
                "model_type": "neo_chat",
            }
        )
    )

    assert is_registered_diffusion_model_path(str(local_path))
    assert get_non_diffusers_pipeline_name(str(local_path)) == "SenseNovaU1Pipeline"

    local_model_info = get_model_info(str(local_path))
    assert local_model_info is not None
    assert local_model_info.pipeline_config_cls is SenseNovaU1PipelineConfig
    assert local_model_info.sampling_param_cls is SenseNovaU1SamplingParams

    model_info = get_model_info("sensenova/SenseNova-U1.5-8B-MoT")
    assert model_info is not None
    assert model_info.pipeline_config_cls is SenseNovaU1PipelineConfig
    assert model_info.sampling_param_cls is SenseNovaU1SamplingParams

    modelscope_id = "SenseNova/SenseNova-U1.5-8B-MoT"
    assert is_registered_diffusion_model_path(modelscope_id)
    assert get_non_diffusers_pipeline_name(modelscope_id) == "SenseNovaU1Pipeline"
    assert get_model_info(modelscope_id) is not None
    get_model_info.cache_clear()


def test_sensenova_u1_registry_requires_exact_hub_id(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        lambda _: {},
    )
    _get_config_info.cache_clear()
    get_model_info.cache_clear()

    unrelated_repo = "acme/SenseNova-U1.5-8B-MoT"
    assert not is_registered_diffusion_model_path(unrelated_repo)
    assert get_non_diffusers_pipeline_name(unrelated_repo) is None
    assert _get_config_info(unrelated_repo) is None

    _get_config_info.cache_clear()
    get_model_info.cache_clear()


def test_sensenova_u1_registry_does_not_route_lora_only_repositories(tmp_path):
    lora_repo = "sensenova/SenseNova-U1.5-8B-MoT-LoRA"
    lora_path = tmp_path / "SenseNova-U1.5-8B-MoT-LoRA"
    lora_path.mkdir()
    (lora_path / "adapter_config.json").write_text("{}")

    assert get_non_diffusers_pipeline_name(lora_repo) is None
    assert get_non_diffusers_pipeline_name(str(lora_path)) is None
    assert not is_registered_diffusion_model_path(lora_repo)
    assert not is_registered_diffusion_model_path(str(lora_path))


@pytest.mark.parametrize("backend", ["auto", "sglang", "diffusers"])
def test_sensenova_u1_known_adapter_only_repo_rejected_before_backend_resolution(
    monkeypatch, backend
):
    def fail_model_index_download(_):
        raise AssertionError("adapter-only repo should not download model_index")

    def fail_diffusers_resolution(**_kwargs):
        raise AssertionError("adapter-only repo should not resolve diffusers info")

    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        fail_model_index_download,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry._get_diffusers_model_info",
        fail_diffusers_resolution,
    )
    get_model_info.cache_clear()

    loras_repo = "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"

    assert get_non_diffusers_pipeline_name(loras_repo) is None
    assert get_model_info(loras_repo, backend=backend) is None
    get_model_info.cache_clear()


def test_sensenova_u1_sampling_params_keep_private_defaults_internal():
    params = SenseNovaU1SamplingParams(prompt="hello", width=2304, height=4096)

    assert params.guidance_scale == 4.0
    assert params.num_inference_steps == 50
    assert params.num_outputs_per_prompt == 1
    assert params.cfg_norm == "none"
    assert params.timestep_shift == 3.0

    extra = params.build_request_extra()[SENSENOVA_U1_REQUEST_EXTRA_KEY]
    assert extra == {
        "cfg_norm": "none",
        "timestep_shift": 3.0,
        "enable_timestep_shift": True,
        "cfg_interval": (0.0, 1.0),
        "t_eps": 0.02,
        "think_mode": False,
        "img_cfg_scale": 1.0,
        "input_max_pixels": None,
        "do_resize": True,
    }


def test_sensenova_u1_rejects_unaligned_resolution():
    with pytest.raises(ValueError, match="divisible by 32"):
        SenseNovaU1SamplingParams(width=2160, height=3840)


def test_sensenova_u1_accepts_openai_image_api_num_frames():
    params = SenseNovaU1SamplingParams(
        prompt="hello",
        width=2048,
        height=2048,
        num_frames=1,
    )

    assert params.num_frames == 1
    assert params.data_type == DataType.IMAGE


def test_sensenova_u1_resolves_edit_auto_size_from_first_input_ratio():
    assert resolve_sensenova_u1_edit_auto_size(416, 608) == (1408, 2048)
    assert resolve_sensenova_u1_edit_auto_size(1600, 800) == (2048, 1024)
    assert resolve_sensenova_u1_edit_auto_size(100, 1000) == (512, 2048)


def test_sensenova_u1_detects_explicit_size_fields():
    assert has_sensenova_u1_explicit_size({"size"})
    assert has_sensenova_u1_explicit_size({"width"})
    assert has_sensenova_u1_explicit_size({"height"})
    assert not has_sensenova_u1_explicit_size({"prompt", "image_path"})


def test_sensenova_u1_sampling_adjust_scales_first_input_to_2k_long_side(tmp_path):
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (1600, 800)).save(image_path)
    params = SenseNovaU1SamplingParams(
        prompt="replace text",
        image_path=str(image_path),
    )
    params._explicit_fields = {"prompt", "image_path"}

    params._adjust(
        SimpleNamespace(
            pipeline_config=SenseNovaU1PipelineConfig(),
            output_path=None,
            comfyui_mode=False,
            num_gpus=1,
        )
    )

    assert (params.width, params.height) == (2048, 1024)


def test_sensenova_u1_sampling_adjust_uses_preprocessed_image_size(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (1600, 800)).save(image_path)

    def load_transposed_image(path, convert_method=None):
        assert path == str(image_path)
        image = Image.new("RGB", (800, 1600))
        return convert_method(image) if convert_method is not None else image

    monkeypatch.setattr(
        "sglang.multimodal_gen.configs.sample.sensenova_u1.load_image",
        load_transposed_image,
    )
    params = SenseNovaU1SamplingParams(
        prompt="replace text",
        image_path=str(image_path),
    )
    params._explicit_fields = {"prompt", "image_path"}

    params._adjust(
        SimpleNamespace(
            pipeline_config=SenseNovaU1PipelineConfig(),
            output_path=None,
            comfyui_mode=False,
            num_gpus=1,
        )
    )

    assert (params.width, params.height) == (1024, 2048)


def test_sensenova_u1_sampling_adjust_preserves_explicit_size(tmp_path):
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (1600, 800)).save(image_path)
    params = SenseNovaU1SamplingParams(
        prompt="replace text",
        image_path=str(image_path),
        width=1024,
        height=1024,
    )
    params._explicit_fields = {"prompt", "image_path", "size"}

    params._adjust(
        SimpleNamespace(
            pipeline_config=SenseNovaU1PipelineConfig(),
            output_path=None,
            comfyui_mode=False,
            num_gpus=1,
        )
    )

    assert (params.width, params.height) == (1024, 1024)


def test_sensenova_u1_scheduler_capabilities():
    config = SenseNovaU1PipelineConfig()

    assert config.task_type.name == "TI2I"
    assert config.supports_dynamic_batching()
    assert config.supports_sequential_multi_output_inference()


def test_sensenova_u1_warmup_defaults_to_text_to_image_signature():
    server_args = SimpleNamespace(
        pipeline_config=SenseNovaU1PipelineConfig(),
        enable_breakable_cuda_graph=False,
    )

    assert should_include_warmup_image(server_args, server_based_warmup=True) is False
    assert should_include_warmup_image(server_args, server_based_warmup=False) is False


def _make_sensenova_u1_scheduler_request(
    request_id: str, prompt: str, seed: int | list[int], **sampling_overrides
) -> Req:
    sampling = SenseNovaU1SamplingParams(
        prompt=prompt,
        seed=seed,
        **sampling_overrides,
    )
    return Req(
        request_id=request_id,
        prompt=prompt,
        seed=seed,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
    )


def test_sensenova_u1_batch_cost_tracks_resolution_steps_and_cfg():
    config = SenseNovaU1PipelineConfig()
    batch = SimpleNamespace(
        width=1024,
        height=1024,
        num_inference_steps=5,
        guidance_scale=4.0,
        num_outputs_per_prompt=1,
    )

    assert config.estimate_request_cost(batch) == 32 * 32 * 5 * 2


def test_sensenova_u1_multi_output_request_is_not_dynamically_batched():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake", num_outputs_per_prompt=2
    )
    request = SimpleNamespace(
        is_warmup=False,
        realtime_session_id=None,
        session=None,
        prompt=sampling.prompt,
        image_path=None,
        return_file_paths_only=False,
        num_outputs_per_prompt=2,
        sampling_params=sampling,
    )

    assert not scheduler._can_dynamic_batch(request, request)
    assert (
        scheduler._get_dynamic_batch_reject_reason(request, request)
        == "sequential_multi_output"
    )


def test_sensenova_u1_think_mode_request_is_dispatched_without_batching():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    scheduler._batch_admission = SimpleNamespace(enabled=True)
    scheduler._batch_metrics_enabled = False
    request = _make_sensenova_u1_scheduler_request(
        "request-0", "a mountain lake", 7, think_mode=True
    )
    scheduler.waiting_queue = deque([(b"identity", request, time.monotonic())])

    assert not scheduler._can_dynamic_batch(request, request)
    assert (
        scheduler._get_dynamic_batch_reject_reason(request, request)
        == "pipeline_request_unsupported"
    )
    items = scheduler.get_next_batch_to_run()
    assert items is not None
    assert items[0][0] == b"identity"
    assert items[0][1] is request
    assert not scheduler.waiting_queue


@pytest.mark.parametrize(
    "sampling_overrides",
    [
        {"guidance_scale": 1.0},
        {"num_inference_steps": 25},
        {"cfg_norm": "global"},
        {"timestep_shift": 2.0},
        {"t_eps": 0.01},
    ],
)
def test_sensenova_u1_scheduler_rejects_heterogeneous_generation_options(
    sampling_overrides,
):
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    base = _make_sensenova_u1_scheduler_request("request-0", "first", 7)
    candidate = _make_sensenova_u1_scheduler_request(
        "request-1", "second", 19, **sampling_overrides
    )

    assert not scheduler._can_dynamic_batch(base, candidate)


def test_sensenova_u1_scheduler_normalizes_single_output_seed_lists():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    requests = [
        _make_sensenova_u1_scheduler_request("request-0", "first", [7]),
        _make_sensenova_u1_scheduler_request("request-1", "second", 19),
    ]

    merged = scheduler._try_merge_generation_reqs(requests)

    assert merged.extra["dynamic_batch_seeds"] == [7, 19]


def test_sensenova_u1_scheduler_merge_and_split_preserve_request_order():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    requests = [
        _make_sensenova_u1_scheduler_request(
            "request-0",
            "short",
            7,
            output_path="/tmp/first",
            output_file_name="first.png",
        ),
        _make_sensenova_u1_scheduler_request(
            "request-1",
            "a longer prompt",
            19,
            output_path="/tmp/second",
            output_file_name="second.png",
        ),
    ]
    merged = scheduler._try_merge_generation_reqs(requests)
    assert merged.prompt == ["short", "a longer prompt"]
    assert merged.extra["dynamic_batch_seeds"] == [7, 19]
    expected_paths = [request.output_file_path() for request in requests]
    assert merged.extra["dynamic_batch_output_paths"] == expected_paths
    assert requests[0].prompt == "short"
    outputs = scheduler._split_batched_output(
        OutputBatch(
            output=[torch.tensor([7]), torch.tensor([19])],
            output_file_paths=expected_paths,
        ),
        requests,
    )
    assert [output.output[0].item() for output in outputs] == [7, 19]
    assert [output.output_file_paths for output in outputs] == [
        [path] for path in expected_paths
    ]
    assert (
        scheduler._split_batched_output(
            OutputBatch(output=[torch.tensor([7])]), requests
        )
        is None
    )
    requests[1].sampling_params.width = 1024
    del requests[1]._dynamic_batch_sig
    assert scheduler._try_merge_generation_reqs(requests) is None


def test_sensenova_u1_rejects_multi_gpu_during_arg_validation():
    config = SenseNovaU1PipelineConfig()

    with pytest.raises(ValueError, match="num_gpus=1"):
        config.validate_server_args(
            SimpleNamespace(
                num_gpus=2,
                enable_torch_compile=False,
                lora_path=None,
                attention_backend=None,
                component_attention_backends={},
            )
        )


def test_sensenova_u1_clears_auto_tuned_runtime_defaults():
    config = SenseNovaU1PipelineConfig()
    args = SimpleNamespace(
        num_gpus=1,
        enable_torch_compile=False,
        lora_path=None,
        component_residency={"transformer": "layerwise-offload"},
        cpu_offload_components=["transformer"],
        dit_cpu_offload=True,
        text_encoder_cpu_offload=True,
        image_encoder_cpu_offload=True,
        vae_cpu_offload=True,
        dit_layerwise_offload=True,
        layerwise_offload_components=["transformer"],
        quantization=None,
        quantization_ignored_layers=None,
        transformer_weights_path=None,
        component_paths={"model": "/tmp/component"},
        component_weights_paths={"model": "/tmp/model.safetensors"},
        component_quantizations={},
        component_quantization_ignored_layers={},
        component_precisions={},
        attention_backend="aiter",
        component_attention_backends={"text_encoder": "torch_sdpa"},
        attention_backend_config={"foo": "bar"},
        is_arg_explicitly_set=lambda _name: False,
    )

    config.validate_server_args(args)

    assert args.component_residency is None
    assert args.cpu_offload_components is None
    assert args.dit_cpu_offload is False
    assert args.text_encoder_cpu_offload is False
    assert args.image_encoder_cpu_offload is False
    assert args.vae_cpu_offload is False
    assert args.dit_layerwise_offload is False
    assert args.layerwise_offload_components is None
    assert args.component_paths == {}
    assert args.component_weights_paths == {}
    assert args.attention_backend is None
    assert args.component_attention_backends == {}
    assert args.attention_backend_config is None


def test_sensenova_u1_allows_explicit_resident_component_residency():
    config = SenseNovaU1PipelineConfig()
    args = SimpleNamespace(
        num_gpus=1,
        enable_torch_compile=False,
        lora_path=None,
        component_residency={"transformer": "resident"},
        cpu_offload_components=None,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        dit_layerwise_offload=False,
        layerwise_offload_components=None,
        quantization=None,
        quantization_ignored_layers=None,
        transformer_weights_path=None,
        component_paths={},
        component_weights_paths={},
        component_quantizations={},
        component_quantization_ignored_layers={},
        component_precisions={},
        attention_backend=None,
        component_attention_backends={},
        attention_backend_config={},
        is_arg_explicitly_set=lambda name: name == "component_residency",
    )

    config.validate_server_args(args)

    assert args.component_residency == {"transformer": "resident"}


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"enable_torch_compile": True}, "torch.compile"),
        ({"lora_path": "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"}, "LoRA adapters"),
        (
            {"component_residency": {"transformer": "component-offload"}},
            "component residency offload",
        ),
        ({"cpu_offload_components": ["transformer"]}, "CPU offload"),
        ({"dit_cpu_offload": True}, "DiT CPU offload"),
        ({"text_encoder_cpu_offload": True}, "text encoder CPU offload"),
        ({"image_encoder_cpu_offload": True}, "image encoder CPU offload"),
        ({"vae_cpu_offload": True}, "VAE CPU offload"),
        ({"dit_layerwise_offload": True}, "DiT layerwise offload"),
        ({"layerwise_offload_components": ["transformer"]}, "layerwise offload"),
        ({"quantization": "fp8"}, "quantization"),
        ({"quantization_ignored_layers": ["foo"]}, "quantization ignored layers"),
        (
            {"transformer_weights_path": "/tmp/transformer.safetensors"},
            "pre-quantized transformer weights",
        ),
        ({"component_paths": {"model": "/tmp/component"}}, "component path overrides"),
        (
            {"component_weights_paths": {"model": "/tmp/model.safetensors"}},
            "component weight path overrides",
        ),
        ({"component_quantizations": {"transformer": "fp8"}}, "component quantization"),
        (
            {"component_quantization_ignored_layers": {"transformer": ["foo"]}},
            "component quantization ignored layers",
        ),
        ({"component_precisions": {"transformer": "fp16"}}, "component precision"),
        ({"attention_backend": "fa"}, "custom attention backends"),
        (
            {"component_attention_backends": {"text_encoder": "torch_sdpa"}},
            "component attention backends",
        ),
        ({"attention_backend_config": {"foo": "bar"}}, "attention backend config"),
    ],
)
def test_sensenova_u1_rejects_unsupported_runtime_modes(override, expected):
    config = SenseNovaU1PipelineConfig()
    args = {
        "num_gpus": 1,
        "enable_torch_compile": False,
        "lora_path": None,
        "component_residency": None,
        "cpu_offload_components": None,
        "dit_cpu_offload": None,
        "text_encoder_cpu_offload": None,
        "image_encoder_cpu_offload": None,
        "vae_cpu_offload": False,
        "dit_layerwise_offload": None,
        "layerwise_offload_components": None,
        "quantization": None,
        "quantization_ignored_layers": None,
        "transformer_weights_path": None,
        "component_paths": {},
        "component_weights_paths": {},
        "component_quantizations": {},
        "component_quantization_ignored_layers": {},
        "component_precisions": {},
        "attention_backend": None,
        "component_attention_backends": {},
        "attention_backend_config": {},
    }
    args.update(override)

    with pytest.raises(ValueError, match=expected):
        config.validate_server_args(SimpleNamespace(**args))


def test_sensenova_u1_rejects_direct_server_args_quantization():
    config = SenseNovaU1PipelineConfig()

    with pytest.raises(ValueError, match="quantization"):
        ServerArgs(
            model_path="sensenova/SenseNova-U1.5-8B-MoT",
            pipeline_config=config,
            quantization="fp8",
        )


def test_sensenova_u1_rejects_file_valued_component_paths(tmp_path):
    config = SenseNovaU1PipelineConfig()

    with pytest.raises(ValueError, match="component weight path overrides"):
        ServerArgs(
            model_path="sensenova/SenseNova-U1.5-8B-MoT",
            pipeline_config=config,
            component_paths={"model": str(tmp_path / "model.safetensors")},
        )


def test_sensenova_u1_vision_config_round_trips_sequence_fields(tmp_path):
    config = NEOVisionConfig(llm_hidden_size=2048, downsample_ratio=0.5)
    config.save_pretrained(tmp_path)

    loaded = NEOVisionConfig.from_pretrained(tmp_path)

    assert loaded.llm_hidden_size == (2048,)
    assert loaded.downsample_ratio == (0.5,)


def test_sensenova_u1_vision_config_normalizes_nested_singletons():
    config = NEOVisionConfig(llm_hidden_size=[[2048]], downsample_ratio=[[0.5]])

    assert config.llm_hidden_size == (2048,)
    assert config.downsample_ratio == (0.5,)


def test_sensenova_u1_rejects_video_frame_count():
    with pytest.raises(ValueError, match="num_frames=1"):
        SenseNovaU1SamplingParams(width=2048, height=2048, num_frames=2)


def test_sensenova_u1_cli_args_expose_only_sglang_compatible_fields():
    args = SimpleNamespace(
        prompt="hello",
        image_path="image_google.png",
        width=2304,
        height=4096,
        guidance_scale=4.5,
        num_inference_steps=30,
        num_outputs_per_prompt=2,
        profile=True,
        profile_all_stages=True,
        num_profiled_timesteps=3,
        perf_dump_path="/tmp/sensenova-perf.json",
        cfg_norm="global",
        timestep_shift=9.0,
        think_mode=True,
        enable_cache_dit=True,
        cache_dit_params={"residual_diff_threshold": 0.1},
    )

    cli_args = SenseNovaU1SamplingParams.get_cli_args(args)

    assert cli_args["prompt"] == "hello"
    assert cli_args["image_path"] == "image_google.png"
    assert cli_args["width"] == 2304
    assert cli_args["height"] == 4096
    assert cli_args["guidance_scale"] == 4.5
    assert cli_args["num_inference_steps"] == 30
    assert cli_args["num_outputs_per_prompt"] == 2
    assert cli_args["enable_cache_dit"] is True
    assert cli_args["cache_dit_params"] == {"residual_diff_threshold": 0.1}
    assert "cfg_norm" not in cli_args
    assert "timestep_shift" not in cli_args
    assert "think_mode" not in cli_args
    request = Req(sampling_params=SenseNovaU1SamplingParams(**cli_args))
    assert request.profile
    assert request.profile_all_stages
    assert request.num_profiled_timesteps == 3
    assert request.perf_dump_path == "/tmp/sensenova-perf.json"


def test_sensenova_u1_generation_stage_loads_image_path_rgba_with_white_background(
    tmp_path,
):
    image_path = tmp_path / "transparent.png"
    image = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    image.putpixel((1, 0), (255, 0, 0, 255))
    image.save(image_path)
    sampling = SenseNovaU1SamplingParams(
        prompt="replace text",
        image_path=str(image_path),
        width=2048,
        height=2048,
        seed=7,
        do_resize=False,
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=sampling.num_outputs_per_prompt,
        condition_image=None,
        image_path=str(image_path),
        extra=sampling.build_request_extra(),
        metrics=None,
        sampling_params=SimpleNamespace(enable_cache_dit=False, cache_dit_params=None),
    )
    model = _FakeSenseNovaModel()

    SenseNovaU1GenerationStage(model=model, tokenizer="tok").forward(
        batch, server_args=SimpleNamespace()
    )

    loaded = model.it2i_calls[0]["images"][0]
    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((1, 0)) == (255, 0, 0)


def test_sensenova_u1_generation_stage_uses_sglang_params_and_single_model_batch():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        guidance_scale=4.5,
        num_inference_steps=30,
        seed=123,
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=sampling.num_outputs_per_prompt,
        extra=sampling.build_request_extra(),
        metrics=None,
        sampling_params=SimpleNamespace(enable_cache_dit=False, cache_dit_params=None),
    )
    model = _FakeSenseNovaModel()
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")

    output = stage.forward(batch, server_args=_cache_dit_server_args())

    assert len(output.output) == 1
    assert torch.allclose(
        output.output[0],
        torch.tensor(
            [
                [[0.0, 0.5], [0.75, 1.0]],
                [[0.0, 0.5], [0.75, 1.0]],
                [[0.0, 0.5], [0.75, 1.0]],
            ]
        ),
    )
    assert model.call_kwargs["tokenizer"] == "tok"
    assert model.call_kwargs["prompt"] == "a mountain lake"
    assert model.call_kwargs["image_size"] == (2304, 4096)
    assert model.call_kwargs["cfg_scale"] == 4.5
    assert model.call_kwargs["num_steps"] == 30
    assert model.call_kwargs["batch_size"] == 1
    assert model.call_kwargs["seed"] == 123
    assert len(model.t2i_calls) == 1
    assert model.it2i_calls == []


def test_sensenova_u1_generation_stage_uses_it2i_for_image_inputs():
    sampling = SenseNovaU1SamplingParams(
        prompt="make the sky orange",
        width=2048,
        height=2048,
        guidance_scale=3.5,
        img_cfg_scale=1.25,
        cfg_norm="channel",
        input_max_pixels=512 * 512,
        seed=11,
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=sampling.num_outputs_per_prompt,
        condition_image=[Image.new("RGBA", (64, 32), (255, 0, 0, 128))],
        image_path=None,
        extra=sampling.build_request_extra(),
        metrics=None,
        sampling_params=SimpleNamespace(enable_cache_dit=False, cache_dit_params=None),
    )
    model = _FakeSenseNovaModel()
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")

    output = stage.forward(batch, server_args=SimpleNamespace())

    assert len(output.output) == 1
    assert model.t2i_calls == []
    assert len(model.it2i_calls) == 1
    call = model.it2i_calls[0]
    assert call["tokenizer"] == "tok"
    assert call["prompt"] == "make the sky orange"
    assert call["image_size"] == (2048, 1024)
    assert call["cfg_scale"] == 3.5
    assert call["img_cfg_scale"] == 1.25
    assert call["cfg_norm"] == "channel"
    assert call["num_steps"] == 50
    assert call["seed"] == 11
    assert len(call["images"]) == 1
    assert call["images"][0].mode == "RGB"
    assert call["images"][0].size != (64, 32)


def test_sensenova_u1_it2i_preserves_input_aspect_ratio_for_output_size():
    sampling = SenseNovaU1SamplingParams(
        prompt="replace the text",
        width=1024,
        height=1024,
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=sampling.num_outputs_per_prompt,
        condition_image=[Image.new("RGB", (1600, 800))],
        image_path=None,
        extra=sampling.build_request_extra(),
        metrics=None,
        sampling_params=SimpleNamespace(enable_cache_dit=False, cache_dit_params=None),
    )
    model = _FakeSenseNovaModel()

    SenseNovaU1GenerationStage(model=model, tokenizer="tok").forward(
        batch, server_args=SimpleNamespace()
    )

    out_width, out_height = model.it2i_calls[0]["image_size"]
    assert (out_width, out_height) == (2048, 1024)
    assert out_width % 32 == 0
    assert out_height % 32 == 0
    assert abs((out_width / out_height) - 2.0) < 0.05
    assert (batch.width, batch.height) == (out_width, out_height)


def test_sensenova_u1_it2i_preserves_explicit_output_size():
    sampling = SenseNovaU1SamplingParams(
        prompt="replace the text",
        width=1024,
        height=1024,
    )
    sampling._explicit_fields = {"width", "height"}
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=sampling.num_outputs_per_prompt,
        condition_image=[Image.new("RGB", (1600, 800))],
        image_path=None,
        extra=sampling.build_request_extra(),
        metrics=None,
        sampling_params=SimpleNamespace(enable_cache_dit=False, cache_dit_params=None),
    )
    model = _FakeSenseNovaModel()

    SenseNovaU1GenerationStage(model=model, tokenizer="tok").forward(
        batch, server_args=SimpleNamespace()
    )

    assert model.it2i_calls[0]["image_size"] == (1024, 1024)
    assert (batch.width, batch.height) == (1024, 1024)


def test_sensenova_u1_generation_stage_rejects_cfg_zero_star_for_it2i():
    sampling = SenseNovaU1SamplingParams(
        prompt="edit",
        width=2048,
        height=2048,
        cfg_norm="cfg_zero_star",
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=sampling.num_outputs_per_prompt,
        condition_image=[Image.new("RGB", (64, 64))],
        image_path=None,
        extra=sampling.build_request_extra(),
        metrics=None,
        sampling_params=SimpleNamespace(enable_cache_dit=False, cache_dit_params=None),
    )

    with pytest.raises(ValueError, match="cfg_zero_star"):
        SenseNovaU1GenerationStage(
            model=_FakeSenseNovaModel(), tokenizer="tok"
        ).forward(batch, server_args=SimpleNamespace())


def test_sensenova_u1_cache_dit_preserves_config_across_sequential_outputs(
    monkeypatch,
):
    calls = _install_sensenova_cache_dit_stub(monkeypatch)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(attention_type="full_attention")],
        config=SimpleNamespace(num_hidden_layers=1),
    )
    model = SimpleNamespace(language_model=SimpleNamespace(model=transformer))
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")
    batch = _cache_dit_batch(
        guidance_scale=4.0,
        cache_dit_params={
            "Fn_compute_blocks": 3,
            "Bn_compute_blocks": 1,
            "max_warmup_steps": 2,
            "residual_diff_threshold": 0.1,
            "max_continuous_cached_steps": 4,
        },
    )

    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )

    assert len(calls["enable"]) == 1
    assert transformer._sensenova_cache_dit_native_layers is transformer.layers
    assert transformer._sensenova_cache_dit_attention_type == "full_attention"
    config = calls["enable"][0][1]
    assert config.kwargs["num_inference_steps"] == 8
    assert config.kwargs["Fn_compute_blocks"] == 3
    assert config.kwargs["Bn_compute_blocks"] == 1
    assert config.kwargs["max_warmup_steps"] == 2
    assert config.kwargs["residual_diff_threshold"] == 0.1
    assert config.kwargs["max_continuous_cached_steps"] == 4
    assert calls["enable"][0][2]["has_separate_cfg"] is True

    # A sequential n>1 request enters the generation stage once per output.
    # The second output refreshes the context instead of remounting Cache-DiT.
    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )
    assert len(calls["refresh"]) == 1
    refreshed_transformer, refreshed_steps, refreshed_config = calls["refresh"][0]
    assert refreshed_transformer is transformer
    assert refreshed_steps == 8
    assert refreshed_config is config
    assert refreshed_config.kwargs == config.kwargs

    batch.sampling_params.enable_cache_dit = False
    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )
    assert calls["disable"] == [transformer]
    assert stage._cache_dit_active_config is None
    assert not hasattr(transformer, "_sensenova_cache_dit_native_layers")
    assert not hasattr(transformer, "_sensenova_cache_dit_attention_type")


@pytest.mark.parametrize("prior_enabled", [False, True])
@pytest.mark.parametrize("disable_reason", ["explicit", "cuda_graph", "partial_cfg"])
def test_sensenova_u1_disabled_cache_ignores_params(
    monkeypatch, prior_enabled, disable_reason
):
    calls = _install_sensenova_cache_dit_stub(monkeypatch)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(attention_type="full_attention")],
        config=SimpleNamespace(num_hidden_layers=1),
    )
    stage = SenseNovaU1GenerationStage(
        model=SimpleNamespace(language_model=SimpleNamespace(model=transformer)),
        tokenizer="tok",
    )
    batch = _cache_dit_batch(guidance_scale=4.0)
    server_args = _cache_dit_server_args()
    cfg_interval = (0.0, 1.0)
    if prior_enabled:
        stage._maybe_enable_cache_dit(
            batch,
            server_args,
            cfg_interval=cfg_interval,
            guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
        )

    # A shared client config may retain knobs that SenseNova does not support.
    batch.sampling_params.cache_dit_params = {"enable_taylorseer": False}
    if disable_reason == "explicit":
        batch.sampling_params.enable_cache_dit = False
    elif disable_reason == "cuda_graph":
        server_args.enable_breakable_cuda_graph = True
    else:
        cfg_interval = (0.2, 0.8)

    stage._maybe_enable_cache_dit(
        batch,
        server_args,
        cfg_interval=cfg_interval,
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )

    assert len(calls["enable"]) == int(prior_enabled)
    assert calls["disable"] == ([transformer] if prior_enabled else [])
    assert calls["refresh"] == []
    assert stage._cache_dit_enabled is False
    assert stage._cache_dit_active_key is None
    assert stage._cache_dit_active_config is None
    assert not hasattr(transformer, "_sensenova_cache_dit_native_layers")
    assert not hasattr(transformer, "_sensenova_cache_dit_attention_type")


def test_sensenova_u1_forward_guards_on_the_request_cfg_interval(monkeypatch):
    """A timestep-gated CFG interval must reach the guard that vetoes the mount."""
    calls = _install_sensenova_cache_dit_stub(monkeypatch)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(attention_type="full_attention")],
        config=SimpleNamespace(num_hidden_layers=1),
    )
    model = _FakeSenseNovaModel()
    model.language_model = SimpleNamespace(model=transformer)
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")
    batch = SimpleNamespace(
        prompt="a mountain lake",
        width=64,
        height=64,
        guidance_scale=4.0,
        num_inference_steps=8,
        seed=7,
        num_outputs_per_prompt=1,
        extra={SENSENOVA_U1_REQUEST_EXTRA_KEY: {"cfg_interval": (0.2, 0.8)}},
        metrics=None,
        sampling_params=SimpleNamespace(enable_cache_dit=True, cache_dit_params=None),
    )

    stage.forward(batch, server_args=_cache_dit_server_args())

    assert calls == {"enable": [], "disable": [], "refresh": []}
    assert model.call_kwargs["cfg_interval"] == (0.2, 0.8)


def test_sensenova_u1_cache_dit_rolls_back_partial_mount(monkeypatch):
    mount_error = RuntimeError("cache-dit mount failed")
    calls = _install_sensenova_cache_dit_stub(monkeypatch, enable_error=mount_error)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(attention_type="full_attention")],
        config=SimpleNamespace(num_hidden_layers=1),
    )
    model = SimpleNamespace(language_model=SimpleNamespace(model=transformer))
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")
    batch = _cache_dit_batch()

    with pytest.raises(RuntimeError, match="cache-dit mount failed"):
        stage._maybe_enable_cache_dit(
            batch,
            _cache_dit_server_args(),
            cfg_interval=(0.0, 1.0),
            guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
        )

    assert calls["disable"] == [transformer]
    assert not hasattr(transformer, "_partial_cache_dit_hook")
    assert not hasattr(transformer, "_sensenova_cache_dit_native_layers")
    assert not hasattr(transformer, "_sensenova_cache_dit_attention_type")
    assert stage._cache_dit_enabled is False
    assert stage._cache_dit_active_key is None
    assert stage._cache_dit_cleanup_required is False


def test_sensenova_u1_cache_dit_failed_rollback_blocks_later_requests(monkeypatch):
    calls = _install_sensenova_cache_dit_stub(
        monkeypatch,
        enable_error=RuntimeError("cache-dit mount failed"),
        disable_error=RuntimeError("cache-dit cleanup failed"),
    )
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(attention_type="full_attention")],
        config=SimpleNamespace(num_hidden_layers=1),
    )
    model = SimpleNamespace(language_model=SimpleNamespace(model=transformer))
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")
    batch = _cache_dit_batch()

    # Preserve the original mount error even when its rollback also fails.
    with pytest.raises(RuntimeError, match="cache-dit mount failed"):
        stage._maybe_enable_cache_dit(
            batch,
            _cache_dit_server_args(),
            cfg_interval=(0.0, 1.0),
            guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
        )

    assert stage._cache_dit_enabled is False
    assert stage._cache_dit_active_key is None
    assert stage._cache_dit_cleanup_required is True
    assert not hasattr(transformer, "_sensenova_cache_dit_native_layers")
    assert not hasattr(transformer, "_sensenova_cache_dit_attention_type")

    # A later ordinary request must retry cleanup and fail closed instead of
    # reaching the early return while the transformer may still be wrapped.
    batch.sampling_params.enable_cache_dit = False
    with pytest.raises(RuntimeError, match="cache-dit cleanup failed"):
        stage._maybe_enable_cache_dit(
            batch,
            _cache_dit_server_args(),
            cfg_interval=(0.0, 1.0),
            guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
        )

    assert calls["disable"] == [transformer, transformer]
    assert stage._cache_dit_cleanup_required is True


@pytest.mark.parametrize(
    ("first_guidance_scale", "second_guidance_scale"),
    [(4.0, 1.0), (1.0, 4.0)],
)
def test_sensenova_u1_cache_dit_remounts_when_cfg_mode_changes(
    monkeypatch, first_guidance_scale, second_guidance_scale
):
    calls = _install_sensenova_cache_dit_stub(monkeypatch)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(attention_type="full_attention")],
        config=SimpleNamespace(num_hidden_layers=1),
    )
    model = SimpleNamespace(language_model=SimpleNamespace(model=transformer))
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")
    batch = _cache_dit_batch(
        guidance_scale=first_guidance_scale,
        cache_dit_params={"residual_diff_threshold": 0.1},
    )

    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )
    batch.guidance_scale = second_guidance_scale
    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )

    assert len(calls["enable"]) == 2
    assert calls["disable"] == [transformer]
    assert calls["refresh"] == []
    assert [call[2]["has_separate_cfg"] for call in calls["enable"]] == [
        first_guidance_scale > 1.0,
        second_guidance_scale > 1.0,
    ]


def test_sensenova_u1_real_cache_dit_wrapper_routes_only_denoising():
    pytest.importorskip("cache_dit")

    transformer = _CacheDitQwen3Model()
    native_layers = transformer.layers
    model = SimpleNamespace(language_model=SimpleNamespace(model=transformer))
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")
    batch = _cache_dit_batch(
        num_inference_steps=2,
        cache_dit_params={
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 2,
            "residual_diff_threshold": 0.1,
        },
    )

    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )
    try:
        inputs = torch.zeros(1, 2, 4)
        transformer(
            inputs,
            image_gen_indicators=torch.zeros(1, 2, dtype=torch.bool),
            update_cache=True,
            sensenova_marker="prefix",
        )
        transformer(
            inputs,
            image_gen_indicators=torch.ones(1, 2, dtype=torch.bool),
            update_cache=False,
            sensenova_marker="denoise",
        )

        # cache-dit patches ``layers`` only inside its forward wrapper.
        # Prefix/text selects the preserved ModuleList, while pure denoising
        # selects the real UnifiedBlocks wrapper. Both native blocks must still
        # receive SenseNova's model-specific kwargs through that wrapper.
        assert transformer.used_native_layers == [True, False]
        assert transformer.layers is native_layers
        assert hasattr(transformer, "_original_forward")
        for layer in native_layers:
            assert [call[0] for call in layer.calls] == ["prefix", "denoise"]
            denoise_kwargs = layer.calls[1][1]
            assert denoise_kwargs["exist_non_image_gen_tokens"] is False
            assert denoise_kwargs["exist_image_gen_tokens"] is True
            assert denoise_kwargs["update_cache"] is False
    finally:
        stage._unmount_cache_dit()

    assert transformer.layers is native_layers
    assert not hasattr(transformer, "_original_forward")
    assert not hasattr(transformer, "_sensenova_cache_dit_native_layers")
    assert not hasattr(transformer, "_sensenova_cache_dit_attention_type")


def test_sensenova_u1_real_cache_dit_wrapper_isolates_cfg_residuals():
    pytest.importorskip("cache_dit")

    transformer = _CacheDitQwen3Model(
        layers=[
            _CacheDitRecordingBlock(transform=lambda hidden_states: hidden_states),
            _CacheDitRecordingBlock(transform=lambda hidden_states: hidden_states * 2),
        ]
    )
    model = SimpleNamespace(language_model=SimpleNamespace(model=transformer))
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")
    batch = _cache_dit_batch(
        guidance_scale=4.0,
        num_inference_steps=2,
        cache_dit_params={
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 1,
            # Force the second visit to each branch to consume its cached
            # residual, making cross-branch state immediately observable.
            "residual_diff_threshold": 1.0,
        },
    )

    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=SenseNovaGuidanceProfile.CONDITION_UNCONDITIONAL,
    )
    try:
        indicators = torch.ones(1, 1, dtype=torch.bool)

        def denoise(value, marker):
            return transformer(
                torch.full((1, 1, 1), value),
                image_gen_indicators=indicators,
                update_cache=False,
                sensenova_marker=marker,
            )

        # First pair populates residuals: +1 for condition, +10 for uncondition.
        torch.testing.assert_close(denoise(1.0, "condition-0"), torch.tensor([[[2.0]]]))
        torch.testing.assert_close(
            denoise(10.0, "uncondition-0"), torch.tensor([[[20.0]]])
        )

        # Each second-pass result must use its own branch's previous residual.
        torch.testing.assert_close(denoise(2.0, "condition-1"), torch.tensor([[[3.0]]]))
        torch.testing.assert_close(
            denoise(20.0, "uncondition-1"), torch.tensor([[[30.0]]])
        )
    finally:
        stage._unmount_cache_dit()


def test_sensenova_u1_invalid_output_count_does_not_mount_cache_dit(monkeypatch):
    calls = _install_sensenova_cache_dit_stub(monkeypatch)
    stage = SenseNovaU1GenerationStage(model=_FakeSenseNovaModel(), tokenizer="tok")
    batch = SimpleNamespace(num_outputs_per_prompt=2)

    with pytest.raises(ValueError, match="expects output expansion"):
        stage.forward(batch, server_args=SimpleNamespace())

    assert calls == {"enable": [], "disable": [], "refresh": []}


def test_sensenova_u1_generation_stage_passes_dynamic_batch_inputs():
    sampling = SenseNovaU1SamplingParams(
        prompt="first prompt",
        width=1024,
        height=1024,
        guidance_scale=4.0,
        num_inference_steps=5,
        seed=7,
    )
    batch = SimpleNamespace(
        prompt=["first prompt", "a longer second prompt"],
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=1,
        extra={
            **sampling.build_request_extra(),
            "dynamic_batch_seeds": [7, 19],
        },
        metrics=None,
        sampling_params=SimpleNamespace(enable_cache_dit=False, cache_dit_params=None),
    )
    model = _FakeSenseNovaModel()
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")

    output = stage.forward(batch, server_args=_cache_dit_server_args())

    assert len(output.output) == 2
    assert model.call_kwargs["prompt"] == [
        "first prompt",
        "a longer second prompt",
    ]
    assert model.call_kwargs["batch_size"] == 2
    assert model.call_kwargs["seed"] == [7, 19]


def test_sensenova_u1_generation_stage_rejects_batched_think_mode():
    sampling = SenseNovaU1SamplingParams(
        prompt="first prompt",
        width=1024,
        height=1024,
        think_mode=True,
    )
    batch = SimpleNamespace(
        prompt=["first prompt", "second prompt"],
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=1,
        extra={
            **sampling.build_request_extra(),
            "dynamic_batch_seeds": [7, 19],
        },
        metrics=None,
        sampling_params=SimpleNamespace(enable_cache_dit=False, cache_dit_params=None),
    )

    with pytest.raises(ValueError, match="think_mode"):
        SenseNovaU1GenerationStage(
            model=_FakeSenseNovaModel(), tokenizer="tok"
        ).forward(batch, server_args=_cache_dit_server_args())


def test_sensenova_u1_multi_output_request_expands_before_generation_stage():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
        # Keep the fake model off the Cache-DiT path regardless of SGLANG_* env.
        enable_cache_dit=False,
    )
    batch = Req(
        request_id="req-0",
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=42,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
        output_file_name="sample.png",
    )
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    input_stage = InputValidationStage()
    stage = SenseNovaU1GenerationStage(model=_FakeSenseNovaModel(), tokenizer="tok")
    batch.metrics.record_stage("InputValidationStage", 0.125)
    batch.metrics.record_memory_snapshot(
        "after_validation",
        MemorySnapshot(
            allocated_mb=100.0,
            reserved_mb=200.0,
            peak_allocated_mb=300.0,
            peak_reserved_mb=400.0,
        ),
    )

    expanded = list(input_stage.iter_sequential_requests(batch, server_args))

    assert [req.num_outputs_per_prompt for req in expanded] == [1, 1]
    assert [req.seed for req in expanded] == [42, 43]
    assert [req.request_id for req in expanded] == ["req-0:0", "req-0:1"]
    assert [req.output_file_name for req in expanded] == [
        "sample_0.png",
        "sample_1.png",
    ]
    assert [req.metrics.request_id for req in expanded] == ["req-0:0", "req-0:1"]
    assert all(req.trace_ctx is batch.trace_ctx for req in expanded)
    assert all(req.metrics is not batch.metrics for req in expanded)
    assert expanded[0].metrics is not expanded[1].metrics
    assert all(
        req.metrics.stages == {"InputValidationStage": 125.0} for req in expanded
    )
    assert all(
        req.metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0
        for req in expanded
    )
    assert (
        expanded[0].metrics.memory_snapshots["after_validation"]
        is not expanded[1].metrics.memory_snapshots["after_validation"]
    )

    expanded[0].metrics.record_stage("child-only", 0.5)
    expanded[0].metrics.memory_snapshots["after_validation"].peak_reserved_mb = 999.0
    assert "child-only" not in expanded[1].metrics.stages
    assert "child-only" not in batch.metrics.stages
    assert (
        expanded[1].metrics.memory_snapshots["after_validation"].peak_reserved_mb
        == 400.0
    )
    assert batch.metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0

    for req in expanded:
        output = stage.forward(req, server_args=_cache_dit_server_args())
        assert len(output.output) == 1


def test_sensenova_u1_multi_output_rejects_short_seed_list():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
        seed=[7],
    )
    batch = Req(
        request_id="req-0",
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
        output_file_name="sample.png",
    )
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())

    with pytest.raises(ValueError, match="seed list length"):
        list(InputValidationStage().iter_sequential_requests(batch, server_args))


def _make_sensenova_u1_sequential_entrypoint(*, fail=False, fail_request_ids=None):
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
        save_output=False,
        suppress_logs=True,
    )
    trace_ctx = _RecordingTraceContext()
    batch = Req(
        request_id="req-0",
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        seed=42,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
        output_file_name="sample.png",
        trace_ctx=trace_ctx,
    )
    server_args = SimpleNamespace(
        pipeline_config=SenseNovaU1PipelineConfig(),
        disable_conditioning_cache=False,
        conditioning_cache_max_size_mb=512,
        use_fsdp_inference=False,
    )
    pipeline = _SequentialTestPipeline(
        server_args, fail=fail, fail_request_ids=fail_request_ids
    )
    worker = GPUWorker.__new__(GPUWorker)
    worker.pipeline = pipeline
    worker.server_args = server_args
    worker.is_output_rank = True
    worker._runtime_peak_reserved_mb = 0.0
    worker._release_warmup_pool_before_serving = False
    worker._realtime_sessions = SimpleNamespace(attach=lambda _req: None)
    return batch, trace_ctx, pipeline.executor, _WorkerBackedSchedulerClient(worker)


def _force_cpu_entrypoint(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_mps", lambda: False)
    monkeypatch.setattr(current_platform, "is_npu", lambda: False)
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.utils.get_global_server_args",
        lambda: SimpleNamespace(batching_max_size=1),
    )


def test_sensenova_u1_multi_output_entrypoint_success(monkeypatch):
    _force_cpu_entrypoint(monkeypatch)
    batch, trace_ctx, executor, scheduler_client = (
        _make_sensenova_u1_sequential_entrypoint()
    )

    paths, result = asyncio.run(process_generation_batch(scheduler_client, batch))

    assert paths == ["sample_0.png", "sample_1.png"]
    assert result.error is None
    assert [req.request_id for req in executor.executed_requests] == [
        "req-0:0",
        "req-0:1",
    ]
    assert [req.seed for req in executor.executed_requests] == [42, 43]
    assert result.metrics_list is not None
    assert [metrics.request_id for metrics in result.metrics_list] == [
        "req-0:0",
        "req-0:1",
    ]
    assert all(
        "InputValidationStage" in metrics.stages
        and "PipelineExecutor.sequential_wait" in metrics.stages
        and metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0
        for metrics in result.metrics_list
    )
    assert all(req.trace_ctx is trace_ctx for req in executor.executed_requests)
    assert trace_ctx.started_slices == [("gpu_forward", 2)]
    assert trace_ctx.finished_slices == [("gpu_forward", 2)]
    assert trace_ctx.finish_count == 1


def test_sensenova_u1_multi_output_entrypoint_failure(monkeypatch):
    _force_cpu_entrypoint(monkeypatch)
    batch, trace_ctx, executor, scheduler_client = (
        _make_sensenova_u1_sequential_entrypoint(fail=True)
    )

    with pytest.raises(RuntimeError, match="generation failed for req-0:0"):
        asyncio.run(process_generation_batch(scheduler_client, batch))

    assert [req.request_id for req in executor.executed_requests] == [
        "req-0:0",
        "req-0:1",
    ]
    assert all(
        "InputValidationStage" in req.metrics.stages
        and "PipelineExecutor.sequential_wait" in req.metrics.stages
        and req.metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0
        for req in executor.executed_requests
    )
    assert all(req.trace_ctx is trace_ctx for req in executor.executed_requests)
    assert trace_ctx.started_slices == [("gpu_forward", 2)]
    assert trace_ctx.finished_slices == [("gpu_forward", 2)]
    assert trace_ctx.finish_count == 1


@pytest.mark.parametrize("failed_request_id", ["req-0:0", "req-0:1"])
def test_sensenova_u1_multi_output_entrypoint_mixed_failure_fails_parent(
    monkeypatch, failed_request_id
):
    _force_cpu_entrypoint(monkeypatch)
    batch, trace_ctx, executor, scheduler_client = (
        _make_sensenova_u1_sequential_entrypoint(fail_request_ids={failed_request_id})
    )

    with pytest.raises(
        RuntimeError, match=f"generation failed for {failed_request_id}"
    ):
        asyncio.run(process_generation_batch(scheduler_client, batch))

    assert [req.request_id for req in executor.executed_requests] == [
        "req-0:0",
        "req-0:1",
    ]
    assert all(req.trace_ctx is trace_ctx for req in executor.executed_requests)
    assert trace_ctx.started_slices == [("gpu_forward", 2)]
    assert trace_ctx.finished_slices == [("gpu_forward", 2)]
    assert trace_ctx.finish_count == 1


_GUIDANCE_SCALE_MATRIX = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0]


def _expected_it2i_branches(cfg_scale, img_cfg_scale, use_cfg):
    """Independent oracle for the public IT2I guidance semantics."""
    if not use_cfg or (cfg_scale == 1 and img_cfg_scale == 1):
        return ("condition",)
    if img_cfg_scale == 1:
        return ("condition", "image_condition")
    if cfg_scale == img_cfg_scale:
        return ("condition", "uncondition")
    return ("condition", "image_condition", "uncondition")


@pytest.mark.parametrize(
    ("cfg_interval", "cfg_active_by_step"),
    [
        ((0.0, 1.0), (True, True, True, True, True)),
        ((0.25, 0.75), (False, False, True, True, False)),
        # IT2I's lo == 0 escape deliberately makes every step active.
        ((0.0, 0.5), (True, True, True, True, True)),
    ],
)
@pytest.mark.parametrize(
    ("cfg_scale", "img_cfg_scale"),
    list(itertools.product(_GUIDANCE_SCALE_MATRIX, repeat=2)),
)
def test_sensenova_it2i_guidance_profile_drives_real_loop_branches(
    monkeypatch,
    cfg_scale,
    img_cfg_scale,
    cfg_interval,
    cfg_active_by_step,
):
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
        NEOChatModel,
    )

    forward_globals = NEOChatModel.it2i_generate.__wrapped__.__globals__
    monkeypatch.setitem(forward_globals, "prepare_flash_kv_cache", lambda *a, **k: None)
    monkeypatch.setitem(forward_globals, "clear_flash_kv_cache", lambda *a: None)
    monkeypatch.setitem(
        forward_globals,
        "load_image_native",
        lambda *a, **k: (torch.zeros(1, 3), torch.tensor([[1, 1]])),
    )

    branch_ids = {
        "condition": 1.0,
        "image_condition": 2.0,
        "uncondition": 3.0,
    }
    branch_names = {int(value): key for key, value in branch_ids.items()}
    calls = []

    def build_query(question, *, system_message=None, append_text=None):
        if system_message is not None:
            return "condition"
        return "image_condition" if question else "uncondition"

    def build_inputs(_tokenizer, query, *_args):
        inputs = torch.full((1, 1, 3), branch_ids[query])
        indexes = torch.zeros(3, 1, dtype=torch.long)
        return inputs, indexes, None

    def prefix_forward(inputs, *_args):
        branch = branch_names[int(inputs[0, 0, 0].item())]
        return SimpleNamespace(branch=branch, layers=[]), torch.zeros(1)

    def predict(_image_embeds, _indexes, _mask, past_key_values, t, *_args, **_kwargs):
        calls.append(
            (
                round(float(t), 6),
                None if past_key_values is None else past_key_values.branch,
            )
        )
        return torch.zeros_like(_image_embeds)

    class _ZeroEmbedder(torch.nn.Module):
        def forward(self, values):
            return torch.zeros(values.numel(), 3, device=values.device)

    model = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(),
        patch_size=1,
        downsample_ratio=1,
        noise_scale=0.0,
        noise_scale_mode="constant",
        noise_scale_max_value=1.0,
        add_noise_scale_embedding=False,
        fm_modules={"timestep_embedder": _ZeroEmbedder()},
        _notify_layer_offload_phase=lambda _phase: None,
        _build_t2i_query=build_query,
        _build_it2i_inputs=build_inputs,
        _build_t2i_image_indexes=lambda h, w, *a, **k: torch.zeros(3, h * w),
        _it2i_prefix_forward=prefix_forward,
        patchify=lambda x, *a, **k: x.flatten(2).transpose(1, 2).contiguous(),
        extract_feature=lambda x, **k: torch.zeros_like(x),
        _t2i_predict_v=predict,
        unpatchify=lambda z, patch, h, w: z.transpose(1, 2).reshape(-1, 3, h, w),
    )
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _token: 0)

    output = NEOChatModel.it2i_generate(
        model,
        tokenizer,
        "prompt",
        [Image.new("RGB", (2, 2))],
        image_size=(2, 2),
        num_steps=len(cfg_active_by_step),
        cfg_scale=cfg_scale,
        img_cfg_scale=img_cfg_scale,
        cfg_interval=cfg_interval,
        enable_timestep_shift=False,
    )

    expected_by_step = [
        _expected_it2i_branches(cfg_scale, img_cfg_scale, use_cfg)
        for use_cfg in cfg_active_by_step
    ]
    actual_by_step = [
        tuple(branch for _, branch in grouped_calls)
        for _, grouped_calls in itertools.groupby(calls, key=lambda call: call[0])
    ]
    assert actual_by_step == expected_by_step
    assert output.shape == (1, 3, 2, 2)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("num_steps", [0, 1, 4])
@pytest.mark.parametrize("cfg_scale", [1.0, 4.0])
def test_sensenova_t2i_reuses_request_noise_embedding(
    monkeypatch, enabled, num_steps, cfg_scale
):
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_fm_modules import (
        TimestepEmbedder,
    )
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
        NEOChatModel,
    )

    # Exercise the real generation loop and embedders without a checkpoint.
    forward_globals = NEOChatModel.t2i_generate.__wrapped__.__globals__
    monkeypatch.setitem(forward_globals, "prepare_flash_kv_cache", lambda *a, **k: None)
    monkeypatch.setitem(forward_globals, "clear_flash_kv_cache", lambda *a: None)
    timestep_embedder = TimestepEmbedder(3).eval()
    noise_embedder = TimestepEmbedder(3).eval()
    calls = []
    predictions = []

    def record_noise(module, args, output):
        calls.append(args[0].clone())

    hook = noise_embedder.register_forward_hook(record_noise)

    def predict(image_embeds, *args, **kwargs):
        predictions.append(image_embeds.clone())
        return image_embeds

    model = SimpleNamespace(
        concat_time_token_num=0,
        downsample_ratio=1,
        patch_size=1,
        config=SimpleNamespace(),
        noise_scale=0.5,
        noise_scale_mode="constant",
        noise_scale_max_value=2.0,
        add_noise_scale_embedding=enabled,
        fm_modules={
            "timestep_embedder": timestep_embedder,
            "noise_scale_embedder": noise_embedder,
        },
        _notify_layer_offload_phase=lambda phase: None,
        _build_t2i_query=lambda *a, **k: "query",
        _build_t2i_text_inputs=lambda *a: (
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(3, 1, dtype=torch.long),
            None,
        ),
        _build_t2i_image_indexes=lambda h, w, *a, **k: torch.zeros(3, h * w),
        _t2i_prefix_forward=lambda *a: (SimpleNamespace(layers=[]), torch.zeros(1)),
        patchify=lambda x, *a, **k: x.flatten(2).transpose(1, 2).contiguous(),
        extract_feature=lambda x, **k: torch.zeros_like(x),
        _t2i_predict_v=predict,
        unpatchify=lambda z, patch, h, w: z.transpose(1, 2).reshape(-1, 3, h, w),
    )
    try:
        # Change both shape and noise scale to detect stale cross-request reuse.
        for width, scale in [(2, 0.5), (3, 1.0)]:
            model.noise_scale = scale
            calls.clear()
            predictions.clear()
            output = NEOChatModel.t2i_generate(
                model,
                None,
                "prompt",
                image_size=(width, 2),
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                enable_timestep_shift=False,
                batch_size=2,
            )
            assert output.shape == (2, 3, 2, width)
            assert len(calls) == int(enabled and num_steps > 0)
            branches = 2 if cfg_scale > 1 else 1
            assert len(predictions) == num_steps * branches
            # Compare each step with the original per-step computation.
            with torch.no_grad():
                for step, t in enumerate(torch.linspace(0, 1, num_steps + 1)[:-1]):
                    expanded = t.expand(2 * 2 * width)
                    expected = timestep_embedder(expanded).view(2, 2 * width, 3)
                    if enabled:
                        expected += noise_embedder(
                            torch.full_like(expanded, scale / 2.0)
                        ).view(2, 2 * width, 3)
                    for branch in range(branches):
                        torch.testing.assert_close(
                            predictions[step * branches + branch],
                            expected,
                            rtol=0,
                            atol=0,
                        )
    finally:
        hook.remove()


@pytest.mark.parametrize("explicit_first", [False, True])
def test_sensenova_cache_dit_effective_defaults_reuse_mount(
    monkeypatch, explicit_first
):
    calls = _install_sensenova_cache_dit_stub(monkeypatch)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(attention_type="full_attention")],
        config=SimpleNamespace(num_hidden_layers=1),
    )
    stage = SenseNovaU1GenerationStage(
        model=SimpleNamespace(language_model=SimpleNamespace(model=transformer)),
        tokenizer="tok",
    )
    # Pin the environment independently of the developer's cache settings.
    monkeypatch.setenv("SGLANG_CACHE_DIT_RDT", "0.24")
    explicit = {"residual_diff_threshold": 0.24}
    batch = _cache_dit_batch(
        cache_dit_params=explicit if explicit_first else None,
    )
    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )
    batch.sampling_params.cache_dit_params = None if explicit_first else explicit
    batch.num_inference_steps = 12
    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )
    assert len(calls["enable"]) == 1
    assert calls["enable"][0][1].kwargs["residual_diff_threshold"] == 0.24
    assert calls["refresh"] == [(transformer, 12, calls["enable"][0][1])]
    assert calls["disable"] == []

    batch.sampling_params.cache_dit_params = {"residual_diff_threshold": 0.1}
    stage._maybe_enable_cache_dit(
        batch,
        _cache_dit_server_args(),
        cfg_interval=(0.0, 1.0),
        guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
    )
    assert len(calls["enable"]) == 2
    assert calls["enable"][1][1].kwargs["residual_diff_threshold"] == 0.1
    assert calls["disable"] == [transformer]


@pytest.mark.parametrize(
    "attention_types", [[], ["full_attention", "sliding_attention"], [None]]
)
def test_sensenova_cache_dit_rejects_invalid_attention_before_mount(
    monkeypatch, attention_types
):
    calls = _install_sensenova_cache_dit_stub(monkeypatch)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(attention_type=value) for value in attention_types],
        config=SimpleNamespace(num_hidden_layers=len(attention_types)),
    )
    stage = SenseNovaU1GenerationStage(
        model=SimpleNamespace(language_model=SimpleNamespace(model=transformer)),
        tokenizer="tok",
    )
    batch = _cache_dit_batch()
    with pytest.raises(ValueError, match="attention type"):
        stage._maybe_enable_cache_dit(
            batch,
            _cache_dit_server_args(),
            cfg_interval=(0.0, 1.0),
            guidance_profile=_t2i_guidance_profile(batch.guidance_scale),
        )
    assert calls == {"enable": [], "disable": [], "refresh": []}
    assert not hasattr(transformer, "_sensenova_cache_dit_native_layers")
    assert not hasattr(transformer, "_sensenova_cache_dit_attention_type")
