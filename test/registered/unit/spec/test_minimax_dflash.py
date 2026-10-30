"""MiniMax DFLASH admission, draft backend selection and target feature taps."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.arg_groups.model_overrides.minimax_m3 import _minimax_m3_overrides
from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.arg_groups.serving_hook import handle_missing_default_values
from sglang.srt.arg_groups.speculative_hook import (
    _resolve_dflash_draft_attention_backend,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.models.minimax_m3 import MiniMaxM3Model
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.srt.runtime_context import get_context, get_platform, override_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NVFP4 = SimpleNamespace(
    architectures=["MiniMaxM3SparseForConditionalGeneration"],
    quantization_config={"quant_method": "minimax_m3_nvfp4"},
)


@pytest.fixture(autouse=True)
def blackwell_static_verify(monkeypatch):
    monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", "static")
    with override_platform(is_blackwell=True, is_xpu=False, is_hip=False):
        yield


def _args(**overrides):
    fields = dict(
        quantization=None,
        _quantization_explicitly_unset=False,
        speculative_algorithm="DFLASH",
        moe_a2a_backend="none",
        moe_runner_backend="flashinfer_trtllm_routed",
        page_size=None,
        kv_cache_dtype="auto",
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        pp_size=1,
        disaggregation_mode="null",
        enable_hierarchical_cache=False,
        speculative_draft_kv_cache_dtype=None,
        speculative_draft_attention_backend=None,
    )
    return SimpleNamespace(**(fields | overrides))


@pytest.mark.parametrize("draft_backend", [None, "fa4"])
def test_dflash_preserves_target_backends_and_defers_draft_backend(draft_backend):
    overrides = _minimax_m3_overrides(
        _args(speculative_draft_attention_backend=draft_backend), NVFP4
    )
    assert overrides["speculative_draft_kv_cache_dtype"] == "bf16"
    assert "speculative_draft_attention_backend" not in overrides
    assert overrides["attention_backend"] == "trtllm_mha"
    assert overrides["kv_cache_dtype"] == "fp8_e4m3"
    assert "moe_a2a_backend" not in overrides


@pytest.mark.parametrize(
    "explicit_backend,expected", [(None, "generic_default"), ("fa4", "fa4")]
)
def test_dflash_draft_backend_uses_generic_attention_policy(explicit_backend, expected):
    if expected == "generic_default":
        # A non-causal draft falls back to fa4 on SM100 and to flashinfer elsewhere.
        expected = "fa4" if get_platform().is_sm100 else "flashinfer"
    draft_config = PretrainedConfig(
        num_hidden_layers=2, is_causal=False, layer_types=["full_attention"] * 2
    )
    with (
        patch(
            "sglang.srt.arg_groups.speculative_hook.current_platform.is_out_of_tree",
            return_value=False,
        ),
        patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=draft_config,
        ),
    ):
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            speculative_algorithm="DFLASH",
            speculative_draft_model_path="draft",
            speculative_draft_attention_backend=explicit_backend,
        )
        handle_missing_default_values(args)
        declare_resolution(args, "minimax", **_minimax_m3_overrides(args, NVFP4))
        _resolve_dflash_draft_attention_backend(args)
    assert resolution_result(args, "speculative_draft_attention_backend") == expected


@pytest.mark.parametrize("algorithm", ["DSPARK", "DFLASH"])
def test_both_algorithms_require_pp_one(algorithm):
    with pytest.raises(ValueError, match="PP=1"):
        _minimax_m3_overrides(_args(speculative_algorithm=algorithm, pp_size=2), NVFP4)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_dflash_colocation_guard_preserves_dspark_disaggregation(mode):
    overrides = _minimax_m3_overrides(
        _args(speculative_algorithm="DSPARK", disaggregation_mode=mode), NVFP4
    )
    assert overrides["speculative_draft_attention_backend"] == "trtllm_mha"
    with pytest.raises(ValueError, match="DFLASH.*colocated"):
        _minimax_m3_overrides(_args(disaggregation_mode=mode), NVFP4)


@pytest.mark.parametrize("algorithm", ["DSPARK", "DFLASH"])
def test_both_algorithms_admit_hicache(algorithm):
    overrides = _minimax_m3_overrides(
        _args(speculative_algorithm=algorithm, enable_hierarchical_cache=True), NVFP4
    )
    assert overrides["speculative_draft_kv_cache_dtype"] == "bf16"
    assert overrides["page_size"] == 128


def test_dflash_hicache_restores_draft_kv_through_a_sidecar():
    # A host restore that skipped the draft pool would leave stale draft KV:
    # serving stays correct but the accept length silently drops.
    draft_pool = object()
    draft_runner = SimpleNamespace(
        token_to_kv_pool=draft_pool,
        # A draft with its own KV pool, not fused into the target's.
        runner_ids=SimpleNamespace(is_fused_draft=False),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["MiniMaxDFlash2DraftModel"]),
            num_nextn_predict_layers=0,
        ),
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DFLASH)
        ),
        draft_worker=SimpleNamespace(model_runner=draft_runner),
    )
    worker._draft_model_runners = lambda: BaseSpecWorker._draft_model_runners(worker)
    with get_context().override_server_args(enable_hierarchical_cache=True):
        plan = BaseSpecWorker._build_hicache_draft_plan(worker)
    assert plan.mode == HiCacheDraftMode.SIDECAR
    assert plan.device_pools == (draft_pool,)


def test_dflash_requires_blackwell_and_static_verification(monkeypatch):
    with (
        override_platform(is_blackwell=False),
        pytest.raises(ValueError, match="Blackwell"),
    ):
        _minimax_m3_overrides(_args(), NVFP4)
    monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", "compact")
    with pytest.raises(ValueError, match="SGLANG_RAGGED_VERIFY_MODE=static"):
        _minimax_m3_overrides(_args(), NVFP4)


@pytest.mark.parametrize("algorithm", ["EAGLE3", "NGRAM"])
def test_dflash_admission_does_not_enable_other_algorithms(algorithm):
    with pytest.raises(ValueError, match="only supports DSPARK or DFLASH"):
        _minimax_m3_overrides(_args(speculative_algorithm=algorithm), NVFP4)


class _CaptureLayer(nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def forward(
        self, *, hidden_states, forward_batch, captured_last_layer_outputs, **kwargs
    ):
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.capture(hidden_states)
        return residual_batch.set_written(
            hidden_states + 1 + self.offset, forward_batch
        )


class _OffsetNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states + 100
        return hidden_states + residual + 100, residual


def _capture_model():
    model = object.__new__(MiniMaxM3Model)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.layers = nn.ModuleList(
        [_CaptureLayer(offset=10), _CaptureLayer(), _CaptureLayer()]
    )
    model.norm = _OffsetNorm()
    model.finalize_norm = None
    model.start_layer, model.end_layer = 0, 3
    model.layers_to_capture = []
    wrapper = object.__new__(MiniMaxM3SparseForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.pp_group, wrapper.model = model.pp_group, model
    return wrapper


def test_dflash_captures_selected_layer_outputs_before_final_norm():
    wrapper = _capture_model()
    wrapper.set_dflash_layers_to_capture([0, 2])
    assert wrapper.capture_aux_hidden_states
    with patch(
        "sglang.srt.models.minimax_m3.check_cuda_graph_backend", return_value=True
    ):
        output, captured = wrapper.model(
            torch.tensor([0]),
            torch.tensor([0]),
            SimpleNamespace(can_run_tbo=False),
            input_embeds=torch.zeros(1, 2),
        )
    assert len(captured) == 2
    torch.testing.assert_close(captured[0], torch.full((1, 2), 11.0))
    torch.testing.assert_close(captured[1], torch.full((1, 2), 13.0))
    torch.testing.assert_close(output, torch.full((1, 2), 113.0))


@pytest.mark.parametrize("ids", [[], [2, 0], [0, 0], [-1], [3]])
def test_dflash_rejects_invalid_taps_without_changing_capture(ids):
    wrapper = _capture_model()
    wrapper.set_dflash_layers_to_capture([0, 2])
    with pytest.raises(ValueError):
        wrapper.set_dflash_layers_to_capture(ids)
    assert wrapper.model.layers_to_capture == [1, 3]
    assert [layer._is_layer_to_capture for layer in wrapper.model.layers] == [
        False,
        True,
        False,
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
