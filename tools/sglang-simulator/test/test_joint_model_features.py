"""Actual native feature-validator code under a function-local target view."""

import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from sglang_simulator.compat import resolve_simulator_server_args
from sglang_simulator.simulation.sglang.dspark_args import (
    install_cpu_joint_model_features,
)


@pytest.fixture
def native(monkeypatch, tmp_path):
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/arg_groups/deepseek_v4_hook.py"
    )
    parsed = ast.parse(source.read_bytes())
    function = next(
        n
        for n in parsed.body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_deepseek_v41_features"
    )
    module = ModuleType("sglang.srt.arg_groups.deepseek_v4_hook")
    physical = SimpleNamespace(is_hip=False, is_cuda=False)
    module.get_platform = lambda: physical
    module.resolving_view = lambda args: args
    module.model_config_of = lambda args: SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41")
    )
    module.attn_dp_enabled_of = lambda args: args.attn_dp_size > 1
    module.Backend = SimpleNamespace(DISABLED="disabled", BREAKABLE="breakable")
    module.declare_resolution = lambda *_a, **_kw: pytest.fail("unexpected declaration")
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        function,
                    ],
                    type_ignores=[],
                )
            ),
            str(source),
            "exec",
        ),
        vars(module),
    )
    original = module.validate_deepseek_v41_features
    monkeypatch.setitem(sys.modules, module.__name__, module)
    gate = ModuleType("sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate")
    gate.is_unified_kv_triton = lambda: False
    monkeypatch.setitem(sys.modules, gate.__name__, gate)
    config = tmp_path / "joint.json"
    value = dict(
        platform=dict(accelerator=dict(name="mi355x")),
        predictor=dict(
            name="infercast_dspark",
            cost_mode="HYBRID",
            progress_kind="source_golden351",
            native_joint_source_binding={"root": "selection-only-not-attestation"},
            native_analytic_model={"EXTEND": {}, "DECODE": {}},
        ),
    )
    config.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config))
    monkeypatch.setenv("SGLANG_SIMULATOR_BOOTSTRAP", "1")
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")
    return SimpleNamespace(
        module=module, original=original, physical=physical, config=config, value=value
    )


def arguments(**changes):
    values = dict(
        device="cpu",
        tp_size=2,
        dp_size=1,
        attn_dp_size=2,
        attn_cp_size=1,
        dcp_size=1,
        ep_size=1,
        pp_size=1,
        nnodes=1,
        moe_a2a_backend="none",
        speculative_algorithm="DSPARK",
        enable_dp_lm_head=True,
        enable_dp_attention_local_control_broadcast=True,
        enable_decoder_swa_bounded_replay=True,
        enable_encoder_swa_bounded_replay=False,
        dsv4_attn_backend="flashmla",
        enable_hisparse=False,
        enable_two_batch_overlap=False,
        disaggregation_mode="null",
        enable_prefill_cp=False,
        cp_strategy="interleave",
        enable_return_hidden_states=False,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend="disabled", max_seq_len=None)
        ),
    )
    return SimpleNamespace(**{**values, **changes})


def test_actual_native_cpu_rejection_then_public_resolution_preserves_flags_and_platform(
    native,
):
    args = arguments()
    before = vars(args).copy()
    platform_function = native.module.get_platform
    with pytest.raises(ValueError, match="DP attention"):
        native.original(args)
    args.resolve_once = lambda: native.module.validate_deepseek_v41_features(args)
    resolve_simulator_server_args(args)
    assert {k: v for k, v in vars(args).items() if k != "resolve_once"} == before
    assert args.enable_decoder_swa_bounded_replay is True
    assert native.module.get_platform is platform_function
    assert native.module.get_platform() is native.physical
    assert not native.physical.is_hip and not native.physical.is_cuda
    assert native.original.__globals__["get_platform"] is platform_function
    assert install_cpu_joint_model_features(args) is True
    installed = native.module.validate_deepseek_v41_features
    assert install_cpu_joint_model_features(args) is True
    assert native.module.validate_deepseek_v41_features is installed


@pytest.mark.parametrize(
    "changes",
    [
        {"tp_size": 4},
        {"dp_size": 2},
        {"attn_dp_size": True},
        {"attn_cp_size": 2},
        {"pp_size": 2},
        {"moe_a2a_backend": "mori"},
        {"device": "cuda"},
        {"enable_dp_lm_head": False},
        {"enable_dp_attention_local_control_broadcast": False},
        {"enable_encoder_swa_bounded_replay": True},
    ],
)
def test_unrepresented_domain_keeps_original_native_failure(native, changes):
    args = arguments(**changes)
    install_cpu_joint_model_features(args)
    try:
        native.original(args)
    except (ValueError, AttributeError) as expected:
        with pytest.raises(type(expected), match=str(expected)):
            native.module.validate_deepseek_v41_features(args)
    else:
        assert native.module.validate_deepseek_v41_features(args) is None
    assert native.module.get_platform() is native.physical


@pytest.mark.parametrize(
    "changes",
    [
        {"enable_hisparse": True},
        {"enable_two_batch_overlap": True},
        {"speculative_algorithm": "EAGLE"},
        {
            "cuda_graph_config": SimpleNamespace(
                prefill=SimpleNamespace(backend="full", max_seq_len=16384)
            )
        },
    ],
)
def test_modeled_feature_view_does_not_bypass_other_native_errors(native, changes):
    args = arguments(**changes)
    install_cpu_joint_model_features(args)
    with pytest.raises(ValueError):
        native.module.validate_deepseek_v41_features(args)
    assert args.enable_decoder_swa_bounded_replay is True
    assert native.module.get_platform() is native.physical


@pytest.mark.parametrize("kind", ["target", "source", "mode", "disabled", "default"])
def test_wrong_target_selection_flags_or_default_use_original(
    native, monkeypatch, kind
):
    args = arguments()
    install_cpu_joint_model_features(args)
    if kind == "target":
        native.value["platform"]["accelerator"]["name"] = "other"
    if kind == "source":
        native.value["predictor"]["native_joint_source_binding"] = None
    if kind == "mode":
        native.value["predictor"]["cost_mode"] = "SILICON"
    if kind == "disabled":
        monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "0")
    if kind == "default":
        native.value["predictor"] = {}
    native.config.write_text(json.dumps(native.value), encoding="utf-8")
    install_cpu_joint_model_features(args)
    with pytest.raises(ValueError, match="DP attention"):
        native.module.validate_deepseek_v41_features(args)
    assert native.module.get_platform() is native.physical


def test_existing_false_or_auto_flag_is_not_reinterpreted(native):
    args = arguments(enable_decoder_swa_bounded_replay=False)
    install_cpu_joint_model_features(args)
    assert native.module.validate_deepseek_v41_features(args) is None
    args.enable_decoder_swa_bounded_replay = None
    # The original CPU automatic-resolution branch remains authoritative.
    native.module.declare_resolution = lambda *_args, **_kw: (_ for _ in ()).throw(
        RuntimeError("original auto branch")
    )
    native.module.logger = SimpleNamespace(info=lambda *_a: None)
    with pytest.raises(RuntimeError, match="original auto branch"):
        native.module.validate_deepseek_v41_features(args)


def test_function_local_view_preserves_defaults_kwdefaults_closure_and_exception(
    native,
):
    failure = RuntimeError("native failure identity")
    marker = object()

    def make():
        retained = marker

        def validate(args, default=marker, *, keyword=marker):
            assert retained is marker and default is marker and keyword is marker
            raise failure

        return validate

    original = make()
    native.module.validate_deepseek_v41_features = original
    install_cpu_joint_model_features(arguments())
    with pytest.raises(RuntimeError) as caught:
        native.module.validate_deepseek_v41_features(arguments())
    assert caught.value is failure
    assert native.module.get_platform() is native.physical
