"""CPU argument compatibility only; no device execution or performance claims."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from sglang_simulator.simulation.sglang.dspark_args import install_cpu_dspark_args


@pytest.fixture
def native(monkeypatch):
    module = ModuleType("sglang.srt.arg_groups.speculative_hook")
    module.calls = []
    module.declarations = []

    def original(args):
        module.calls.append(args)
        return "native result"

    def resolving(args):
        return SimpleNamespace(**{**vars(args), **getattr(args, "_overlay", {})})

    def declare(args, source, **fields):
        module.declarations.append((args, source, fields))
        args._overlay = {**getattr(args, "_overlay", {}), **fields}

    module._handle_dspark = original
    module.resolving_view = resolving
    module.declare_resolution = declare
    module.envs = SimpleNamespace(
        SGLANG_RAGGED_VERIFY_MODE=SimpleNamespace(
            get=lambda: os.getenv("SGLANG_RAGGED_VERIFY_MODE", "static")
        ),
        SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=SimpleNamespace(
            get=lambda: (
                os.getenv("SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION", "0") == "1"
            )
        ),
        SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER=SimpleNamespace(
            get=lambda: (
                os.getenv("SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER", "0") == "1"
            )
        ),
        SGLANG_SIMULATE_ACC_LEN=SimpleNamespace(
            get=lambda: float(os.getenv("SGLANG_SIMULATE_ACC_LEN", "-1"))
        ),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for name in (
        "SGLANG_SIMULATOR_CONFIG_PATH",
        "SGLANG_SIMULATOR_COORDINATOR",
        "SGLANG_RAGGED_VERIFY_MODE",
        "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION",
        "SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER",
        "SGLANG_SIMULATE_ACC_LEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SGLANG_SIMULATOR_BOOTSTRAP", "1")
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")
    return module


def arguments(**changes):
    values = {
        "device": "cpu",
        "speculative_algorithm": "DSPARK",
        "speculative_dspark_block_size": 5,
        "speculative_num_draft_tokens": None,
        "speculative_num_steps": None,
        "speculative_eagle_topk": None,
        "speculative_draft_model_path": "/local/draft",
        "speculative_draft_model_revision": "a" * 40,
        "max_running_requests": 8,
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "nnodes": 1,
        "disable_overlap_schedule": True,
        "disable_cuda_graph": True,
        "cuda_graph_config": SimpleNamespace(
            decode=SimpleNamespace(backend="disabled"),
            prefill=SimpleNamespace(backend="disabled"),
        ),
    }
    return SimpleNamespace(**{**values, **changes})


def test_plain_import_and_disabled_installer_import_no_native_or_device_modules():
    src = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sglang', 'sgl_kernel', 'aiter'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from sglang_simulator.compat import native_joint_reference_requested
from sglang_simulator.simulation.sglang.dspark_args import install_cpu_dspark_args
assert native_joint_reference_requested() is False
assert install_cpu_dspark_args() is False
"""
    env = {**os.environ, "PYTHONPATH": str(src)}
    env.pop("SGLANG_SIMULATOR_BOOTSTRAP", None)
    env.pop("SGLANG_USE_CPU_ENGINE", None)
    env.pop("SGLANG_SIMULATOR_CONFIG_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("coordinated", [False, True])
def test_native_region_config_activates_original_overlap_before_cpu_resolution(
    native, monkeypatch, tmp_path, coordinated
):
    from sglang_simulator.compat import apply_simulator_server_args

    config = tmp_path / "native.json"
    config.write_text(
        json.dumps(
            {
                "predictor": {
                    "name": "infercast_dspark",
                    "native_segment_profile_id": "b" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", "")
    if coordinated:
        monkeypatch.setenv("SGLANG_SIMULATOR_COORDINATOR", "tcp://127.0.0.1:9000")
        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
    args = arguments(sim_config_path=str(config))
    apply_simulator_server_args(args)
    assert args.disable_overlap_schedule is False
    assert install_cpu_dspark_args() is True
    native._handle_dspark(args)
    assert native.calls == []
    assert len(native.declarations) == 1


@pytest.mark.parametrize(
    "bootstrap,cpu", [(None, None), ("1", None), (None, "1"), ("0", "1")]
)
def test_install_requires_both_exact_flags(native, monkeypatch, bootstrap, cpu):
    for name, value in (
        ("SGLANG_SIMULATOR_BOOTSTRAP", bootstrap),
        ("SGLANG_USE_CPU_ENGINE", cpu),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    original = native._handle_dspark
    assert install_cpu_dspark_args() is False
    assert native._handle_dspark is original


@pytest.mark.parametrize("width", [None, 6])
def test_cpu_declarations_preserve_raw_device_and_native_resolving_view(native, width):
    args = arguments(speculative_num_draft_tokens=width)
    args._overlay = {"grammar_backend": "xgrammar"}
    assert install_cpu_dspark_args() is True
    assert native._handle_dspark(args) is None
    assert not native.calls
    assert args.device == "cpu"
    assert args.speculative_num_draft_tokens is width
    assert native.resolving_view(args).device == "cpu"
    assert native.resolving_view(args).speculative_num_draft_tokens == 6
    assert len(native.declarations) == 1
    assert native.declarations[0][2] == {
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 6,
    }


def test_installer_is_idempotent_and_does_not_stack_wrappers(native):
    assert install_cpu_dspark_args() is True
    installed = native._handle_dspark
    assert install_cpu_dspark_args() is True
    assert native._handle_dspark is installed
    native._handle_dspark(arguments())
    assert len(native.declarations) == 1


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "npu", "xpu"])
def test_non_cpu_call_is_delegated_unchanged(native, device):
    install_cpu_dspark_args()
    args = SimpleNamespace(device=device)
    assert native._handle_dspark(args) == "native result"
    assert native.calls == [args]
    assert vars(args) == {"device": device}
    assert not native.declarations


def test_disabling_flags_after_install_preserves_native_handler(native, monkeypatch):
    install_cpu_dspark_args()
    monkeypatch.delenv("SGLANG_USE_CPU_ENGINE")
    args = arguments()
    assert native._handle_dspark(args) == "native result"
    assert native.calls == [args] and not native.declarations


@pytest.mark.parametrize(
    "changes",
    [
        {"speculative_algorithm": "DFLASH"},
        {"speculative_dspark_block_size": None},
        {"speculative_dspark_block_size": 4},
        {"speculative_dspark_block_size": 5.0},
        {"speculative_num_draft_tokens": 5},
        {"speculative_num_draft_tokens": 6.0},
        {"speculative_num_steps": 2},
        {"speculative_eagle_topk": True},
        {"speculative_draft_model_path": None},
        {"speculative_draft_model_revision": ""},
        {"speculative_draft_model_revision": "main"},
        {"speculative_draft_model_revision": "A" * 40},
        {"max_running_requests": None},
        {"max_running_requests": 0},
        {"max_running_requests": True},
        *[
            {name: 2}
            for name in (
                "tp_size",
                "ep_size",
                "dp_size",
                "pp_size",
                "attn_cp_size",
                "dcp_size",
                "nnodes",
            )
        ],
        {"tp_size": True},
        {"disable_overlap_schedule": False},
        {"disable_cuda_graph": False},
        {"cuda_graph_backend_decode": "full"},
        {"cuda_graph_backend_prefill": "breakable"},
        {
            "cuda_graph_config": {
                "decode": {"backend": "full"},
                "prefill": {"backend": "disabled"},
            }
        },
        {
            "cuda_graph_config": SimpleNamespace(
                decode=SimpleNamespace(backend="disabled"),
                prefill=SimpleNamespace(backend="full"),
            )
        },
        {"disaggregation_mode": "decode"},
        *[
            {name: True}
            for name in (
                "enable_dp_attention",
                "enable_dp_lm_head",
                "enable_torch_compile",
                "enable_multi_layer_eagle",
                "speculative_adaptive",
                "speculative_dspark_align_verify_tokens_to_graph_tier",
                "speculative_use_block_verification",
                "speculative_use_rejection_sampling",
                "enable_strict_thinking",
                "constrained_json_disable_any_whitespace",
            )
        ],
        *[
            {name: "unsupported"}
            for name in (
                "speculative_adaptive_config",
                "speculative_dspark_sps_table_path",
                "speculative_dspark_confidence_sts_path",
                "speculative_token_map",
                "constrained_json_whitespace_pattern",
                "grammar_backend",
            )
        ],
        {"speculative_draft_window_size": 128},
        {"speculative_draft_sink_size": 0},
        {"constrained_json_max_whitespace_cnt": 0},
        {"speculative_attention_mode": "decode"},
        {"speculative_accept_threshold_single": 0.5},
        {"speculative_accept_threshold_acc": True},
    ],
)
def test_unsupported_arguments_fail_before_any_declaration(native, changes):
    install_cpu_dspark_args()
    args = arguments(**changes)
    before = vars(args).copy()
    with pytest.raises(ValueError, match="CPU DSPark simulator"):
        native._handle_dspark(args)
    assert not native.calls and not native.declarations
    assert vars(args) == before


@pytest.mark.parametrize(
    "name,value",
    [
        ("SGLANG_RAGGED_VERIFY_MODE", "compact"),
        ("SGLANG_RAGGED_VERIFY_MODE", "cap-accept"),
        ("SGLANG_RAGGED_VERIFY_MODE", "typo"),
        ("SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION", "1"),
        ("SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER", "1"),
        ("SGLANG_SIMULATE_ACC_LEN", "3"),
        ("SGLANG_SIMULATE_ACC_LEN", "nan"),
        ("SGLANG_SIMULATOR_COORDINATOR", "tcp://127.0.0.1:9000"),
    ],
)
def test_unsupported_environment_fails_without_partial_mutation(
    native, monkeypatch, name, value
):
    install_cpu_dspark_args()
    monkeypatch.setenv(name, value)
    args = arguments()
    before = vars(args).copy()
    with pytest.raises(ValueError, match="CPU DSPark simulator"):
        native._handle_dspark(args)
    assert not native.calls and not native.declarations
    assert vars(args) == before


def test_native_default_revision_is_not_an_explicit_pin(native):
    install_cpu_dspark_args()
    args = arguments(speculative_draft_model_revision=None)
    args._overlay = {"speculative_draft_model_revision": "main"}
    with pytest.raises(ValueError, match="explicit speculative_draft_model_revision"):
        native._handle_dspark(args)
    assert not native.declarations


def test_native_default_max_running_is_not_an_explicit_capacity(native):
    install_cpu_dspark_args()
    args = arguments(max_running_requests=None)
    args._overlay = {"max_running_requests": 48}
    with pytest.raises(ValueError, match="positive explicit max_running_requests"):
        native._handle_dspark(args)
    assert not native.declarations


def test_resolved_draft_identity_cannot_differ_from_explicit_input(native):
    install_cpu_dspark_args()
    args = arguments()
    args._overlay = {"speculative_draft_model_revision": "b" * 40}
    with pytest.raises(ValueError, match="another speculative_draft_model_revision"):
        native._handle_dspark(args)
    assert not native.declarations


def test_non_cpu_original_exception_is_preserved(native):
    def reject(args):
        raise LookupError("original native failure")

    native._handle_dspark = reject
    install_cpu_dspark_args()
    with pytest.raises(LookupError, match="original native failure"):
        native._handle_dspark(SimpleNamespace(device="cuda"))


def test_explicit_one_step_and_disabled_phase_configuration_is_accepted(native):
    install_cpu_dspark_args()
    args = arguments(
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        cuda_graph_config={
            "decode": {"backend": "disabled"},
            "prefill": {"backend": "disabled"},
        },
    )
    native._handle_dspark(args)
    assert len(native.declarations) == 1


@pytest.fixture
def joint_config(native, monkeypatch, tmp_path):
    # Selection intent only. Constructor tests own complete source attestation.
    predictor = {
        "name": "infercast_dspark",
        "native_joint_source_binding": {"root": "not-an-attestation"},
        "cost_mode": "HYBRID",
        "progress_kind": "source_golden351",
        "native_analytic_model": {"EXTEND": {}, "DECODE": {}},
    }
    path = tmp_path / "joint.json"
    path.write_text(json.dumps({"predictor": predictor}), encoding="utf-8")
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(path))
    return path, predictor


def joint_arguments(**changes):
    return arguments(
        **{
            "tp_size": 2,
            "attn_dp_size": 2,
            "enable_dp_lm_head": True,
            "enable_dp_attention_local_control_broadcast": True,
            "moe_a2a_backend": "none",
            "disable_overlap_schedule": False,
            "max_running_requests": 64,
            **changes,
        }
    )


@pytest.mark.parametrize("coordinated", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("legacy_flag", [False, True])
def test_explicit_joint_entry_preserves_native_two_worker_topology_and_pool_input(
    native, joint_config, monkeypatch, coordinated, as_dict, legacy_flag
):
    from sglang_simulator.compat import (
        apply_simulator_server_args,
        native_joint_reference_requested,
        native_reference_requested,
        validate_simulator_server_args,
    )

    path, _ = joint_config
    monkeypatch.delenv("SGLANG_SIMULATOR_CONFIG_PATH")
    if coordinated:
        monkeypatch.setenv("SGLANG_SIMULATOR_COORDINATOR", "tcp://127.0.0.1:9000")
        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
    args = joint_arguments(sim_config_path=str(path), enable_dp_attention=legacy_flag)
    target = vars(args).copy() if as_dict else args
    apply_simulator_server_args(target)
    args = SimpleNamespace(**target) if as_dict else target
    assert native_joint_reference_requested() and native_reference_requested()
    validate_simulator_server_args(args)
    install_cpu_dspark_args()
    # The native resolution view owns legacy flag normalization. It does not
    # turn native DP1 into two model replicas or divide max-running a second time.
    args._overlay = {"enable_dp_attention": False}
    native._handle_dspark(args)
    resolved = native.resolving_view(args)
    assert (resolved.tp_size, resolved.dp_size, resolved.attn_dp_size) == (2, 1, 2)
    assert resolved.tp_size * resolved.dp_size * resolved.pp_size == 2
    assert resolved.tp_size // resolved.attn_dp_size // resolved.attn_cp_size == 1
    assert resolved.max_running_requests == args.max_running_requests == 64
    assert resolved.disable_overlap_schedule is False
    assert resolved.disable_cuda_graph is True
    assert resolved.speculative_num_draft_tokens == 6
    assert args.speculative_num_draft_tokens is None
    assert len(native.declarations) == 1 and not native.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "other"},
        {"native_joint_source_binding": None},
        {"native_joint_source_binding": "untyped"},
        {"cost_mode": "SILICON"},
        {"progress_kind": "empirical"},
        {"native_analytic_model": None},
    ],
)
def test_joint_entry_requires_the_explicit_predictor_domain(joint_config, changes):
    from sglang_simulator.compat import (
        SIMULATOR_SERVER_ARG_OVERRIDES,
        apply_simulator_server_args,
        native_joint_reference_requested,
        native_reference_requested,
    )

    path, predictor = joint_config
    path.write_text(
        json.dumps({"predictor": {**predictor, **changes}}), encoding="utf-8"
    )
    args = {}
    apply_simulator_server_args(args)
    assert not native_joint_reference_requested() and not native_reference_requested()
    assert args == SIMULATOR_SERVER_ARG_OVERRIDES


@pytest.mark.parametrize(
    "changes",
    [
        {"tp_size": 1},
        {"tp_size": True},
        {"dp_size": 2},
        {"attn_dp_size": 1},
        {"attn_dp_size": 2.0},
        *[
            {name: 2}
            for name in ("ep_size", "pp_size", "attn_cp_size", "dcp_size", "nnodes")
        ],
        {"enable_dp_lm_head": False},
        {"enable_dp_lm_head": 1},
        {"enable_dp_attention_local_control_broadcast": False},
        {"enable_dp_attention_local_control_broadcast": 1},
        {"moe_a2a_backend": "mori"},
        {"speculative_moe_a2a_backend": "mori"},
        {"enable_two_batch_overlap": True},
        {"disable_overlap_schedule": True},
        {"disable_cuda_graph": False},
    ],
)
def test_joint_arguments_reject_unrepresented_physical_execution(
    native, joint_config, changes
):
    install_cpu_dspark_args()
    args = joint_arguments(**changes)
    before = vars(args).copy()
    with pytest.raises(ValueError, match="CPU DSPark simulator"):
        native._handle_dspark(args)
    assert vars(args) == before
    assert not native.declarations and not native.calls


@pytest.mark.parametrize(
    "name,value",
    [
        ("SGLANG_RAGGED_VERIFY_MODE", "compact"),
        ("SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION", "1"),
        ("SGLANG_SIMULATE_ACC_LEN", "3.51"),
    ],
)
def test_joint_selection_does_not_relax_native_sampling_or_phase_serialization(
    native, joint_config, monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    install_cpu_dspark_args()
    with pytest.raises(ValueError, match="CPU DSPark simulator"):
        native._handle_dspark(joint_arguments())
    assert not native.declarations and not native.calls


@pytest.mark.parametrize(
    "changes,env,error",
    [
        ({"enable_lora": True}, {}, "enable_lora"),
        ({"enable_hierarchical_cache": True}, {}, "enable_hierarchical_cache"),
        ({"disaggregation_mode": "decode"}, {}, "disaggregation"),
        ({"speculative_algorithm": "EAGLE"}, {}, "source-bound native DSPark"),
        ({}, {"SGLANG_SIMULATOR_OUTPUT_MODE": "BLOCKING"}, "requires OFFLINE"),
        ({}, {"SGLANG_REQ_RUNNING_TIMEOUT": "1"}, "virtual deadline"),
    ],
)
def test_joint_coordinator_preserves_existing_restrictions(
    joint_config, monkeypatch, changes, env, error
):
    from sglang_simulator.compat import (
        apply_simulator_server_args,
        validate_simulator_server_args,
    )

    monkeypatch.setenv("SGLANG_SIMULATOR_COORDINATOR", "tcp://127.0.0.1:9000")
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
    for name in ("SGLANG_REQ_WAITING_TIMEOUT", "SGLANG_REQ_RUNNING_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    args = joint_arguments(**changes)
    apply_simulator_server_args(args)
    with pytest.raises(ValueError, match=error):
        validate_simulator_server_args(args)
