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
from sglang_simulator.simulation.sglang.dspark_args import install_cpu_dspark_args
assert install_cpu_dspark_args() is False
"""
    env = {**os.environ, "PYTHONPATH": str(src)}
    env.pop("SGLANG_SIMULATOR_BOOTSTRAP", None)
    env.pop("SGLANG_USE_CPU_ENGINE", None)
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
