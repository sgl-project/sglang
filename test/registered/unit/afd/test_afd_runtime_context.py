"""Role publication, validation purity and native weight ownership."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import json
import os
from types import SimpleNamespace

import pytest

from sglang.srt import runtime_context
from sglang.srt.afd import config, model_hooks
from sglang.srt.afd.contracts import AFDError
from sglang.test.afd.config_fixtures import _server_args


@pytest.mark.parametrize("lanes", [1, 2])
@pytest.mark.parametrize("explicit", [None, "0", "1"])
@pytest.mark.parametrize("nnodes", [1, 2])
def test_ffn_inherits_native_communication_environment(
    monkeypatch, lanes, explicit, nnodes
):
    from sglang.srt.afd import ffn_server
    from sglang.srt.entrypoints import engine

    # Execute the real native environment policy, isolating process-wide setup.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in ("NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE", "NCCL_MNNVL_ENABLE"):
        monkeypatch.delenv(key, raising=False)
    if explicit is not None:
        monkeypatch.setenv("NCCL_CUMEM_ENABLE", explicit)
        monkeypatch.setenv("NCCL_NVLS_ENABLE", explicit)
    monkeypatch.setenv("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
    monkeypatch.setattr(engine, "resolving_view", lambda args: args)
    monkeypatch.setattr(engine, "is_mnnvl_fabric_device", lambda: True)
    monkeypatch.setattr(engine, "set_ulimit", lambda: None)
    monkeypatch.setattr(engine.signal, "signal", lambda *args: None)
    monkeypatch.setattr(engine.mp, "set_start_method", lambda *args, **kwargs: None)
    args = SimpleNamespace(
        afd_execution_mode="ffn",
        afd_config=config.AFDConfig(lanes=lanes * nnodes),
        nnodes=nnodes,
        node_rank=0,
        enable_symm_mem=False,
        enable_nccl_nvls=False,
        dcp_size=1,
        enable_metrics=False,
        custom_sigquit_handler=None,
        gc_threshold=None,
        check_server_args=lambda: None,
    )
    entered = []

    def enter(*args, **kwargs):
        entered.append(True)
        assert os.environ["NCCL_CUMEM_ENABLE"] == (
            explicit if explicit is not None else str(int(nnodes > 1))
        )
        assert os.environ["NCCL_NVLS_ENABLE"] == (explicit or "0")
        if nnodes > 1:
            assert os.environ["NCCL_MNNVL_ENABLE"] == "1"

    monkeypatch.setattr(ffn_server, "run_lane", enter)
    monkeypatch.setattr(ffn_server, "_run_lanes", enter)
    ffn_server.launch_server(args)
    assert entered == [True]


@pytest.mark.parametrize("role", ["off", "attention", "ffn"])
def test_runtime_context_is_the_only_role_owner(monkeypatch, role):
    context = runtime_context.RuntimeContext(parallel=runtime_context.ParallelContext())
    monkeypatch.setattr(runtime_context, "_CONTEXT", context)
    published = SimpleNamespace(afd_execution_mode=role)
    context.set_server_args(published)
    assert model_hooks.afd_execution_mode().value == role
    attention = (("qkv", "q", "q"),)
    mlp = (("up_gate", "up", 0),)
    assert (
        model_hooks.afd_stacked_params_mapping(attention=attention, dense_mlp=mlp)
        == {
            "off": attention + mlp,
            "attention": attention,
            "ffn": (),
        }[role]
    )
    assert model_hooks.afd_owns_expert_weights() == (role != "attention")
    # Validation of a different role (successful or failed) must not publish it.
    config.validate_afd_server_args(_server_args())
    with pytest.raises(AFDError, match="AFD_EXECUTION_MODE_INVALID"):
        config.validate_afd_server_args(_server_args(afd_execution_mode="invalid"))
    assert runtime_context.get_server_args() is published
    assert model_hooks.afd_execution_mode().value == role


@pytest.mark.parametrize(
    "role,draft", [("attention", False), ("off", False), ("attention", True)]
)
def test_attention_runner_uses_captured_tp_rank(monkeypatch, role, draft):
    from sglang.srt.afd import integration
    from sglang.srt.model_executor.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.is_draft_worker = draft
    runner.server_args = SimpleNamespace(afd_execution_mode=role, afd_config=object())
    runner.afd_runtime = None
    attached, calls = [], []
    pipeline = object()
    enabled = role == "attention" and not draft
    if enabled:
        # ModelRunner captures placement during native distributed initialization.
        # No legacy ``ps`` attribute or published global context is needed here.
        runner.tp_rank = 3
        runner.model = SimpleNamespace(
            model=SimpleNamespace(set_afd_pipeline=attached.append)
        )
        runner.attn_backend = object()
        runner.device, runner.dtype, runner.max_running_requests = "cpu", "bf16", 8
    monkeypatch.setattr(
        integration,
        "build_attention_pipeline",
        lambda **kwargs: (calls.append(kwargs), pipeline)[1],
    )

    runner.maybe_init_afd_runtime()

    if enabled:
        assert calls[0]["lane"] == 3
        assert runner.afd_runtime is pipeline
        assert attached == [pipeline]
    else:
        assert calls == attached == []
        assert runner.afd_runtime is None


@pytest.mark.parametrize(
    "role,draft,tp,ep,error",
    [
        ("attention", False, 3, 1, None),
        ("attention", False, 20, 1, None),
        ("attention", False, 4, 1, None),
        ("attention", False, 8, 1, None),
        ("off", False, 3, 1, "moe_intermediate_size"),
        ("off", False, 20, 1, "moe_intermediate_size"),
        ("ffn", False, 3, 1, "moe_intermediate_size"),
        ("ffn", False, 20, 1, "moe_intermediate_size"),
        ("attention", True, 3, 1, "moe_intermediate_size"),
        ("attention", True, 20, 1, "moe_intermediate_size"),
        ("off", False, 4, 1, None),
        ("ffn", False, 4, 4, None),
        ("ffn", False, 32, 32, None),
        ("off", False, 3, 2, "must be divisible by ep_size"),
        ("ffn", False, 32, 1, "For quantized MoE models"),
    ],
)
def test_runner_quantized_moe_check_applies_to_local_expert_owners(
    monkeypatch, role, draft, tp, ep, error
):
    from sglang.srt.model_executor import model_runner
    from sglang.srt.model_executor.model_runner_components import moe_ep_setup

    monkeypatch.setenv("SGLANG_SHARED_EXPERT_TP1", "0")
    monkeypatch.setattr(moe_ep_setup, "_use_aiter", False)
    monkeypatch.setattr(
        model_runner,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_size=ep, moe_dp_size=1),
    )
    runner = object.__new__(model_runner.ModelRunner)
    runner.is_draft_worker = draft
    runner.server_args = SimpleNamespace(afd_execution_mode=role)
    runner.tp_size = tp
    hf = SimpleNamespace(
        moe_intermediate_size=2048,
        quantization_config={"weight_block_size": [128, 128]},
    )
    runner.model_config = SimpleNamespace(hf_config=hf, hf_text_config=hf)
    calls = []

    def check(**kwargs):
        calls.append(kwargs)
        return moe_ep_setup.check_quantized_moe_compatibility(**kwargs)

    monkeypatch.setattr(model_runner, "check_quantized_moe_compatibility", check)
    if error is None:
        runner.check_quantized_moe_compatibility()
    else:
        with pytest.raises(ValueError, match=error):
            runner.check_quantized_moe_compatibility()
    assert len(calls) == int(role != "attention" or draft)


@pytest.mark.parametrize(
    "name,value",
    [
        ("old_internal_flag", 1),
        ("observe", 2),
        ("quantum", 8),
        ("max_buckets", 8),
        ("graph_mode", "child"),
        ("router_on_attention", True),
        ("ffn_backend", "none"),
        ("ffn_merge_reduce_scatter", True),
        ("ffn_expert_parallel", True),
        ("mtp_draft_extend_graph", True),
    ],
)
def test_unknown_config_fields_are_rejected(name, value):
    with pytest.raises(ValueError, match="unknown field"):
        config.AFDConfig.from_json(json.dumps({name: value}))


def test_real_msgspec_types_and_frozen_configuration():
    with pytest.raises(ValueError, match="AFD_GRAPH_CONFIG_INVALID"):
        config.AFDConfig.from_json('{"stages": "2"}')
    value = config.AFDConfig.from_json('{"stages": 2, "nccl_num_channels": 16}')
    assert value.stages == 2
    assert value.nccl_num_channels == 16
    with pytest.raises(AttributeError):
        value.stages = 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
