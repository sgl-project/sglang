from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


def test_flag_off_model_import_does_not_load_helion() -> None:
    root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "python")
    env.pop("SGLANG_FORCE_FUSED_OP_BACKEND", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sglang.srt.models.muse_spark; "
            "assert 'helion' not in sys.modules; "
            "assert 'sglang.kernels.ops.muse_spark.helion_megakernels.integration' "
            "not in sys.modules",
        ],
        check=True,
        cwd=root,
        env=env,
    )


def test_non_int4_expert_tp_requires_opt_in() -> None:
    from sglang.kernels.spec import KernelBackend
    from sglang.srt.layers.muse_spark import muse_spark_expert_tp

    with patch.object(
        muse_spark_expert_tp,
        "get_parallel",
        return_value=SimpleNamespace(moe_tp_size=1, moe_ep_size=1),
    ):
        muse_spark_expert_tp.validate(False)
    with (
        patch.object(
            muse_spark_expert_tp,
            "get_parallel",
            return_value=SimpleNamespace(moe_tp_size=4, moe_ep_size=1),
        ),
        patch.object(muse_spark_expert_tp, "get_fused_op_backend", return_value=None),
    ):
        muse_spark_expert_tp.validate(True)
        with pytest.raises(ValueError, match="requires.*FUSED_OP_BACKEND=helion"):
            muse_spark_expert_tp.validate(False)
    with (
        patch.object(
            muse_spark_expert_tp,
            "get_parallel",
            return_value=SimpleNamespace(moe_tp_size=4, moe_ep_size=1),
        ),
        patch.object(
            muse_spark_expert_tp,
            "get_fused_op_backend",
            return_value=KernelBackend.HELION,
        ),
    ):
        muse_spark_expert_tp.validate(False)


def test_expert_tp_partition_comes_from_parallel_context() -> None:
    from sglang.srt.layers.muse_spark import muse_spark_expert_tp

    with patch.object(
        muse_spark_expert_tp,
        "get_parallel",
        return_value=SimpleNamespace(moe_tp_size=1),
    ):
        assert muse_spark_expert_tp.partition(16384) == (16384, False)
    with patch.object(
        muse_spark_expert_tp,
        "get_parallel",
        return_value=SimpleNamespace(moe_tp_size=4),
    ):
        assert muse_spark_expert_tp.partition(16384) == (4096, True)


def test_verify_megakernels_are_gated_to_gb300() -> None:
    from sglang.srt.layers.muse_spark import muse_spark_expert_tp

    parallel = SimpleNamespace(tp_size=4, moe_tp_size=4, moe_ep_size=1)
    spec = SimpleNamespace(speculative_num_draft_tokens=16)
    for device_sm, enabled in ((103, True), (100, False)):
        with (
            patch.object(muse_spark_expert_tp, "helion_enabled", return_value=True),
            patch.object(muse_spark_expert_tp, "get_parallel", return_value=parallel),
            patch.object(muse_spark_expert_tp, "get_spec", return_value=spec),
            patch.object(
                muse_spark_expert_tp,
                "get_platform",
                return_value=SimpleNamespace(device_sm=device_sm),
            ),
        ):
            assert muse_spark_expert_tp.helion_verify_enabled() is enabled


def test_expert_tp_checkpoint_slicing() -> None:
    from sglang.srt.layers.muse_spark import muse_spark_expert_tp

    parallel = SimpleNamespace(moe_tp_size=4, moe_tp_rank=1)
    gate_up = torch.empty(1, 4)
    loaded_gate_up = torch.arange(16).reshape(1, 16)
    down = torch.empty(1, 3, 2)
    loaded_down = torch.arange(24).reshape(1, 3, 8)
    with patch.object(muse_spark_expert_tp, "get_parallel", return_value=parallel):
        assert muse_spark_expert_tp.copy_shard(
            gate_up, loaded_gate_up, "layers.0.mlp.experts.gate_up_proj"
        )
        assert muse_spark_expert_tp.copy_shard(
            down, loaded_down, "layers.0.mlp.experts.down_proj"
        )
    torch.testing.assert_close(
        gate_up, torch.tensor([[2, 3, 10, 11]], dtype=gate_up.dtype)
    )
    torch.testing.assert_close(down, loaded_down[:, :, 2:4].to(down.dtype))


def _verify_call(rows: int, *, verify: bool = True, batch_size: int = 1):
    tensor = SimpleNamespace(ndim=2, shape=(rows, 8192), dtype=torch.bfloat16)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: verify),
        batch_size=batch_size,
    )
    return tensor, forward_batch


def test_single_request_keeps_other_fused_ops_on_default_backends() -> None:
    from sglang.kernels import fused_op
    from sglang.kernels.spec import KernelBackend
    from sglang.srt.layers.muse_spark import muse_spark_expert_tp

    previous = fused_op.get_fused_op_backend()
    fused_op.set_fused_op_backend(KernelBackend.HELION)
    try:
        with patch.object(
            muse_spark_expert_tp, "helion_verify_enabled", return_value=True
        ):
            for rows, verify, batch_size, inside in (
                (16, True, 1, None),
                (16, False, 1, None),
                (16, True, 2, KernelBackend.HELION),
            ):
                _, batch = _verify_call(rows, verify=verify, batch_size=batch_size)
                with muse_spark_expert_tp.default_fused_ops_for_single_request(batch):
                    assert fused_op.get_fused_op_backend() is inside
                assert fused_op.get_fused_op_backend() is KernelBackend.HELION
    finally:
        fused_op.set_fused_op_backend(previous)


def _verify_layer(bank: object = object()) -> SimpleNamespace:
    return SimpleNamespace(
        mlp=SimpleNamespace(experts=SimpleNamespace(helion_verify_bank=bank))
    )


def test_megakernel_eligibility_is_single_request_verify() -> None:
    from sglang.kernels.ops.muse_spark.helion_megakernels import (
        integration,
    )

    backend = SimpleNamespace(forward_metadata=SimpleNamespace(is_ragged_verify=False))
    with patch.object(integration, "get_attn_backend", return_value=backend):
        for rows in (8, 16):
            tensor, batch = _verify_call(rows)
            assert integration.megakernel_verify_enabled(_verify_layer(), batch, tensor)
        for rows in (1, 12, 32):
            tensor, batch = _verify_call(rows)
            assert not integration.megakernel_verify_enabled(
                _verify_layer(), batch, tensor
            )
        tensor, batch = _verify_call(16, verify=False)
        assert not integration.megakernel_verify_enabled(_verify_layer(), batch, tensor)
        tensor, batch = _verify_call(16, batch_size=2)
        assert not integration.megakernel_verify_enabled(_verify_layer(), batch, tensor)
        tensor, batch = _verify_call(16)
        assert not integration.megakernel_verify_enabled(
            _verify_layer(None), batch, tensor
        )


def _fake_helion(*, launch_dependents: bool = True, pipeline: bool = True) -> dict:
    """A stand-in Helion with just the features the integration checks for."""
    hl = ModuleType("helion.language")
    hl.pdl_wait = lambda: None
    if launch_dependents:
        hl.pdl_launch_dependents = lambda: None

    def config(cross_loop_pipeline=None) -> None:
        del cross_loop_pipeline

    helion = ModuleType("helion")
    helion.language = hl
    helion.Config = config if pipeline else (lambda: None)
    return {"helion": helion, "helion.language": hl}


def test_megakernel_requires_helion_features() -> None:
    from sglang.kernels.ops.muse_spark.helion_megakernels import integration

    for modules, version, missing in (
        (_fake_helion(), "1.5.0", None),
        (_fake_helion(), "1.4.1.dev438", "version >=1.5.0"),
        (_fake_helion(launch_dependents=False), "1.5.0", "hl.pdl_launch_dependents"),
        (_fake_helion(pipeline=False), "1.5.0", "cross_loop_pipeline"),
    ):
        integration.require_helion_features.cache_clear()
        with (
            patch.dict(sys.modules, modules),
            patch(
                "sglang.kernels.ops.muse_spark.helion_runtime._base_package_version",
                return_value=version,
            ),
        ):
            if missing is None:
                integration.require_helion_features()
            else:
                with pytest.raises(RuntimeError, match=missing):
                    integration.require_helion_features()
    integration.require_helion_features.cache_clear()


def test_nvfp4_marlin_rejects_expert_tp() -> None:
    from sglang.srt.layers.muse_spark import muse_spark_fp4_moe

    with (
        patch.object(
            muse_spark_fp4_moe.expert_tp, "partition", return_value=(1024, True)
        ),
        patch.object(
            muse_spark_fp4_moe,
            "_resolve_moe_runner_backend",
            return_value=muse_spark_fp4_moe.MoeRunnerBackend.MARLIN,
        ),
        patch.object(muse_spark_fp4_moe, "is_cuda", return_value=False),
        pytest.raises(ValueError, match="Marlin path does not reduce"),
    ):
        muse_spark_fp4_moe.MuseSpark1xFp4Experts(
            num_experts=8,
            num_local_experts=8,
            local_expert_offset=0,
            hidden_size=1024,
            intermediate_size=4096,
            top_k=2,
            quant_config=None,
        )


@pytest.mark.parametrize("use_rope", [True, False], ids=["sliding", "global"])
def test_helion_forward_publishes_routed_experts(use_rope: bool) -> None:
    from sglang.kernels.ops.muse_spark.helion_megakernels import integration

    topk_ids = torch.tensor([[7, 3]], dtype=torch.int32)
    gated = []
    runner = SimpleNamespace(
        topk_ids=topk_ids,
        sliding_attention=lambda **_kwargs: "mk-gated",
        layer=lambda **kwargs: (
            gated.append(kwargs["gated"]) or ("new-stream", "next-attention", None)
        ),
    )
    module_name = "sglang.kernels.ops.muse_spark.helion_megakernels.megakernel_verify"
    fake_module = ModuleType(module_name)
    fake_module.get_megakernel_verify = lambda **_kwargs: runner
    captures = []
    capturer = SimpleNamespace(
        capture=lambda **kwargs: captures.append(kwargs),
    )
    layer = SimpleNamespace(
        layer_idx=11,
        _verify_qkv=None,
        _megakernel_verify_params=(object(), object(), object()),
        _fused_ffn_boundary=object(),
        self_attn=SimpleNamespace(
            use_rope=use_rope,
            qkv_proj=lambda _value: (object(), None),
            gated_attention=lambda *_args: "stock-gated",
        ),
    )
    attn_input, forward_batch = _verify_call(16)
    attn_input.device = torch.device("cuda", 0)

    with (
        patch.dict(sys.modules, {module_name: fake_module}),
        integration.get_parallel().override(
            tp_group=SimpleNamespace(device_group=object())
        ),
        patch.object(integration, "_attention_inputs", return_value=object()),
        patch.object(integration, "require_helion_features"),
        patch.object(integration, "get_global_experts_capturer", return_value=capturer),
        patch.object(integration, "megakernel_verify_enabled", return_value=True),
    ):
        result = integration.try_helion_megakernel_forward(
            layer, object(), object(), attn_input, forward_batch
        )

    assert result == ("new-stream", "next-attention")
    # Global layers keep the stock attention; both feed the layer megakernels.
    assert gated == ["mk-gated" if use_rope else "stock-gated"]
    assert captures == [{"layer_id": 11, "topk_indices": topk_ids}]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
