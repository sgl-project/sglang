"""Keep optional region requirements separate from standard Torch MPS."""

import json
from unittest import mock

import pytest
import torch

from sglang.srt import server_args
from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.hardware_backend.mlx import region_runtime, runtime
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


@pytest.mark.parametrize(
    "device,native_mlx,detected_mps,region_enabled,expected",
    [
        ("mps", False, False, False, False),
        ("mps", False, False, True, True),
        (None, False, True, True, True),
        ("cpu", False, True, True, False),
        (None, False, False, True, False),
        ("mps", True, True, True, False),
    ],
)
def test_server_args_gates_only_the_selected_region(
    device, native_mlx, detected_mps, region_enabled, expected
):
    with (
        mock.patch.object(server_args, "use_mlx", return_value=native_mlx),
        override_platform(is_mps=detected_mps),
        mock.patch.object(server_args, "validate_mps_runtime"),
        mock.patch.object(
            server_args.envs.SGLANG_ENABLE_MLX_WHOLE_REGION,
            "get",
            return_value=region_enabled,
        ),
        mock.patch.object(region_runtime, "validate_mlx_region_runtime") as validate,
    ):
        ServerArgs(model_path="dummy", device=device).resolve_once()
    assert validate.call_count == int(expected)


def test_region_validates_both_runtimes_and_kv_commit_api():
    with (
        mock.patch.object(region_runtime, "validate_mps_runtime") as mps,
        mock.patch.object(runtime, "_validate_runtime") as mlx,
        mock.patch.object(torch.mps, "compile_shader", None, create=True),
        pytest.raises(RuntimeError, match="compile_shader"),
    ):
        region_runtime.validate_mlx_region_runtime()
    mps.assert_called_once_with()
    mlx.assert_called_once_with()


def test_region_errors_identify_the_region_flag():
    with (
        mock.patch.object(region_runtime, "validate_mps_runtime"),
        mock.patch.object(
            runtime,
            "_validate_runtime",
            side_effect=RuntimeError("SGLANG_USE_MLX requires MLX"),
        ),
        pytest.raises(
            RuntimeError, match="SGLANG_ENABLE_MLX_WHOLE_REGION requires MLX"
        ),
    ):
        region_runtime.validate_mlx_region_runtime()


@pytest.mark.parametrize("disabled_phase", [None, "decode", "prefill"])
def test_resolution_preserves_explicit_phase_disables(disabled_phase, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_hidden_layers": 2,
                "vocab_size": 128,
                "max_position_embeddings": 2048,
            }
        )
    )
    args = [
        "--model-path",
        str(tmp_path),
        "--device",
        "mps",
        "--attention-backend",
        "torch_native",
        "--disable-overlap-schedule",
    ]
    if disabled_phase is not None:
        args += [f"--cuda-graph-backend-{disabled_phase}", "disabled"]
    with (
        mock.patch.object(
            server_args.envs.SGLANG_ENABLE_MLX_WHOLE_REGION, "get", return_value=True
        ),
        mock.patch.object(server_args, "validate_mps_runtime"),
        mock.patch.object(region_runtime, "validate_mlx_region_runtime"),
    ):
        parsed = prepare_server_args(args)
        parsed.resolve_once()
        resolved = resolved_view(parsed)
    for phase in ("decode", "prefill"):
        assert resolved.cuda_graph_config[phase].backend == (
            "disabled" if phase == disabled_phase else "full"
        )
    assert resolved.cuda_graph_config.decode.bs == [1, 2, 4, 8, 12, 16]
    assert resolved.cuda_graph_config.prefill.max_bs == 2048
