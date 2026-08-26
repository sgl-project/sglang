from unittest.mock import patch

import torch

from sglang.srt.models import qwen3_5
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class _Linear:
    quant_method = object()


class _ModelOptFp8LinearMethod:
    use_marlin = False


class _RecordingLinear:
    def __init__(self, quant_method):
        self.quant_method = quant_method
        self.input = None

    def __call__(self, hidden_states):
        self.input = hidden_states
        return hidden_states, None


def test_cuda_modelopt_fp8_tuple_reuses_existing_handoff():
    bf16 = torch.randn(2, 8)
    fp8 = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1)
    linear = _Linear()
    linear.quant_method = _ModelOptFp8LinearMethod()

    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(qwen3_5, "_fp8_static_input_scale", return_value=scale),
    ):
        assert qwen3_5._linear_accepts_fp8_tuple(linear)
        dual_result = qwen3_5._select_fused_ar_input_for_linear(
            (bf16, fp8, scale), linear
        )
        quant_result = qwen3_5._select_fused_ar_input_for_linear((fp8, scale), linear)
        assert dual_result[0] is fp8 and dual_result[1] is scale
        assert quant_result[0] is fp8 and quant_result[1] is scale


def test_gdn_routes_dual_output_to_quantized_and_bf16_projections():
    bf16 = torch.randn(2, 8)
    fp8 = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1)
    gdn = type("GDN", (), {})()
    gdn.in_proj_qkvz = _RecordingLinear(_ModelOptFp8LinearMethod())
    gdn.in_proj_ba = _RecordingLinear(object())
    gdn.alt_stream = None

    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(qwen3_5, "check_cuda_graph_backend", return_value=False),
        patch.object(qwen3_5, "_fp8_static_input_scale", return_value=scale),
    ):
        qkvz, ba = qwen3_5.Qwen3_5GatedDeltaNet._forward_input_proj_fused_quant(
            gdn, (bf16, fp8, scale)
        )

    assert gdn.in_proj_qkvz.input[0] is fp8
    assert gdn.in_proj_qkvz.input[1] is scale
    assert gdn.in_proj_ba.input is bf16
    assert qkvz is gdn.in_proj_qkvz.input
    assert ba is bf16


def test_cuda_non_modelopt_fp8_method_does_not_accept_tuple():
    linear = _Linear()
    linear.quant_method = type("Fp8LinearMethod", (), {"block_quant": False})()
    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(qwen3_5, "_fp8_static_input_scale", return_value=None),
    ):
        assert not qwen3_5._linear_accepts_fp8_tuple(linear)


def test_cuda_tuple_gate_matches_static_per_tensor_fp8_producer():
    linear = _Linear()
    scale = torch.ones(1)

    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(
            qwen3_5, "_fp8_static_input_scale", return_value=scale, create=True
        ),
    ):
        assert qwen3_5._linear_accepts_fp8_tuple(linear)

    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(
            qwen3_5, "_fp8_static_input_scale", return_value=None, create=True
        ),
    ):
        assert not qwen3_5._linear_accepts_fp8_tuple(linear)


def test_communicator_gate_uses_quant_method_before_scale_is_scalarized():
    linear = _Linear()
    linear.input_scale = torch.ones(2)

    with (
        patch.object(qwen3_5, "_enable_qwen35_fused_ar_quant", return_value=True),
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(
            qwen3_5, "_is_static_per_tensor_fp8_linear", return_value=True, create=True
        ),
        patch.object(qwen3_5, "_fp8_static_input_scale", return_value=None),
    ):
        assert qwen3_5._enable_fused_ar_quant_for_linear(linear)
        assert not qwen3_5._linear_accepts_fp8_tuple(linear)


def test_rocm_communicator_gate_matches_per_group_tuple_consumer():
    linear = _Linear()
    linear.quant_method = type(
        "Fp8LinearMethod", (), {"block_quant": False, "use_mxfp8": False}
    )()

    with (
        patch.object(qwen3_5, "_enable_qwen35_fused_ar_quant", return_value=True),
        patch.object(qwen3_5, "_use_aiter", True),
        patch.object(qwen3_5, "_is_static_per_tensor_fp8_linear", return_value=True),
    ):
        assert not qwen3_5._enable_fused_ar_quant_for_linear(linear)
        linear.quant_method.block_quant = True
        assert qwen3_5._enable_fused_ar_quant_for_linear(linear)


def test_cuda_prequantized_handoff_preserves_linear_output_dtype():
    fp8 = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1)
    linear = _Linear()
    linear.orig_dtype = torch.float16

    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(qwen3_5, "_fp8_static_input_scale", return_value=scale),
    ):
        selected = qwen3_5._select_fused_ar_input_for_linear((fp8, scale), linear)

    assert len(selected) == 3
    assert selected[0] is fp8 and selected[1] is scale
    assert selected[2] is torch.float16


def test_shared_gate_can_keep_importing_models_rocm_only():
    context = type(
        "Context",
        (),
        {"is_config_namespace_published": lambda self, name: True},
    )()
    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(qwen3_5, "_is_cuda", True),
        patch.object(
            qwen3_5,
            "is_flashinfer_allreduce_fusion_arch_supported",
            return_value=True,
            create=True,
        ),
        patch.object(qwen3_5, "get_context", return_value=context),
        patch.object(
            qwen3_5,
            "get_exec",
            return_value=type(
                "Exec",
                (),
                {
                    "comm": type(
                        "Comm",
                        (),
                        {"flashinfer_allreduce_fusion_backend": "trtllm"},
                    )()
                },
            )(),
        ),
    ):
        assert qwen3_5._enable_qwen35_fused_ar_quant()
        assert not qwen3_5._enable_qwen35_fused_ar_quant(allow_cuda=False)


def test_cuda_gate_fails_closed_before_exec_config_is_published():
    context = type(
        "Context",
        (),
        {"is_config_namespace_published": lambda self, name: False},
    )()
    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(qwen3_5, "_is_cuda", True),
        patch.object(
            qwen3_5,
            "is_flashinfer_allreduce_fusion_arch_supported",
            return_value=True,
            create=True,
        ),
        patch.object(qwen3_5, "get_context", return_value=context),
        patch.object(
            qwen3_5,
            "get_exec",
            side_effect=ValueError("config namespace 'exec' not published"),
        ) as get_exec,
    ):
        assert not qwen3_5._enable_qwen35_fused_ar_quant()
        get_exec.assert_not_called()


def test_cuda_gate_rejects_unsupported_flashinfer_allreduce_arch():
    context = type(
        "Context",
        (),
        {"is_config_namespace_published": lambda self, name: True},
    )()
    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(qwen3_5, "_is_cuda", True),
        patch.object(
            qwen3_5,
            "is_flashinfer_allreduce_fusion_arch_supported",
            return_value=False,
            create=True,
        ),
        patch.object(qwen3_5, "get_context", return_value=context),
    ):
        assert not qwen3_5._enable_qwen35_fused_ar_quant()


def test_amd_tuple_predicate_remains_per_group_only():
    linear = _Linear()
    with patch.object(qwen3_5, "_use_aiter", True):
        assert not qwen3_5._linear_accepts_fp8_tuple(linear)

    linear.quant_method = type(
        "Fp8LinearMethod", (), {"block_quant": True, "use_mxfp8": False}
    )()
    with patch.object(qwen3_5, "_use_aiter", True):
        assert qwen3_5._linear_accepts_fp8_tuple(linear)
