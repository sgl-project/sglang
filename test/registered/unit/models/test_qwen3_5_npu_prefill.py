"""CPU contracts for Qwen3.5's fused NPU prefill adapter."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen3_5
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen35NPUPrefill(unittest.TestCase):
    def test_only_npu_prefill_changes_dispatch(self):
        prefill_modes = {
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
            ForwardMode.SPLIT_PREFILL,
        }
        for backend in ("npu", "cuda", "hip", "cpu", "xpu"):
            flags = {
                f"_is_{name}": name == backend
                for name in ("npu", "cuda", "hip", "cpu", "xpu")
            }
            for gate in (False, True):
                for mode in ForwardMode:
                    with (
                        self.subTest(backend=backend, gate=gate, mode=mode),
                        mock.patch.multiple(qwen3_5, **flags),
                    ):
                        layer = SimpleNamespace(attn_output_gate=gate)
                        for path in (
                            "native",
                            "cuda_fused",
                            "fused_gate",
                            "npu",
                            "npu_prefill",
                        ):
                            setattr(
                                layer,
                                f"forward_prepare_{path}",
                                mock.Mock(return_value=path),
                            )
                        batch = SimpleNamespace(forward_mode=mode)
                        result = qwen3_5.Qwen3_5AttentionDecoderLayer._prepare_qkv_gate(
                            layer, None, None, batch
                        )
                        if backend == "npu":
                            expected = (
                                "npu_prefill"
                                if mode in prefill_modes
                                else ("npu" if gate else "native")
                            )
                        else:
                            expected = (
                                ("cuda_fused" if backend == "cuda" else "fused_gate")
                                if gate
                                else "native"
                            )
                        self.assertEqual(result, expected)

    def test_fused_kernel_receives_local_heads_and_mrope(self):
        # Qwen3.5-27B uses 24 Q heads and 4 KV heads (replicated at TP8).
        for tp in (1, 2, 4, 8):
            for dtype in (torch.float16, torch.bfloat16):
                for gate in (False, True):
                    for text_only, interleaved in (
                        (False, True),
                        (False, False),
                        (True, True),
                    ):
                        for tokens in (1, 80):
                            with self.subTest(
                                tp=tp,
                                dtype=dtype,
                                gate=gate,
                                text_only=text_only,
                                interleaved=interleaved,
                                tokens=tokens,
                            ):
                                nq, nk = 24 // tp, max(1, 4 // tp)
                                qkv = torch.zeros(
                                    tokens,
                                    (nq * (1 + gate) + 2 * nk) * 256,
                                    dtype=dtype,
                                )
                                positions = torch.arange(tokens)
                                if not text_only:
                                    positions = positions[None, :] + torch.tensor(
                                        [[0], [1], [3]]
                                    )
                                cache = (
                                    torch.arange(128 * 64).reshape(128, 64).float()
                                    / 8192
                                )
                                q_weight = torch.linspace(-0.5, 0.5, 256).to(dtype)
                                k_weight = q_weight.flip(0)
                                layer = SimpleNamespace(
                                    qkv_proj=mock.Mock(return_value=(qkv, None)),
                                    num_heads=nq,
                                    num_kv_heads=nk,
                                    head_dim=256,
                                    attn_output_gate=gate,
                                    q_norm=SimpleNamespace(
                                        weight=q_weight, variance_epsilon=1e-6
                                    ),
                                    k_norm=SimpleNamespace(weight=k_weight),
                                    rotary_emb=SimpleNamespace(
                                        cos_sin_cache=cache,
                                        mrope_section=[11, 11, 10],
                                        mrope_interleaved=interleaved,
                                        rotary_dim=64,
                                    ),
                                )
                                outputs = tuple(
                                    qkv.new_full((tokens, heads * 256), value)
                                    for heads, value in (
                                        (nq, 1),
                                        (nk, 2),
                                        (nk, 3),
                                        (nq if gate else 0, 4),
                                    )
                                )
                                with mock.patch.object(
                                    qwen3_5,
                                    "triton_split_qkv_rmsnorm_mrope",
                                    create=True,
                                    return_value=outputs,
                                ) as fused:
                                    result = qwen3_5.Qwen3_5AttentionDecoderLayer.forward_prepare_npu_prefill(
                                        layer, positions, mock.sentinel.hidden_states
                                    )
                                layer.qkv_proj.assert_called_once_with(
                                    mock.sentinel.hidden_states
                                )
                                fused.assert_called_once()
                                args = fused.call_args.kwargs
                                self.assertIs(args["qkv"], qkv)
                                torch.testing.assert_close(
                                    args["q_weight"], 1 + q_weight
                                )
                                torch.testing.assert_close(
                                    args["k_weight"], 1 + k_weight
                                )
                                expanded = (
                                    positions.expand(3, -1) if text_only else positions
                                )
                                torch.testing.assert_close(
                                    args["cos_sin"], cache[expanded].to(dtype)
                                )
                                self.assertTrue(args["cos_sin"].is_contiguous())
                                for name, expected in dict(
                                    num_q_heads=nq,
                                    num_kv_heads=nk,
                                    head_size=256,
                                    eps=1e-6,
                                    mrope_section=[11, 11, 10],
                                    is_interleaved=interleaved,
                                    rope_dim=64,
                                    has_gate=gate,
                                ).items():
                                    self.assertEqual(args[name], expected)
                                for actual, expected in zip(result[:3], outputs[:3]):
                                    self.assertIs(actual, expected)
                                self.assertIs(result[3], outputs[3] if gate else None)

    def test_non_mrope_config_keeps_native_prefill(self):
        layer = SimpleNamespace(
            rotary_emb=SimpleNamespace(), forward_prepare_native=mock.Mock()
        )
        result = qwen3_5.Qwen3_5AttentionDecoderLayer.forward_prepare_npu_prefill(
            layer, mock.sentinel.positions, mock.sentinel.hidden_states
        )
        layer.forward_prepare_native.assert_called_once_with(
            mock.sentinel.positions, mock.sentinel.hidden_states
        )
        self.assertIs(result, layer.forward_prepare_native.return_value)


if __name__ == "__main__":
    unittest.main()
