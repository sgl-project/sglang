"""LoRA stage dispatch and the lifetime of quantized activation buffers.

Only the SM100 shared-outer NVFP4 decode plan selects it, so this real CPU
matmul is its only executor oracle; the other families run on GPU through the
runner with real routes.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.lora.moe.plan import AFamily, BridgeLayout, Site
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/lora/moe"


def _method(path, name, **scope):
    """Execute the production host method without importing CUDA extensions."""
    tree = ast.parse((SOURCE / path).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, method],
        type_ignores=[],
    )
    exec(compile(module, str(SOURCE / path), "exec"), scope)
    return scope[name]


def test_a_family_writes_its_bridge():
    # Regression: NVFP4 shared plans admitted token_dense before it had an executor.
    spec = SimpleNamespace(
        family=AFamily.TOKEN_DENSE,
        site=Site.GATE_UP,
        is_shared_outer=True,
        output_layout=BridgeLayout.TOKEN_MAJOR,
    )
    owner = SimpleNamespace(
        workspace=SimpleNamespace(
            tensor=lambda _, shape, **kw: torch.empty(shape, **kw)
        ),
        lora_delta_dtype=torch.float32,
    )
    routes = SimpleNamespace(
        raw=lambda _: SimpleNamespace(num_tokens=5, num_rows=10, max_loras=3)
    )
    run = _method(
        "runner.py",
        "_run_a",
        torch=torch,
        AFamily=AFamily,
        BridgeLayout=BridgeLayout,
        Site=Site,
    )
    x = torch.arange(5 * 16, dtype=torch.float32).reshape(5, 16) / 80
    weight = torch.arange(3 * 8 * 16, dtype=torch.float32).reshape(3, 8, 16) / 384
    result = run(
        owner, SimpleNamespace(for_a=lambda _: {}), spec, x, weight, routes, "a"
    )
    # One bridge plane per resident slot, dense over every token.
    torch.testing.assert_close(result, torch.stack([x @ slot.T for slot in weight]))


class TestActQuantReadiness(CustomTestCase):
    @staticmethod
    def _provider(cls, **attributes):
        provider = object.__new__(cls)
        provider.quant_info = SimpleNamespace(
            num_local_experts=2, intermediate_size=128, hidden_size=128
        )
        provider._launch = mock.Mock()
        for name, value in attributes.items():
            setattr(provider, name, value)
        return provider

    @staticmethod
    def _row_state(cls, **fields):
        import msgspec

        names, defaults = cls.__struct_fields__, cls.__struct_defaults__
        # The defaults tuple omits required fields before the first default.
        defaults = (msgspec.NODEFAULT,) * (len(names) - len(defaults)) + defaults
        required = dict.fromkeys(
            name
            for name, default in zip(names, defaults)
            if default is msgspec.NODEFAULT
        )
        return cls(**(required | fields))

    @staticmethod
    def _activate(provider, row_state, *, fused, tile_width=64):
        from sglang.kernels.ops.lora.moe import activation_delta, fused_act

        with (
            mock.patch.object(activation_delta, "act_delta") as act_delta,
            mock.patch.object(fused_act, "fused_b_act") as fused_b_act,
        ):
            if not fused:
                provider.act_with_delta(row_state, None, None, None, None, None)
                return act_delta.call_args.kwargs["act_quant"]
            provider.fused_act(
                row_state,
                activation="silu",
                base_gateup=None,
                act_rows=None,
                act_pairs=None,
                routing=None,
                config={"BLOCK_SIZE_W": tile_width},
            )
            return fused_b_act.call_args.kwargs["act_quant"]

    def test_block_scaled_down_uses_only_rows_written_by_the_last_activation(self):
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_blockscaled as bs

        for per_token in (False, True):
            with self.subTest(per_token=per_token):
                provider = self._provider(
                    bs.CuteDslNvFp4ContiguousProvider,
                    _per_row=per_token,
                    _rows_spec=mock.Mock(return_value=object()),
                    act_out_shape=lambda state: (8, 128),
                    _quantize_rows=mock.Mock(),
                )
                state = self._row_state(
                    bs.CuteDslBlockScaledContiguousRowState,
                    act_q="act_q",
                    act_sf="act_sf",
                    act_rs="act_rs" if per_token else None,
                    pair_to_row="pair_to_row",
                )
                act_out = torch.zeros((8, 128), dtype=torch.bfloat16)
                for fused, width, ready in (
                    (False, 32, True),
                    (True, 24, False),
                    (True, 32, not per_token),
                ):
                    target = self._activate(
                        provider, state, fused=fused, tile_width=width
                    )
                    self.assertEqual(target is not None, ready)
                    self.assertEqual(state.act_quant_ready, ready)
                    provider._quantize_rows.reset_mock()
                    provider.down(state, act_out, "out")
                    self.assertEqual(provider._quantize_rows.called, not ready)

    def test_fp8_down_quantizes_after_fused_activation(self):
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_fp8

        for cls, group in (
            (cutedsl_fp8.CuteDslFp8ContiguousProvider, 128),
            (cutedsl_fp8.CuteDslFp8ChannelContiguousProvider, 256),
        ):
            with (
                self.subTest(provider=cls.__name__),
                mock.patch.object(
                    cutedsl_fp8, "quantize_pair_rows_fp8", return_value=("q", "scales")
                ) as quantize,
            ):
                provider = self._provider(cls)
                provider.quant_info.intermediate_size = 256
                state = self._row_state(
                    provider._row_state_cls,
                    act_fp8="act_fp8",
                    act_scale="act_scale",
                    pair_to_row="pair_to_row",
                )
                self.assertEqual(
                    self._activate(provider, state, fused=False),
                    ("act_fp8", "act_scale", group),
                )
                provider.down(state, "act_out", "out")
                quantize.assert_not_called()
                self.assertIsNone(self._activate(provider, state, fused=True))
                provider.down(state, "act_out", "out")
                quantize.assert_called_once_with(
                    "act_out",
                    q_out="act_fp8",
                    scale_out="act_scale",
                    pair_to_row="pair_to_row",
                    group=group,
                )

    def test_triton_fp8_down_takes_rows_only_from_the_activation_that_just_ran(self):
        from sglang.srt.lora.moe.base_gemm_provider import triton

        provider = self._provider(
            triton.TritonFp8PairProvider,
            act_out_shape=lambda row_state: (4, 128),
            _invoke=mock.Mock(),
        )
        provider.quant_info.w2_weight = provider.quant_info.w2_scale = None
        state = self._row_state(
            triton.TritonRowState,
            pair_to_row=torch.zeros(4, dtype=torch.int32),
            workspace=None,
            down_config={},
        )

        def down_input():
            provider._invoke.reset_mock()
            provider.down(state, "act_out", "out")
            call = provider._invoke.call_args
            return call.args[0], call.kwargs["a_scale"]

        self.assertIsNotNone(self._activate(provider, state, fused=False))
        rows, scales = state.act_quant
        self.assertTrue(state.act_quant_ready)
        self.assertEqual([id(x) for x in down_input()], [id(rows), id(scales)])
        # Fused B/activation leaves stale FP8 buffers, so down quantizes BF16 again.
        self.assertIsNone(self._activate(provider, state, fused=True))
        self.assertIs(state.act_quant[0], rows)
        self.assertFalse(state.act_quant_ready)
        self.assertEqual(down_input(), ("act_out", None))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
