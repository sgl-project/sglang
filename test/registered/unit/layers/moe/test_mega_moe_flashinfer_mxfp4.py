"""Unit tests for the FlashInfer SM107 MegaMoE path of the W4A4 MXFP4 layers."""

import sys
import unittest
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import mega_moe

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclass
class _Tensors:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: Optional[torch.Tensor] = None


def _fake_flashinfer_moe_ep() -> ModuleType:
    module = ModuleType("flashinfer.moe_ep")
    module.MoEEpTensors = _Tensors
    return module


class _FakeLayer:
    """Records the forward and writes a recognizable routed sum into the output."""

    def __init__(self):
        self.calls = []

    def forward(self, t):
        self.calls.append(t)
        t.output.copy_(
            t.hidden_states * t.topk_weights.sum(dim=1, keepdim=True).to(torch.bfloat16)
        )
        return t.output


class TestFlashInferMxfp4MegaMoe(CustomTestCase):
    def test_scales_become_e8m0(self):
        sf = torch.tensor([[0.5, 1.0, 2.0, 2.0**-10]], dtype=torch.float32)
        e8m0 = mega_moe._as_e8m0(sf)
        self.assertEqual(e8m0.dtype, torch.float8_e8m0fnu)
        self.assertEqual(e8m0.view(torch.uint8).tolist(), [[126, 127, 128, 117]])
        raw = torch.tensor([[1, 127, 254]], dtype=torch.uint8)
        self.assertEqual(
            mega_moe._as_e8m0(raw).view(torch.uint8).data_ptr(), raw.data_ptr()
        )
        with self.assertRaises(ValueError):
            mega_moe._as_e8m0(torch.tensor([3.0]))
        with self.assertRaises(ValueError):
            mega_moe._as_e8m0(torch.tensor([-2.0]))

    def test_expert_byte_slabs_alias_each_experts_slice(self):
        # Expert-major with a transposed inner layout, like FlashInfer's FC1 weight.
        base = torch.arange(3 * 4 * 6, dtype=torch.uint8).reshape(3, 4, 6)
        t = base.transpose(1, 2)
        slabs = mega_moe._expert_byte_slabs(t, 3, "w13_weight")
        self.assertEqual(tuple(slabs.shape), (3, 24))
        self.assertTrue(slabs.is_contiguous())
        self.assertEqual(slabs.data_ptr(), t.data_ptr())
        slabs[1] = slabs[2].clone()  # what an EPLB move does
        self.assertTrue(torch.equal(t[1], t[2]))
        with self.assertRaises(ValueError):
            mega_moe._expert_byte_slabs(base[:, :, ::2], 3, "strided")
        with self.assertRaises(ValueError):
            mega_moe._expert_byte_slabs(base.view(torch.uint8)[:2], 3, "count")

    def test_routed_experts_go_to_flashinfer_without_deep_gemm(self):
        layer = _FakeLayer()
        experts = SimpleNamespace(
            _flashinfer_mega_layer=layer, _flashinfer_mega_hidden=4
        )
        x = torch.randn((3, 4), dtype=torch.bfloat16)
        ids = torch.tensor([[0, 5], [1, 2], [7, 3]], dtype=torch.int64)
        weights = torch.full((3, 2), 0.25, dtype=torch.float32)
        # deep_gemm is not importable here: the FlashInfer layer must not need it.
        with patch.dict(
            sys.modules,
            {"flashinfer.moe_ep": _fake_flashinfer_moe_ep(), "deep_gemm": None},
        ):
            out = mega_moe.run_mega_routed_experts(
                experts,
                x,
                ids,
                weights,
                hidden_size=4,
                intermediate_size=8,
                top_k=2,
                num_tokens=3,
                activation_clamp=10.0,
                routed_scaling_factor=2.0,
            )
        (t,) = layer.calls
        self.assertEqual(t.topk_ids.dtype, torch.int32)
        self.assertTrue(torch.equal(t.topk_ids, ids.to(torch.int32)))
        self.assertTrue(torch.equal(t.topk_weights, torch.full((3, 2), 0.5)))
        self.assertEqual(t.output.shape, (3, 4))
        self.assertEqual(t.output.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, x))

    def test_idle_rank_joins_with_zero_rows(self):
        layer = _FakeLayer()
        experts = SimpleNamespace(
            _flashinfer_mega_layer=layer, _flashinfer_mega_hidden=4
        )
        with patch.dict(sys.modules, {"flashinfer.moe_ep": _fake_flashinfer_moe_ep()}):
            out = mega_moe.run_mega_routed_experts(
                experts,
                torch.empty((0, 4), dtype=torch.bfloat16),
                None,
                None,
                hidden_size=4,
                intermediate_size=8,
                top_k=6,
                num_tokens=0,
            )
        (t,) = layer.calls
        self.assertEqual(tuple(t.topk_ids.shape), (0, 6))
        self.assertEqual(t.topk_ids.dtype, torch.int32)
        self.assertEqual(tuple(t.topk_weights.shape), (0, 6))
        self.assertEqual(tuple(out.shape), (0, 4))

    def test_only_the_mxfp4_mma_type_is_selected(self):
        with envs.SGLANG_OPT_MEGA_MOE_MXFP4_FLASHINFER.override(True):
            self.assertTrue(mega_moe._use_flashinfer_mxfp4_mega_moe("mxf4xmxf4"))
            self.assertFalse(mega_moe._use_flashinfer_mxfp4_mega_moe("fp8xfp4"))
            self.assertFalse(mega_moe._use_flashinfer_mxfp4_mega_moe("nvfp4xnvfp4"))
        with envs.SGLANG_OPT_MEGA_MOE_MXFP4_FLASHINFER.override(False):
            self.assertFalse(mega_moe._use_flashinfer_mxfp4_mega_moe("mxf4xmxf4"))

    def test_build_refuses_other_gpus(self):
        experts = SimpleNamespace()
        with patch.object(mega_moe, "_device_sm", 100):
            with self.assertRaisesRegex(RuntimeError, "SM107"):
                mega_moe._build_flashinfer_mxfp4_mega_moe(experts)
        self.assertFalse(hasattr(experts, "_flashinfer_mega_layer"))

    def test_buffers_skip_deep_gemm_and_warm_the_first_flashinfer_layer(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        def moe_with(experts):
            moe = object.__new__(DeepseekV2MoE)
            moe.__dict__["experts"] = experts
            return moe

        fi = [
            SimpleNamespace(
                _flashinfer_mega_layer=MagicMock(), _mega_moe_weights_built=True
            )
            for _ in range(2)
        ]
        model = MagicMock()
        model.modules.return_value = [moe_with(fi[0]), moe_with(fi[1])]
        backend = MagicMock()
        backend.is_megamoe.return_value = True
        with (
            patch.object(mega_moe, "get_moe_a2a_backend", return_value=backend),
            patch.object(mega_moe, "_use_amd_flydsl_mega_moe", return_value=False),
            patch.object(
                mega_moe, "_is_mega_moe_localization_enabled", return_value=False
            ),
            patch.object(mega_moe, "_mega_moe_buffer") as dg_buffer,
            patch.object(mega_moe, "_warmup_flashinfer_mxfp4_mega_moe") as warmup,
        ):
            mega_moe.allocate_mega_moe_buffers(model)
        dg_buffer.assert_not_called()
        warmup.assert_called_once_with(fi[0])


if __name__ == "__main__":
    unittest.main()
