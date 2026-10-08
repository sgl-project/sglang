import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models import qwen3_5
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen3_5PackedVerify(CustomTestCase):
    def setUp(self):
        self.qkvz = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        self.ba = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        self.mixed_qkv = self.qkvz[:, :6]
        self.z = self.qkvz[:, 6:].reshape(2, 1, 2)
        self.b, self.a = self.ba[:, :1], self.ba[:, 1:]
        self.core_out = torch.ones(2, 1, 2)

    def _forward(
        self, backend, *, packed, mode=ForwardMode.TARGET_VERIFY, enabled=True
    ):
        attn = Mock(return_value=(self.core_out, self.z) if packed else self.core_out)
        model = SimpleNamespace(
            _forward_input_proj=lambda hidden_states: (self.qkvz, self.ba),
            num_v_heads=1,
            num_k_heads=1,
            attn_tp_size=1,
            head_k_dim=2,
            head_v_dim=2,
            attn=attn,
            norm=lambda core_out, z: core_out + z,
            out_proj=lambda out: (out, None),
        )
        # Keep the model's routing and output handling real; replace GPU operators.
        with (
            patch.multiple(
                qwen3_5,
                _is_xpu=False,
                _is_cuda=False,
                _is_cpu=True,
                _gdn_decode_fused_proj_conv=enabled,
            ),
            patch.object(
                qwen3_5,
                "fused_qkvzba_split_reshape_cat_contiguous",
                return_value=(self.mixed_qkv, self.z, self.b, self.a),
            ),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            result = qwen3_5.Qwen3_5GatedDeltaNet.forward(
                model, torch.zeros(2, 2), SimpleNamespace(forward_mode=mode)
            )

        torch.testing.assert_close(result, self.z.reshape(2, 2) + 1)
        inputs = attn.call_args.kwargs
        if packed:
            self.assertIsInstance(inputs["mixed_qkv"], tuple)
            self.assertIs(inputs["mixed_qkv"][0], self.qkvz)
            self.assertIs(inputs["mixed_qkv"][1], self.ba)
            self.assertIs(inputs["a"], self.ba)
            self.assertIs(inputs["b"], self.ba)
        else:
            self.assertIs(inputs["mixed_qkv"], self.mixed_qkv)
            self.assertIs(inputs["a"], self.a)
            self.assertIs(inputs["b"], self.b)

    def test_verify_passes_packed_projections_to_opted_in_backend(self):
        backend = SimpleNamespace(supports_packed_verify=True)
        for active_backend in (
            backend,
            SimpleNamespace(linear_attn_backend=backend),
        ):
            with self.subTest(backend=active_backend):
                self._forward(active_backend, packed=True)

    def test_verify_keeps_unpacked_inputs_without_opt_in(self):
        for backend in (
            SimpleNamespace(),
            SimpleNamespace(supports_packed_verify=False),
        ):
            for active_backend in (
                backend,
                SimpleNamespace(linear_attn_backend=backend),
            ):
                with self.subTest(backend=active_backend):
                    self._forward(active_backend, packed=False)

    def test_fusion_flag_still_disables_packed_verify(self):
        self._forward(
            SimpleNamespace(supports_packed_verify=True), packed=False, enabled=False
        )

    def test_other_forward_modes_keep_existing_routing(self):
        backend = SimpleNamespace(supports_packed_verify=True)
        for mode, packed in (
            (ForwardMode.DECODE, True),
            (ForwardMode.EXTEND, False),
            (ForwardMode.DRAFT_EXTEND_V2, False),
        ):
            with self.subTest(mode=mode):
                self._forward(backend, mode=mode, packed=packed)

    def test_packed_verify_rejects_missing_z(self):
        with self.assertRaisesRegex(
            RuntimeError, "Fused GDN projection/Conv1D backend must return"
        ):
            self._forward(SimpleNamespace(supports_packed_verify=True), packed=False)


if __name__ == "__main__":
    unittest.main()
