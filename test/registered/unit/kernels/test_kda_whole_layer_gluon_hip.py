import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention import kda_prefill_gluon_hip as prefill_kda
from sglang.kernels.ops.attention import kda_whole_layer_gluon_hip as whole_kda


class _FakeQuantMethod:
    def __init__(self):
        self.native_calls = 0

    def apply(self, layer, x, bias=None):
        self.native_calls += 1
        return x

    def apply_into(self, layer, x, out, bias=None):
        self.native_calls += 1
        out.zero_()
        return out


class _FakeProjection:
    def __init__(self, use_apply_into: bool):
        self.quant_method = _FakeQuantMethod()
        self.use_apply_into = use_apply_into

    def __call__(self, x):
        if self.use_apply_into:
            out = torch.empty((x.shape[0], 7168), dtype=torch.bfloat16)
            result = self.quant_method.apply_into(self, x, out, bias=None)
        else:
            result = self.quant_method.apply(self, x, bias=None)
        return result, None


class TestKimiK3WholeLayerGluonAdapter(unittest.TestCase):
    def test_explicit_backend_selection(self):
        with mock.patch.dict(
            "os.environ", {"SGLANG_ROCM_K3_KDA_FUSED_BACKEND": "GLUON"}
        ):
            self.assertTrue(whole_kda.enabled())
        with mock.patch.dict(
            "os.environ", {"SGLANG_ROCM_K3_KDA_FUSED_BACKEND": "aiter"}
        ):
            self.assertFalse(whole_kda.enabled())

    def test_shape_family_dispatch(self):
        exact = {
            1: "kda_layer_decode_m1",
            2: "kda_layer_decode_m2",
            4: "kda_layer_decode_m4",
            32: "kda_layer_decode_m32",
            64: "kda_layer_decode_m64",
            128: "kda_layer_decode_m128",
            256: "kda_layer_decode_m256",
        }
        for rows, name in exact.items():
            with self.subTest(rows=rows):
                self.assertEqual(whole_kda._entrypoint_name(rows), name)
        for rows in (3, 8, 16, 31, 33, 63, 65, 127, 129, 255):
            with self.subTest(rows=rows):
                self.assertEqual(
                    whole_kda._entrypoint_name(rows),
                    "kda_layer_decode_m1_256",
                )

    def test_runtime_coverage_is_fail_closed(self):
        hidden = torch.empty((1, 7168), dtype=torch.bfloat16)
        conv = torch.empty((2, 3, 4608), dtype=torch.bfloat16)
        state = torch.empty((2, 12, 128, 128), dtype=torch.float32)
        indices = torch.tensor([1], dtype=torch.int32)
        with mock.patch.object(whole_kda, "available", return_value=True):
            self.assertTrue(whole_kda.covered(hidden, conv, state, indices))
            self.assertFalse(whole_kda.covered(hidden[:, :7167], conv, state, indices))
            self.assertFalse(
                whole_kda.covered(hidden, conv, state, indices.to(torch.int16))
            )

    def test_projection_interception_preserves_native_path(self):
        projection = _FakeProjection(use_apply_into=False)
        whole_kda.bind_output_projection(projection)
        carrier = torch.empty((2, 1536), dtype=torch.bfloat16)

        self.assertIs(projection(carrier)[0], carrier)
        self.assertEqual(projection.quant_method.native_calls, 1)

        expected = torch.ones((2, 7168), dtype=torch.bfloat16)
        output = whole_kda.project_output(
            projection, carrier, lambda output_tensor: expected
        )
        self.assertIs(output, expected)
        self.assertEqual(projection.quant_method.native_calls, 1)

    def test_projection_interception_uses_caller_output(self):
        projection = _FakeProjection(use_apply_into=True)
        whole_kda.bind_output_projection(projection)
        carrier = torch.empty((2, 1536), dtype=torch.bfloat16)

        def invoke(output_tensor):
            self.assertIsNotNone(output_tensor)
            output_tensor.fill_(2)
            return output_tensor

        output = whole_kda.project_output(projection, carrier, invoke)
        self.assertEqual(tuple(output.shape), (2, 7168))
        self.assertTrue(torch.all(output == 2))
        self.assertEqual(projection.quant_method.native_calls, 0)


class TestKimiK3PrefillGluonAdapter(unittest.TestCase):
    def test_prefill_layout(self):
        self.assertTrue(prefill_kda.prefill_layout([512, 512], [0, 17], 1024))
        self.assertFalse(prefill_kda.prefill_layout([], [], 0))
        self.assertFalse(prefill_kda.prefill_layout([1024], [0, 1], 1024))
        self.assertFalse(prefill_kda.prefill_layout([512, 511], [0, 0], 1024))
        self.assertFalse(prefill_kda.prefill_layout([1024], [-1], 1024))

    def test_final_state_tracking_rejects_snapshots(self):
        final_only = type("Metadata", (), {"has_mamba_track_mask": False})()
        snapshots = type("Metadata", (), {"has_mamba_track_mask": True})()
        self.assertTrue(prefill_kda.final_state_tracking(final_only))
        self.assertFalse(prefill_kda.final_state_tracking(snapshots))
        self.assertFalse(prefill_kda.final_state_tracking(None))

    def test_prefill_runtime_coverage_is_fail_closed(self):
        rows = 1024
        hidden = torch.empty((rows, 7168), dtype=torch.bfloat16)
        conv = torch.empty((1, 3, 4608), dtype=torch.bfloat16)
        state = torch.empty((1, 12, 128, 128), dtype=torch.float32)
        indices = torch.tensor([0], dtype=torch.int32)
        cu = torch.tensor([0, rows], dtype=torch.int32)
        prefix = torch.tensor([0], dtype=torch.int32)
        with mock.patch.object(
            prefill_kda.kda_whole_layer_gluon_hip,
            "available",
            return_value=True,
        ):
            self.assertTrue(
                prefill_kda.covered(
                    hidden,
                    [rows],
                    [0],
                    indices,
                    cu,
                    prefix,
                    conv,
                    state,
                )
            )
            self.assertFalse(
                prefill_kda.covered(
                    hidden[:512],
                    [512],
                    [0],
                    indices,
                    torch.tensor([0, 512], dtype=torch.int32),
                    prefix,
                    conv,
                    state,
                )
            )
            self.assertFalse(
                prefill_kda.covered(
                    hidden,
                    [rows],
                    [0],
                    indices.to(torch.int64),
                    cu,
                    prefix,
                    conv,
                    state,
                )
            )


if __name__ == "__main__":
    unittest.main()
