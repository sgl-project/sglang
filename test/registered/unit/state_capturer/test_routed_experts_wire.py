"""Routed expert wire compatibility, bounds, and row layout."""

import base64
import unittest

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.state_capturer.routed_experts_wire import (
    encode_routed_experts_for_wire,
    extract_routed_experts_from_meta_info,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRoutedExpertsWire(CustomTestCase):
    def test_roundtrip_preserves_rows_and_advertises_actual_dtype(self):
        # Noncontiguous input must serialize in logical (position, layer, topk)
        # order, and the consumer's compression setting must not affect decode.
        for dtype_name, maximum, itemsize in (
            ("int32", 65536, 4),
            ("uint16", 65535, 2),
            ("uint8", 255, 1),
        ):
            with self.subTest(dtype=dtype_name):
                source = torch.arange(12, dtype=torch.int32).reshape(2, 3, 2)
                source[1, 2, 1] = maximum
                source = source.transpose(1, 2)
                with envs.SGLANG_ROUTED_EXPERTS_DTYPE.override(dtype_name):
                    payload, advertised = encode_routed_experts_for_wire(source)
                with envs.SGLANG_ROUTED_EXPERTS_DTYPE.override("uint8"):
                    decoded = extract_routed_experts_from_meta_info(
                        {
                            "meta_info": {
                                "routed_experts": payload,
                                "routed_experts_dtype": advertised,
                            }
                        },
                        num_layers=2,
                        topk=3,
                    )
                self.assertEqual(advertised, dtype_name)
                self.assertEqual(len(base64.b64decode(payload)), 12 * itemsize)
                self.assertEqual(decoded.dtype, np.dtype(dtype_name))
                np.testing.assert_array_equal(decoded, source.numpy().reshape(2, 6))

    def test_overflow_and_negative_sentinels_fall_back_without_data_loss(self):
        # The old uint8 path wraps -1 to 255; overflow fallback must describe
        # int32 bytes so consumers never misread them as unsigned routing IDs.
        for dtype_name, bad_id in (
            ("uint8", -1),
            ("uint8", 256),
            ("uint16", -1),
            ("uint16", 65536),
        ):
            with self.subTest(dtype=dtype_name, expert_id=bad_id):
                source = torch.tensor([0, bad_id], dtype=torch.int32)
                with envs.SGLANG_ROUTED_EXPERTS_DTYPE.override(dtype_name):
                    payload, advertised = encode_routed_experts_for_wire(source)
                self.assertEqual(advertised, "int32")
                decoded = extract_routed_experts_from_meta_info(
                    {
                        "meta_info": {
                            "routed_experts": payload,
                            "routed_experts_dtype": advertised,
                        }
                    }
                )
                np.testing.assert_array_equal(decoded, source.numpy())

    def test_legacy_payload_defaults_to_int32_independent_of_local_env(self):
        source = np.array([1, 255, 65536], dtype=np.int32)
        payload = base64.b64encode(source.tobytes()).decode("utf-8")
        for metadata in ({}, {"routed_experts_dtype": None}):
            with self.subTest(metadata=metadata):
                meta_info = {"routed_experts": payload, **metadata}
                with envs.SGLANG_ROUTED_EXPERTS_DTYPE.override("uint8"):
                    decoded = extract_routed_experts_from_meta_info(
                        {"meta_info": meta_info}
                    )
                np.testing.assert_array_equal(decoded, source)

    def test_empty_capture_retains_zero_positions(self):
        source = torch.empty((0, 2, 3), dtype=torch.int32)
        with envs.SGLANG_ROUTED_EXPERTS_DTYPE.override("uint8"):
            payload, advertised = encode_routed_experts_for_wire(source)
        decoded = extract_routed_experts_from_meta_info(
            {
                "meta_info": {
                    "routed_experts": payload,
                    "routed_experts_dtype": advertised,
                }
            },
            num_layers=2,
            topk=3,
        )
        self.assertEqual(decoded.shape, (0, 6))
        self.assertEqual(decoded.dtype, np.dtype(np.uint8))

    def test_decoder_rejects_invalid_dtype_and_row_dimensions(self):
        payload = base64.b64encode(np.arange(4, dtype=np.int32).tobytes()).decode()
        for dtype_name in ("float32", "", 8):
            with self.subTest(dtype=dtype_name), self.assertRaises(ValueError):
                extract_routed_experts_from_meta_info(
                    {
                        "meta_info": {
                            "routed_experts": payload,
                            "routed_experts_dtype": dtype_name,
                        }
                    }
                )
        for num_layers, topk in ((1, None), (None, 2), (0, 2), (-2, -2), (2, 3)):
            with self.subTest(layers=num_layers, topk=topk):
                with self.assertRaises(ValueError):
                    extract_routed_experts_from_meta_info(
                        {"meta_info": {"routed_experts": payload}},
                        num_layers=num_layers,
                        topk=topk,
                    )

    def test_encoder_rejects_unsupported_configuration(self):
        # An invalid configuration must not silently disable compression.
        with envs.SGLANG_ROUTED_EXPERTS_DTYPE.override("float32"):
            with self.assertRaises(ValueError):
                encode_routed_experts_for_wire(torch.tensor([1], dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
