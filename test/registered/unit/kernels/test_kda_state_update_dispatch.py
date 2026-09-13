import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla import chunk_delta_h
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKdaStateUpdateDispatch(CustomTestCase):
    def test_gfx950_fast_config_gate(self):
        valid = dict(
            num_heads=16,
            num_sequences=1,
            num_chunks=128,
            k_dim=128,
            v_dim=128,
            use_gk=True,
            is_varlen=True,
            track_state=None,
        )
        with (
            patch.object(chunk_delta_h, "is_gfx95_supported", return_value=True),
            patch.object(chunk_delta_h, "_GDN_CHUNK_H_CONFIG_OVERRIDDEN", False),
        ):
            for num_heads in (8, 16):
                with self.subTest(num_heads=num_heads):
                    self.assertTrue(
                        chunk_delta_h._use_gfx950_128_config(
                            **(valid | {"num_heads": num_heads})
                        )
                    )
            for field, value in (
                ("num_heads", 4),
                ("num_heads", 32),
                ("num_sequences", 2),
                ("num_chunks", 1),
                ("num_chunks", 2049),
                ("k_dim", 64),
                ("v_dim", 64),
                ("use_gk", False),
                ("is_varlen", False),
                ("track_state", torch.empty(0)),
            ):
                with self.subTest(field=field, value=value):
                    self.assertFalse(
                        chunk_delta_h._use_gfx950_128_config(**(valid | {field: value}))
                    )

    def test_other_hardware_and_explicit_override_keep_default(self):
        kwargs = dict(
            num_heads=16,
            num_sequences=1,
            num_chunks=128,
            k_dim=128,
            v_dim=128,
            use_gk=True,
            is_varlen=True,
            track_state=None,
        )
        with patch.object(chunk_delta_h, "is_gfx95_supported", return_value=False):
            self.assertFalse(chunk_delta_h._use_gfx950_128_config(**kwargs))
        with (
            patch.object(chunk_delta_h, "is_gfx95_supported", return_value=True),
            patch.object(chunk_delta_h, "_GDN_CHUNK_H_CONFIG_OVERRIDDEN", True),
        ):
            self.assertFalse(chunk_delta_h._use_gfx950_128_config(**kwargs))


if __name__ == "__main__":
    unittest.main(verbosity=3)
