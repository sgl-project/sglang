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

    def test_fused_state_output_gate(self):
        shape = (1, 128, 8, 128)
        kwargs = dict(
            q=torch.empty(shape, dtype=torch.bfloat16),
            k=torch.empty(shape, dtype=torch.bfloat16),
            v=torch.empty(shape, dtype=torch.bfloat16),
            w=torch.empty(shape, dtype=torch.bfloat16),
            gk=torch.empty(shape, dtype=torch.float32),
            A=torch.empty((1, 128, 8, 64), dtype=torch.bfloat16),
            initial_state=torch.empty((1, 8, 128, 128), dtype=torch.bfloat16),
            cu_seqlens=torch.tensor([0, 128]),
            num_chunks=2,
        )
        with patch.object(
            chunk_delta_h,
            "is_gfx95_supported",
            return_value=True,
        ):
            self.assertTrue(chunk_delta_h.can_use_fused_kda_state_output(**kwargs))
            for field, value in (
                ("num_chunks", 1),
                ("num_chunks", 5),
                ("cu_seqlens", torch.tensor([0, 64, 128])),
                ("q", torch.empty((1, 128, 4, 128), dtype=torch.bfloat16)),
                ("gk", torch.empty(shape, dtype=torch.bfloat16)),
                ("A", torch.empty((1, 128, 8, 32), dtype=torch.bfloat16)),
            ):
                with self.subTest(field=field):
                    self.assertFalse(
                        chunk_delta_h.can_use_fused_kda_state_output(
                            **(kwargs | {field: value})
                        )
                    )
        with patch.object(
            chunk_delta_h,
            "is_gfx95_supported",
            return_value=False,
        ):
            self.assertFalse(chunk_delta_h.can_use_fused_kda_state_output(**kwargs))


if __name__ == "__main__":
    unittest.main(verbosity=3)
