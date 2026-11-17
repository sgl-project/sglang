import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ExternalLinearBackend(MambaAttnBackendBase):
    """A linear attention backend registered from outside SGLang."""


class TrackConvIndicesTest(unittest.TestCase):
    def _backend(self, conv_states_shape):
        # Skip __init__, which needs a model runner; tracking reads only these.
        backend = ExternalLinearBackend.__new__(ExternalLinearBackend)
        backend.conv_states_shape = conv_states_shape
        backend.device = "cpu"
        return backend

    def _forward_batch(self):
        return SimpleNamespace(
            mamba_track_aligned_lens=lambda: torch.tensor([64, 128]),
            mamba_track_mask=torch.tensor([True, True]),
        )

    def test_unset_conv_states_shape_names_the_backend(self):
        with self.assertRaisesRegex(
            ValueError, "ExternalLinearBackend.*conv_states_shape"
        ):
            self._backend(None)._init_track_conv_indices(
                torch.tensor([0, 100, 300]), self._forward_batch()
            )

    def test_indices_end_at_the_tracked_position(self):
        indices = self._backend((8, 1024, 3))._init_track_conv_indices(
            torch.tensor([0, 100, 300]), self._forward_batch()
        )
        # The 3 inputs before 64 tokens into the first extend and 128 into the second.
        self.assertEqual(indices.tolist(), [[61, 62, 63], [225, 226, 227]])


if __name__ == "__main__":
    unittest.main()
