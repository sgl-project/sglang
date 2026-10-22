"""Regression tests for the MLX ``--disable-overlap-schedule`` crash.

On MLX the FutureMap relay buffers live on the Metal device (``mps``) while
the stub worker materializes sampled tokens on CPU.  With
``--disable-overlap-schedule`` the scheduler still relays decode inputs
through ``FutureMap.stash()``, which used to convert only the dtype.  For
batched scatters (n >= 2) ``torch.index_put_`` requires the value and the
buffer on the same device, so any concurrent decode batch crashed the
scheduler (single-row n == 1 scatters hid the bug via index_put_'s
single-element fast path).
"""

from __future__ import annotations

import platform
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_SKIP_REASON = "requires Apple Silicon with an available MPS device"


def _make_future_map(device: str) -> FutureMap:
    # Minimal pool stand-in: FutureMap only reads req_to_token.shape[0].
    pool = SimpleNamespace(req_to_token=torch.zeros((8, 16), dtype=torch.int64))
    return FutureMap(
        device=torch.device(device),
        spec_algo=SpeculativeAlgorithm.NONE,
        req_to_token_pool=pool,
    )


@unittest.skipUnless(
    _IS_APPLE_SILICON and torch.backends.mps.is_available(), _SKIP_REASON
)
class TestStashCrossDevice(unittest.TestCase):
    """stash() must match both the device and dtype of the relay buffers."""

    def test_stash_cpu_payload_into_mps_relay_buf(self):
        """The MLX non-overlap combination: CPU tokens, mps relay buf, n>=2."""
        future_map = _make_future_map("mps")
        indices = torch.tensor([1, 2], dtype=torch.int64)
        payload = RelayPayload(bonus_tokens=torch.tensor([101, 202], dtype=torch.int64))

        future_map.stash(indices, payload)

        self.assertEqual(future_map.output_tokens_buf[indices].tolist(), [101, 202])

    def test_stash_same_device_payload(self):
        """Sanity guard: the same-device (CUDA-like) path keeps working."""
        future_map = _make_future_map("mps")
        indices = torch.tensor([3, 4], dtype=torch.int64)
        payload = RelayPayload(
            bonus_tokens=torch.tensor([7, 9], dtype=torch.int64, device="mps")
        )

        future_map.stash(indices, payload)

        self.assertEqual(future_map.output_tokens_buf[indices].tolist(), [7, 9])


if __name__ == "__main__":
    unittest.main()
