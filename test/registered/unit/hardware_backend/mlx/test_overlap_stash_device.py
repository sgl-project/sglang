"""Regression tests for the MLX ``--disable-overlap-schedule`` crash.

On MLX the scheduler's device is ``mps`` while the req_to_token_pool, every
ScheduleBatch tensor and the stub worker's sampled tokens live on CPU.  With
``--disable-overlap-schedule`` the scheduler still relays decode inputs
through ``FutureMap.stash()``.  A FutureMap built on the scheduler's device
put its relay buffers on ``mps``, and for batched scatters (n >= 2)
``torch.index_put_`` requires the value and the buffer on the same device, so
any concurrent decode batch crashed the scheduler (single-row n == 1 scatters
hid the bug via index_put_'s single-element fast path).  The FutureMap must
follow the req_to_token_pool's device instead.

Adapted from the tests in #43099.
"""

from __future__ import annotations

import platform
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_SKIP_REASON = "requires Apple Silicon with an available MPS device"


def _init_overlap_mlx_scheduler(device: str) -> Scheduler:
    """Run ``Scheduler.init_overlap`` on a bare MLX-style scheduler.

    ``device`` is the scheduler's accelerator device; the MLX backend keeps
    ``req_to_token_pool`` on CPU regardless.
    """
    scheduler = object.__new__(Scheduler)
    scheduler.device = device
    scheduler.draft_worker = None
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=None)
    )
    scheduler.server_args = SimpleNamespace(
        enable_two_batch_overlap=False,
        cuda_graph_config=None,
        speculative_algorithm=None,
    )
    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
    scheduler.req_to_token_pool = ReqToTokenPool(
        size=8,
        max_context_len=16,
        device="cpu",
        enable_memory_saver=False,
    )
    scheduler.enable_overlap = False

    with mock.patch.object(scheduler_module, "use_mlx", lambda: True):
        Scheduler.init_overlap(scheduler)
    return scheduler


@unittest.skipUnless(
    _IS_APPLE_SILICON and torch.backends.mps.is_available(), _SKIP_REASON
)
class TestStashCrossDevice(unittest.TestCase):
    """The FutureMap relay must live where the scheduler's batches live."""

    @classmethod
    def setUpClass(cls):
        # init_overlap reads the exec config published with the server args.
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_future_map_follows_req_pool_device(self):
        scheduler = _init_overlap_mlx_scheduler(device="mps")

        self.assertEqual(scheduler.future_map.output_tokens_buf.device.type, "cpu")

    def test_stash_cpu_payload_batch(self):
        """The MLX non-overlap combination: CPU tokens, mps scheduler, n>=2."""
        scheduler = _init_overlap_mlx_scheduler(device="mps")
        indices = torch.tensor([1, 2], dtype=torch.int64)
        payload = RelayPayload(bonus_tokens=torch.tensor([101, 202], dtype=torch.int64))

        scheduler.future_map.stash(indices, payload)

        self.assertEqual(
            scheduler.future_map.output_tokens_buf[indices].tolist(), [101, 202]
        )

    def test_stash_single_row(self):
        """Sanity guard: the n == 1 path that already worked keeps working."""
        scheduler = _init_overlap_mlx_scheduler(device="mps")
        indices = torch.tensor([3], dtype=torch.int64)
        payload = RelayPayload(bonus_tokens=torch.tensor([7], dtype=torch.int64))

        scheduler.future_map.stash(indices, payload)

        self.assertEqual(scheduler.future_map.output_tokens_buf[indices].tolist(), [7])


if __name__ == "__main__":
    unittest.main()
