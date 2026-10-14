"""DSpark's verify-budget prepare reads the relayed confidence at the forward's
entry, on every device, without a host read of device memory: that read would
wait on the previous forward still running on the forward stream."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import CONFIDENCE_RELAY_RING_LAG, FutureMap
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

POOL = 8
GAMMA = 4


class _DeviceTensor(torch.Tensor):
    """A CPU tensor standing for device memory: reading its values on the host
    fails, as that would wait for the stream that produces it."""

    _HOST_READS = {
        torch.Tensor.item,
        torch.Tensor.tolist,
        torch.Tensor.numpy,
        torch.Tensor.cpu,
        torch.Tensor.__bool__,
        torch.Tensor.__int__,
        torch.Tensor.__index__,
    }

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        to_host = func is torch.Tensor.to and any(
            a == "cpu" or a == torch.device("cpu")
            for a in (*args[1:], *kwargs.values())
        )
        blocking_copy_to_host = (
            func is torch.Tensor.copy_
            and not isinstance(args[0], _DeviceTensor)
            and not kwargs.get("non_blocking", args[2] if len(args) > 2 else False)
        )
        if func in cls._HOST_READS or to_host or blocking_copy_to_host:
            raise AssertionError(f"host read of a device tensor: {func.__name__}")
        return super().__torch_function__(func, types, args, kwargs)


def _device(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(_DeviceTensor)


def _step(slots):
    return SimpleNamespace(
        spec_info=SimpleNamespace(future_indices=_device(torch.tensor(slots))),
        req_pool_indices=_device(torch.tensor(slots)),
        req_pool_indices_cpu=torch.tensor(slots),
    )


class TestConfidencePrepareReadsNoDeviceValue(CustomTestCase):
    def test_non_cuda_prepare_reads_a_covered_ring_copy(self):
        pool = SimpleNamespace(
            req_to_token=torch.zeros((POOL, 8), dtype=torch.int32),
            req_generation=torch.ones(POOL, dtype=torch.int64),
        )
        with patch.object(overlap_utils, "_is_cuda", False):
            # The scheduler passes the device as a string.
            future_map = FutureMap(
                "cpu",
                SpeculativeAlgorithm.DSPARK,
                pool,
                needs_confidence_relay=True,
            )
        relay = future_map.confidence_relay
        slots = [1, 3]
        published = []
        for step_id in range(CONFIDENCE_RELAY_RING_LAG + 1):
            confidence = torch.full((len(slots), GAMMA), 0.1 * (step_id + 1))
            published.append(confidence)
            future_map.publish(
                _device(torch.tensor(slots)),
                _device(torch.tensor([10, 11])),
                _device(confidence),
                indices_cpu=torch.tensor(slots),
            )
            relay.confidence_buf = _device(relay.confidence_buf)

        resolved = future_map.resolve_confidence_cpu(_step(slots))

        self.assertIsNotNone(resolved)
        self.assertNotIsInstance(resolved.confidence, _DeviceTensor)
        torch.testing.assert_close(
            resolved.confidence.as_subclass(torch.Tensor),
            published[len(published) - CONFIDENCE_RELAY_RING_LAG],
        )


if __name__ == "__main__":
    unittest.main()
