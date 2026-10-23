"""Scheduler steps never read device schedule values on the host.

Each test runs the same steps twice, once with plain tensors and once with the
schedule and the latents as ``SyncOnRead``, and requires bitwise-equal results.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.schedulers.kandinsky6_piflow import (
    PiflowScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_helios import (
    HeliosScheduler,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import SyncOnRead

_full = torch.full


def _full_without_host_read(size, fill_value, **kwargs):
    if isinstance(fill_value, torch.Tensor):
        raise AssertionError("torch.full reads a tensor fill value on the host")
    return _full(size, fill_value, **kwargs)


def _assert_bitwise_equal(test, actual, expected):
    for result, reference in zip(actual, expected, strict=True):
        test.assertTrue(torch.equal(result.as_subclass(torch.Tensor), reference))


class TestPiflowHostSync(unittest.TestCase):
    def _run(self, on_device):
        scheduler = PiflowScheduler(n_grid=10)
        scheduler.set_timesteps(4)
        scheduler.set_begin_index(0)
        generator = torch.Generator().manual_seed(0)
        sample = torch.randn(2, 5, 4, generator=generator)
        outputs = [torch.randn(2, 5, 40, generator=generator) for _ in range(4)]
        timesteps = scheduler.timesteps
        if on_device:
            scheduler._piflow_raw_timesteps = on_device(scheduler._piflow_raw_timesteps)
            scheduler.sigmas = on_device(scheduler.sigmas)
            timesteps, sample = on_device(timesteps), on_device(sample)
        samples = []
        for timestep, output in zip(timesteps, outputs):
            sample = scheduler.step(output, timestep, sample, return_dict=False)[0]
            samples.append(sample)
        return samples

    def test_policy_rollout_never_reads_device_substeps(self):
        expected = self._run(None)
        actual = self._run(lambda t: t.as_subclass(SyncOnRead))
        _assert_bitwise_equal(self, actual, expected)


class TestHeliosHostSync(unittest.TestCase):
    def _run_unipc(self, on_device, dtype):
        scheduler = HeliosScheduler(shift=3.0, solver_order=2, scheduler_type="unipc")
        scheduler.set_timesteps(6, device="cpu")
        generator = torch.Generator().manual_seed(1)
        sample = torch.randn(1, 4, 2, 3, 3, generator=generator).to(dtype)
        outputs = [
            torch.randn(1, 4, 2, 3, 3, generator=generator).to(dtype) for _ in range(6)
        ]
        timesteps = scheduler.timesteps
        if on_device:
            scheduler.sigmas = on_device(scheduler.sigmas)
            timesteps, sample = on_device(timesteps), on_device(sample)
        samples = []
        for timestep, output in zip(timesteps, outputs):
            sample = scheduler.step(output, timestep, sample, return_dict=False)[0]
            samples.append(sample)
        return samples

    def _run_dmd(self, on_device):
        scheduler = HeliosScheduler(scheduler_type="dmd")
        scheduler.set_timesteps(3, device="cpu")
        generator = torch.Generator().manual_seed(2)
        sample = torch.randn(1, 4, 2, 3, 3, generator=generator)
        noisy = torch.randn(1, 4, 2, 3, 3, generator=generator)
        outputs = [torch.randn(1, 4, 2, 3, 3, generator=generator) for _ in range(3)]
        sigmas, timesteps = scheduler.sigmas, scheduler.timesteps
        if on_device:
            sigmas, timesteps = on_device(sigmas), on_device(timesteps)
            sample, noisy = on_device(sample), on_device(noisy)
        samples = []
        for index, (timestep, output) in enumerate(zip(timesteps, outputs)):
            sample = scheduler.step(
                output,
                timestep,
                sample,
                return_dict=False,
                cur_sampling_step=index,
                dmd_noisy_tensor=noisy,
                dmd_sigmas=sigmas,
                dmd_timesteps=timesteps,
                all_timesteps=timesteps,
            )[0]
            samples.append(sample)
        return samples

    def test_unipc_never_reads_device_coefficients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                expected = self._run_unipc(None, dtype)
                actual = self._run_unipc(lambda t: t.as_subclass(SyncOnRead), dtype)
                _assert_bitwise_equal(self, actual, expected)

    def test_dmd_never_reads_device_timesteps(self):
        expected = self._run_dmd(None)
        with patch.object(torch, "full", _full_without_host_read):
            actual = self._run_dmd(lambda t: t.as_subclass(SyncOnRead))
        _assert_bitwise_equal(self, actual, expected)


if __name__ == "__main__":
    unittest.main()
