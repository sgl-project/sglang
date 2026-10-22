"""The SSU dispatcher strides verify snapshots by the buffer's step pitch.

Bug: the kernels address snapshot ``(slot, step)`` as
``slot * cache_steps + step`` and the accept commit reads
``buffer[slot, step]``, but Mamba2's target verify passed the verify width as
``cache_steps``. A verify narrower than the buffer (adaptive speculative steps)
then wrote every slot but slot 0 into another slot's rows. The dispatcher must
hand every backend the buffer's per-slot pitch, whatever the caller passed.
"""

import unittest

import torch

from sglang.kernels.ops.mamba.triton_ops import ssu_dispatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Recorder(ssu_dispatch.MambaSSUBackend):
    name = "recorder"

    def __call__(self, *args, **kwargs):
        self.cache_steps = kwargs["cache_steps"]


class TestSSUDispatchCacheSteps(CustomTestCase):
    def setUp(self):
        saved = ssu_dispatch._mamba_ssu_backend
        self.addCleanup(setattr, ssu_dispatch, "_mamba_ssu_backend", saved)
        ssu_dispatch._mamba_ssu_backend = self.backend = _Recorder()

    def _dispatch(self, buffer, cache_steps):
        t = torch.zeros(1)
        ssu_dispatch.selective_state_update(
            t, t, t, t, t, t,
            intermediate_states_buffer=buffer,
            cache_steps=cache_steps,
        )  # fmt: skip
        return self.backend.cache_steps

    def test_pitch_comes_from_the_buffer(self):
        buffer = torch.zeros(4, 8, 2, 3, 5)  # [slots, steps, heads, dim, dstate]
        self.assertEqual(self._dispatch(buffer, cache_steps=3), 8)
        self.assertEqual(self._dispatch(buffer[:, :3], cache_steps=3), 8)

    def test_without_a_buffer_cache_steps_passes_through(self):
        self.assertEqual(self._dispatch(None, cache_steps=3), 3)


if __name__ == "__main__":
    unittest.main()
