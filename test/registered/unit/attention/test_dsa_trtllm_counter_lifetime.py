import gc
import unittest
import weakref
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTrtllmCounterBufferLifetime(CustomTestCase):
    """Decode CUDA graphs capture the raw address of the trtllm-gen multi-CTA
    KV counter buffer. When a larger eager prefill grows the buffer, the
    backend must keep the superseded tensor alive; otherwise every later graph
    replay writes through freed storage."""

    def _backend(self, initial_bytes=64):
        backend = MagicMock(spec=DeepseekSparseAttnBackend)
        backend.device = "cpu"
        backend.num_q_heads = 16
        backend._multi_ctas_kv_counter_buffer = torch.zeros(
            initial_bytes, dtype=torch.uint8
        )
        backend._retired_multi_ctas_kv_counter_buffers = []
        return backend

    def _ensure(self, backend, batch_size, grow):
        # A plain function replaces the helper so nothing records (and thereby
        # keeps alive) the buffer that was passed in.
        with patch(
            "sglang.srt.layers.attention.dsa_backend."
            "grow_multi_ctas_kv_counter_buffer_if_needed",
            new=grow,
        ):
            DeepseekSparseAttnBackend._ensure_multi_ctas_kv_counter_buffer(
                backend, batch_size
            )

    def test_growth_retains_captured_buffer(self):
        backend = self._backend()
        captured = weakref.ref(backend._multi_ctas_kv_counter_buffer)
        larger = torch.zeros(128, dtype=torch.uint8)

        self._ensure(backend, 16384, grow=lambda *_: larger)
        gc.collect()

        self.assertIs(backend._multi_ctas_kv_counter_buffer, larger)
        self.assertIsNotNone(captured())
        self.assertEqual(backend._retired_multi_ctas_kv_counter_buffers, [captured()])

    def test_no_growth_leaves_ownership_unchanged(self):
        backend = self._backend()
        current = backend._multi_ctas_kv_counter_buffer

        self._ensure(backend, 32, grow=lambda buffer, *_: buffer)

        self.assertIs(backend._multi_ctas_kv_counter_buffer, current)
        self.assertEqual(backend._retired_multi_ctas_kv_counter_buffers, [])

    def test_repeated_growth_retains_every_generation(self):
        backend = self._backend()
        first = backend._multi_ctas_kv_counter_buffer
        second = torch.zeros(128, dtype=torch.uint8)
        third = torch.zeros(256, dtype=torch.uint8)

        self._ensure(backend, 16384, grow=lambda *_: second)
        self._ensure(backend, 32768, grow=lambda *_: third)

        self.assertIs(backend._multi_ctas_kv_counter_buffer, third)
        self.assertEqual(
            backend._retired_multi_ctas_kv_counter_buffers, [first, second]
        )

    def test_retired_buffers_die_with_backend(self):
        backend = self._backend()
        captured = weakref.ref(backend._multi_ctas_kv_counter_buffer)
        larger = torch.zeros(128, dtype=torch.uint8)

        self._ensure(backend, 16384, grow=lambda *_: larger)
        del backend
        gc.collect()

        self.assertIsNone(captured())


if __name__ == "__main__":
    unittest.main()
