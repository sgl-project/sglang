import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost


class TestAscendMambaAsyncLayout(unittest.TestCase):
    def test_accepts_contiguous_slot_payload(self):
        state = torch.empty(3, 5, 2, 4)
        self.assertIsNone(
            MambaPoolHost._dense_device_slot_payload_unavailable_reason(state, 0)
        )

    def test_accepts_nextn_dense_permutation(self):
        state = torch.empty(3, 5, 2, 4).transpose(-1, -2)
        self.assertFalse(state.is_contiguous())
        self.assertIsNone(
            MambaPoolHost._dense_device_slot_payload_unavailable_reason(state, 0)
        )

    def test_rejects_slot_payload_with_holes(self):
        state = torch.empty(3, 5, 2, 4)[..., ::2]
        reason = MambaPoolHost._dense_device_slot_payload_unavailable_reason(state, 0)
        self.assertIsNotNone(reason)
        self.assertIn("not physically dense", reason)

    def test_conv_only_components_skip_empty_temporal_state(self):
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.temporal_state_elem_size = 0
        pool.temporal_buffer = torch.empty(4, 3, 1, 0)
        pool.conv_buffer = [torch.empty(4, 3, 1, 2, 4)]
        device_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.empty(3, 4, 0),
                conv=[torch.empty(3, 4, 2, 4)],
            )
        )

        device_states, host_states = pool._state_components(device_pool)

        self.assertEqual(len(device_states), 1)
        self.assertEqual(len(host_states), 1)
        self.assertIs(device_states[0], device_pool.mamba_cache.conv[0])
        self.assertIs(host_states[0], pool.conv_buffer[0])


if __name__ == "__main__":
    unittest.main()
