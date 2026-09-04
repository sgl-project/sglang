# SPDX-License-Identifier: Apache-2.0
"""`_ipc_input_a2a_qkv` must decline cross-attention shapes.

It sizes one staging slot from `q` and reuses it for q, k and v, which only
holds when all three share a sequence length. Cross-attention with unequal
query and key/value lengths -- LTX-2's video-to-audio blocks, say -- has to fall
back to the general exchange, which handles them.
"""

import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers import usp


class TestIpcInputA2AQkvGuard(unittest.TestCase):
    def _call(self, q, k, v):
        # Pretend ulysses degree 2 so the guard, not the degree check, decides.
        with mock.patch.object(usp, "get_ulysses_parallel_world_size", lambda: 2):
            return usp._ipc_input_a2a_qkv(q, k, v)

    def test_declines_when_kv_length_differs(self):
        q = torch.zeros(1, 1530, 8, 64)
        kv = torch.zeros(1, 43, 8, 64)
        self.assertIsNone(self._call(q, kv, kv))

    def test_declines_when_only_v_differs(self):
        q = torch.zeros(1, 128, 8, 64)
        k = torch.zeros(1, 128, 8, 64)
        v = torch.zeros(1, 64, 8, 64)
        self.assertIsNone(self._call(q, k, v))

    def test_self_attention_shapes_reach_the_ipc_path(self):
        # Without a real IPC group this returns None either way, so patch the
        # group lookup to prove the guard is not what rejected it.
        q = torch.zeros(1, 128, 8, 64)
        group = mock.MagicMock(return_value=None)
        with (
            mock.patch.object(usp, "get_ulysses_parallel_world_size", lambda: 2),
            mock.patch.object(usp, "_ipc_ready_group", group),
        ):
            self.assertIsNone(usp._ipc_input_a2a_qkv(q, q.clone(), q.clone()))
        # Reached the group lookup, so the shape guard did not reject it.
        self.assertEqual(group.call_count, 1)

    def test_degree_other_than_two_declines(self):
        q = torch.zeros(1, 128, 8, 64)
        with mock.patch.object(usp, "get_ulysses_parallel_world_size", lambda: 4):
            self.assertIsNone(usp._ipc_input_a2a_qkv(q, q.clone(), q.clone()))


if __name__ == "__main__":
    unittest.main()
