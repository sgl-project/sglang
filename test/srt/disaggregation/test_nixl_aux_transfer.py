import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.nixl.conn import NixlKVManager


class FakeNixlAgent:
    def __init__(self):
        self.get_xfer_descs_calls = []
        self.initialize_calls = []
        self.transfer_calls = []
        self._handle_counter = 0

    def register_memory(self, *_args, **_kwargs):  # pragma: no cover - not used in test
        return []

    def get_xfer_descs(self, addrs, region):
        self.get_xfer_descs_calls.append((tuple(addrs), region))
        # Return descriptors mirroring inputs for assertions
        return [(addr, length, region) for (addr, length, _gpu) in addrs]

    def initialize_xfer(self, kind, src_descs, dst_descs, peer_name, notif):
        self._handle_counter += 1
        handle = f"handle-{self._handle_counter}"
        self.initialize_calls.append(
            (kind, tuple(src_descs), tuple(dst_descs), peer_name, notif)
        )
        return handle

    def transfer(self, handle):
        self.transfer_calls.append(handle)
        return "DONE"


class TestNixlAuxTransfer(unittest.TestCase):
    def setUp(self):
        self.agent = FakeNixlAgent()
        self.manager = NixlKVManager.__new__(NixlKVManager)
        # Seed only fields touched by send_aux to avoid importing nixl._api
        self.manager.agent = self.agent
        self.manager.kv_args = SimpleNamespace(
            aux_data_ptrs=[1000, 2000, 3000],
            aux_item_lens=[8, 16, 24],
        )

    def test_send_aux_transfers_all_buffers(self):
        dst_aux_ptrs = [5000, 6000, 7000]
        handle = self.manager.send_aux(
            peer_name="decode-node",
            prefill_aux_index=2,
            dst_aux_ptrs=dst_aux_ptrs,
            dst_aux_index=4,
            notif="room_aux",
        )

        # Expect one transfer handle for all aux buffers (batched transfer)
        self.assertEqual(handle, "handle-1")
        self.assertEqual(self.agent.transfer_calls, ["handle-1"])

        expected_src_addrs = [
            (1000 + 2 * 8, 8, 0),
            (2000 + 2 * 16, 16, 0),
            (3000 + 2 * 24, 24, 0),
        ]
        expected_dst_addrs = [
            (5000 + 4 * 8, 8, 0),
            (6000 + 4 * 16, 16, 0),
            (7000 + 4 * 24, 24, 0),
        ]

        # get_xfer_descs called twice (once for all src, once for all dst)
        self.assertEqual(len(self.agent.get_xfer_descs_calls), 2)
        src_call_addrs, src_region = self.agent.get_xfer_descs_calls[0]
        dst_call_addrs, dst_region = self.agent.get_xfer_descs_calls[1]

        self.assertEqual(src_region, "DRAM")
        self.assertEqual(dst_region, "DRAM")
        self.assertEqual(list(src_call_addrs), expected_src_addrs)
        self.assertEqual(list(dst_call_addrs), expected_dst_addrs)

        # One initialize_xfer call with all buffers batched together
        self.assertEqual(len(self.agent.initialize_calls), 1)
        _kind, src_descs, dst_descs, peer, notif = self.agent.initialize_calls[0]

        self.assertEqual(peer, "decode-node")
        self.assertEqual(notif, b"room_aux")
        # Verify all addresses are included in the batch transfer
        self.assertEqual(len(src_descs), 3)
        self.assertEqual(len(dst_descs), 3)
        for idx in range(3):
            self.assertEqual(src_descs[idx][:2], expected_src_addrs[idx][:2])
            self.assertEqual(dst_descs[idx][:2], expected_dst_addrs[idx][:2])


if __name__ == "__main__":
    unittest.main()
