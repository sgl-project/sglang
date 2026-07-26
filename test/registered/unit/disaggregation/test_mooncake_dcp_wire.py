import struct
import unittest

from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _registration_message(
    *, dcp_size=None, dcp_rank=None, kv_layer_ids=None, state_layer_ids=None
):
    # Frame layout: 0-11 base, 12-13 layer ids, 14-15 staging, 16-17 DCP.
    msg = [
        b"None",
        b"127.0.0.1",
        b"12345",
        b"session",
        struct.pack("Q", 1000),
        struct.pack("Q", 2000),
        pack_int_lists([[3000]], "Q"),
        b"3",
        b"4",
        b"4096",
        pack_int_lists([[128]], "I"),
        pack_int_lists([[16]], "I"),
        b"".join(struct.pack("I", layer_id) for layer_id in kv_layer_ids or []),
        pack_int_lists(state_layer_ids or [], "I"),
        b"",
        b"",
    ]
    if dcp_size is not None or dcp_rank is not None:
        msg.extend(
            [
                str(1 if dcp_size is None else dcp_size).encode("ascii"),
                str(0 if dcp_rank is None else dcp_rank).encode("ascii"),
            ]
        )
    return msg


class TestMooncakeDCPWire(CustomTestCase):
    def test_registration_defaults_old_peer_to_dcp_one(self):
        info = KVArgsRegisterInfo.from_zmq(_registration_message())
        self.assertEqual(info.dst_dcp_size, 1)
        self.assertEqual(info.dst_dcp_rank, 0)

    def test_registration_round_trips_dcp_topology(self):
        info = KVArgsRegisterInfo.from_zmq(
            _registration_message(
                dcp_size=4,
                dcp_rank=3,
                kv_layer_ids=[7],
                state_layer_ids=[[4]],
            )
        )
        self.assertEqual(info.dst_dcp_size, 4)
        self.assertEqual(info.dst_dcp_rank, 3)
        self.assertEqual(info.dst_kv_ptrs, [1000])
        self.assertEqual(info.dst_kv_item_len, 4096)
        self.assertEqual(info.dst_kv_layer_ids, [7])
        self.assertEqual(info.dst_state_layer_ids, [[4]])


if __name__ == "__main__":
    unittest.main()
