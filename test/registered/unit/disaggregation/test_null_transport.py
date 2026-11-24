import unittest

from sglang.srt.disaggregation.mooncake.buffers import (
    BufferKind,
    MooncakeBufferKey,
    copy_ranges,
)
from sglang.srt.disaggregation.transport.base import Status
from sglang.srt.disaggregation.transport.null import NullTransport
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

KEY = MooncakeBufferKey(BufferKind.KV, 0)
OTHER_KEY = MooncakeBufferKey(BufferKind.KV, 7)


def _prefill_with_decode_peer(*, src_length: int, dst_length: int):
    prefill, decode = NullTransport(), NullTransport()
    prefill.register(key=KEY, ptr=0x1000, length=src_length, device=0)
    decode.register(key=KEY, ptr=0x9000, length=dst_length, device=0)
    remote = prefill.import_registration(lengths=decode.export_registration())
    return prefill, remote


class TestNullTransport(CustomTestCase):
    def test_write_an_engine_would_reject_is_failed_on_its_handle(self):
        """A protocol bug that addresses an unregistered buffer, or bytes past the
        end of one, must surface in CPU CI as FAILED rather than pass as DONE."""
        prefill, remote = _prefill_with_decode_peer(src_length=1024, dst_length=2048)
        rejected = {
            "source": (OTHER_KEY, KEY, (0, 0, 8)),
            "destination": (KEY, OTHER_KEY, (0, 0, 8)),
            "reads": (KEY, KEY, (1000, 0, 100)),
            "writes": (KEY, KEY, (0, 2000, 100)),
            # src_offset + length is 8 in uint64 arithmetic, inside the buffer.
            "past its length": (KEY, KEY, (2**64 - 8, 0, 16)),
        }
        for word, (src, dst, block) in rejected.items():
            with self.subTest(word):
                handle = prefill.write(remote, src, dst, copy_ranges([block]))
                result = prefill.status(handle)
                self.assertIs(result.status, Status.FAILED)
                self.assertIn(word, result.reason)
        self.assertEqual(prefill.writes, [])

    def test_write_from_a_deregistered_buffer_fails(self):
        prefill, remote = _prefill_with_decode_peer(src_length=1024, dst_length=1024)
        prefill.deregister([KEY])

        handle = prefill.write(remote, KEY, KEY, copy_ranges([(0, 0, 8)]))

        self.assertIs(prefill.status(handle).status, Status.FAILED)

    def test_final_status_cannot_change(self):
        """DONE and FAILED are final in the Transport contract, so a test cannot
        script a completion that no engine could deliver after a failure."""
        prefill, remote = _prefill_with_decode_peer(src_length=1024, dst_length=1024)
        prefill.hold()
        handle = prefill.write(remote, KEY, KEY, copy_ranges([(0, 0, 8)]))
        self.assertEqual(prefill.pending(), [handle])

        prefill.fail(handle, reason="link down")

        with self.assertRaises(RuntimeError):
            prefill.complete(handle)
        self.assertEqual(prefill.status(handle).reason, "link down")
        self.assertEqual(prefill.pending(), [])


if __name__ == "__main__":
    unittest.main()
