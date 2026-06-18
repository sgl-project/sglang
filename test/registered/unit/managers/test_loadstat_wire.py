"""Wire-contract test for the LoadStat load-snapshot event.

Locks the msgpack array shape the Rust router decoder depends on
(experimental/sgl-router/src/policies/kv_events/engine_load.rs, `LoadStat`):

    ["LoadStat", num_running_reqs, num_waiting_reqs, num_tokens,
     max_total_num_tokens, attn_dp_rank?]

The Rust side hand-encodes these bytes in its own test; this verifies the
Python publisher actually *emits* that shape, so the two sides can't drift.
CPU-only: just exercises the msgspec encoding.
"""

import unittest

import msgspec.msgpack

from sglang.srt.managers.scheduler_components.load_publisher import LoadStat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLoadStatWire(CustomTestCase):
    def test_loadstat_msgpack_array_shape(self):
        # `attn_dp_rank` is stamped by ZmqEventPublisher.publish in production;
        # set it here to assert the full on-the-wire shape.
        stat = LoadStat(
            num_running_reqs=7,
            num_waiting_reqs=3,
            num_tokens=1024,
            max_total_num_tokens=8192,
            attn_dp_rank=2,
        )
        # Same encoder the publisher thread uses (msgspec.msgpack.Encoder).
        raw = msgspec.msgpack.Encoder().encode(stat)
        decoded = msgspec.msgpack.Decoder().decode(raw)

        # tag=True + array_like → [tag, *fields] in declaration order. The Rust
        # decoder reads the tag + first four counts and ignores any trailing
        # fields (attn_dp_rank).
        self.assertEqual(
            decoded,
            ["LoadStat", 7, 3, 1024, 8192, 2],
            "LoadStat wire shape must match the Rust decoder's expectation",
        )

    def test_loadstat_tag_is_class_name(self):
        # The Rust decoder matches the literal tag string "LoadStat"; guard
        # against an accidental msgspec `tag=` override or class rename.
        raw = msgspec.msgpack.Encoder().encode(LoadStat(0, 0, 0, 0))
        decoded = msgspec.msgpack.Decoder().decode(raw)
        self.assertEqual(decoded[0], "LoadStat")


if __name__ == "__main__":
    unittest.main()
