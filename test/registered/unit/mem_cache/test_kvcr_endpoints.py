"""One source replica's rank ports must not cross endpoint or topology bounds."""

import subprocess
import sys
import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.storage.kvcr import endpoints
from sglang.srt.mem_cache.storage.kvcr.endpoints import rank_endpoint, rank_offset
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKVCRRankEndpoints(CustomTestCase):
    def test_shard_offsets_are_unique_within_one_replica(self):
        config = SimpleNamespace(tp_size=4, attn_cp_size=2)
        offsets = []
        for pp in range(2):
            config.pp_rank = pp
            for cp in range(2):
                config.attn_cp_rank = cp
                for tp in range(4):
                    config.tp_rank = tp
                    offsets.append(rank_offset(config))
        self.assertEqual(offsets, list(range(16)))
        self.assertEqual(
            rank_endpoint("tcp://source.example:32100", offsets[-1]),
            "tcp://source.example:32115",
        )

    def test_ipv6_host_and_zero_offset_remain_unambiguous(self):
        for source, offset, expected in (
            ("tcp://127.0.0.1:32100", 0, "tcp://127.0.0.1:32100"),
            ("tcp://source.example:32100", 2, "tcp://source.example:32102"),
            ("tcp://source.example:32102", -2, "tcp://source.example:32100"),
            ("tcp://[2001:db8::1]:32100", 2, "tcp://[2001:db8::1]:32102"),
            ("tcp://source.example:65535", 0, "tcp://source.example:65535"),
        ):
            with self.subTest(source=source, offset=offset):
                self.assertEqual(rank_endpoint(source, offset), expected)

    def test_invalid_addresses_cannot_select_another_source(self):
        for source in (
            "http://source.example:32100",
            "tcp://source.example",
            "tcp://source.example:bad",
            "tcp://source.example:0",
            "tcp://source.example:65536",
            "tcp://source.example:32100/path",
            "tcp://source.example:32100?query",
            "tcp://source.example:32100#fragment",
            "tcp://user:password@source.example:32100",
            "tcp://[2001:db8::1:32100",
            "tcp://:32100",
            None,
        ):
            with self.subTest(source=source):
                self.assertIsNone(rank_endpoint(source, 1))
        self.assertIsNone(rank_endpoint("tcp://source.example:65535", 1))
        self.assertIsNone(rank_endpoint("tcp://source.example:1", -1))
        for offset in (True, 0.5, "1"):
            with self.subTest(offset=offset):
                self.assertIsNone(rank_endpoint("tcp://source.example:32100", offset))

    def test_shared_helpers_do_not_import_the_optional_runtime(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy,sys; sys.modules['kvcr']=None; "
                "runpy.run_path(sys.argv[1])",
                endpoints.__file__,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
