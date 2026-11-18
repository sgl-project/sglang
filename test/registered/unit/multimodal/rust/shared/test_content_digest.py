import os
import sys
import unittest
from pathlib import Path

import blake3

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import load_core  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CORE = load_core()
DIGEST = CORE and CORE.common.content_digest


@unittest.skipUnless(DIGEST, "sglang-mm digest binding not built")
class TestRustContentDigest(CustomTestCase):
    def test_matches_python_blake3(self):
        for size in (0, 1, 63, 64, 65, 1023, 1024, 1025, 65536, (1 << 20) + 7):
            with self.subTest(size=size):
                data = os.urandom(size)
                self.assertEqual(
                    DIGEST(data), "blake3:" + blake3.blake3(data).hexdigest()
                )


if __name__ == "__main__":
    unittest.main()
