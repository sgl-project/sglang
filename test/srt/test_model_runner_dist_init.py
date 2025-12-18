# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for model_runner distributed initialization method generation."""

import unittest

from sglang.srt.utils import configure_ipv6


class TestDistInitMethodGeneration(unittest.TestCase):
    """Test the logic for generating dist_init_method in model_runner.

    This tests the fix for issue #15385 where data-parallel-size > 1
    in multi-node mode would cause port conflicts.
    """

    def _generate_dist_init_method(self, dist_init_addr, dist_port):
        """Simulate the logic in model_runner.init_torch_distributed()."""
        if dist_init_addr:
            if dist_init_addr.startswith("["):  # IPv6 address
                _, host = configure_ipv6(dist_init_addr)
                dist_init_method = f"tcp://{host}:{dist_port}"
            else:  # IPv4 address
                host = dist_init_addr.split(":")[0]
                dist_init_method = f"tcp://{host}:{dist_port}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{dist_port}"
        return dist_init_method

    def test_no_dist_init_addr(self):
        """Test when dist_init_addr is None (single node mode)."""
        result = self._generate_dist_init_method(None, 25001)
        self.assertEqual(result, "tcp://127.0.0.1:25001")

    def test_ipv4_dist_init_addr(self):
        """Test with IPv4 dist_init_addr.

        The host should be extracted and combined with dist_port,
        not using the port from dist_init_addr.
        """
        # This is the key test for issue #15385
        # Even though dist_init_addr has port 33333, we should use dist_port (25001)
        result = self._generate_dist_init_method("10.86.158.116:33333", 25001)
        self.assertEqual(result, "tcp://10.86.158.116:25001")

    def test_ipv4_dist_init_addr_different_ports(self):
        """Test that different dist_ports produce different addresses.

        This simulates multiple DP ranks using the same dist_init_addr
        but different dist_ports (nccl_ports).
        """
        dist_init_addr = "192.168.1.100:30000"

        result1 = self._generate_dist_init_method(dist_init_addr, 25001)
        result2 = self._generate_dist_init_method(dist_init_addr, 25002)

        self.assertEqual(result1, "tcp://192.168.1.100:25001")
        self.assertEqual(result2, "tcp://192.168.1.100:25002")
        self.assertNotEqual(result1, result2)

    def test_ipv6_dist_init_addr(self):
        """Test with IPv6 dist_init_addr."""
        result = self._generate_dist_init_method("[2001:db8::1]:33333", 25001)
        self.assertEqual(result, "tcp://[2001:db8::1]:25001")

    def test_ipv6_dist_init_addr_different_ports(self):
        """Test that different dist_ports produce different addresses for IPv6."""
        dist_init_addr = "[2001:db8::1]:30000"

        result1 = self._generate_dist_init_method(dist_init_addr, 25001)
        result2 = self._generate_dist_init_method(dist_init_addr, 25002)

        self.assertEqual(result1, "tcp://[2001:db8::1]:25001")
        self.assertEqual(result2, "tcp://[2001:db8::1]:25002")
        self.assertNotEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
