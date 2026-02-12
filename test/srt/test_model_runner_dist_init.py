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


def configure_ipv6(dist_init_addr):
    """Extract host and port from IPv6 address format [host]:port."""
    addr = dist_init_addr
    end = addr.find("]")
    if end == -1:
        raise ValueError("invalid IPv6 address format: missing ']'")

    host = addr[: end + 1]

    port_str = None
    if len(addr) > end + 1:
        if addr[end + 1] == ":":
            port_str = addr[end + 2 :]
        else:
            raise ValueError("received IPv6 address format: expected ':' after ']'")

    return port_str, host


class TestDistInitMethodGeneration(unittest.TestCase):
    """Test the logic for generating dist_init_method in model_runner.

    This tests the fix for issue #15385 where data-parallel-size > 1
    in multi-node mode would cause port conflicts.
    """

    def _generate_dist_init_method(
        self, dist_init_addr, dist_port, nnodes=1, tp_rank=0, tp_size=1, dp_size=1
    ):
        """Simulate the logic in model_runner.init_torch_distributed()."""
        if nnodes == 1:
            # Single node: all processes on the same machine, use loopback
            dist_init_method = f"tcp://127.0.0.1:{dist_port}"
        elif tp_rank == 0:
            # Multi-node, master (tp_rank=0): bind to all interfaces
            # This allows connections from both local and remote processes
            dist_init_method = f"tcp://0.0.0.0:{dist_port}"
        else:
            # Multi-node, non-master: connect to the master node
            # Calculate if we're on the same node as the master (tp_rank=0)
            nnodes_per_dp_group = max(nnodes // dp_size, 1)
            tp_size_per_node = tp_size // nnodes_per_dp_group

            # tp_rank 0 to tp_size_per_node-1 are on the first node of each DP group
            master_on_same_node = tp_rank < tp_size_per_node

            if master_on_same_node:
                # Master is on the same node, use loopback
                dist_init_method = f"tcp://127.0.0.1:{dist_port}"
            elif dist_init_addr:
                # Master is on a different node, use dist_init_addr
                if dist_init_addr.startswith("["):  # IPv6
                    _, host = configure_ipv6(dist_init_addr)
                else:  # IPv4 address
                    host = dist_init_addr.split(":")[0]
                dist_init_method = f"tcp://{host}:{dist_port}"
            else:
                # Fallback for edge cases
                dist_init_method = f"tcp://127.0.0.1:{dist_port}"
        return dist_init_method

    def test_single_node_no_dist_init_addr(self):
        """Test when dist_init_addr is None (single node mode)."""
        result = self._generate_dist_init_method(None, 25001, nnodes=1)
        self.assertEqual(result, "tcp://127.0.0.1:25001")

    def test_single_node_with_dist_init_addr(self):
        """Test single node mode ignores dist_init_addr."""
        result = self._generate_dist_init_method("10.86.158.116:33333", 25001, nnodes=1)
        self.assertEqual(result, "tcp://127.0.0.1:25001")

    def test_single_node_different_dp_ranks(self):
        """Test single node with multiple DP ranks uses different ports.

        This simulates multiple DP ranks on a single node, each with
        different nccl_ports but same loopback address.
        """
        # DP rank 0 with nccl_port 25001
        result1 = self._generate_dist_init_method(None, 25001, nnodes=1, tp_rank=0)
        # DP rank 1 with nccl_port 25002
        result2 = self._generate_dist_init_method(None, 25002, nnodes=1, tp_rank=0)

        self.assertEqual(result1, "tcp://127.0.0.1:25001")
        self.assertEqual(result2, "tcp://127.0.0.1:25002")
        self.assertNotEqual(result1, result2)

    def test_multi_node_master_binds_to_all_interfaces(self):
        """Test that tp_rank=0 (master) binds to 0.0.0.0 in multi-node mode."""
        result = self._generate_dist_init_method(
            "10.86.158.116:33333", 25001, nnodes=2, tp_rank=0, tp_size=16, dp_size=1
        )
        self.assertEqual(result, "tcp://0.0.0.0:25001")

    def test_multi_node_non_master_same_node(self):
        """Test non-master on same node as master uses loopback.

        With 2 nodes and tp_size=16, tp_rank 0-7 are on node 0,
        tp_rank 8-15 are on node 1. tp_rank 5 should use loopback.
        """
        result = self._generate_dist_init_method(
            "10.86.158.116:33333", 25001, nnodes=2, tp_rank=5, tp_size=16, dp_size=1
        )
        self.assertEqual(result, "tcp://127.0.0.1:25001")

    def test_multi_node_non_master_different_node(self):
        """Test non-master on different node uses dist_init_addr.

        With 2 nodes and tp_size=16, tp_rank 0-7 are on node 0,
        tp_rank 8-15 are on node 1. tp_rank 10 should connect to dist_init_addr.
        """
        result = self._generate_dist_init_method(
            "10.86.158.116:33333", 25001, nnodes=2, tp_rank=10, tp_size=16, dp_size=1
        )
        self.assertEqual(result, "tcp://10.86.158.116:25001")

    def test_multi_node_multi_dp_different_ports(self):
        """Test multi-node with multiple DP groups use different ports.

        Each DP group has its own nccl_port to avoid conflicts.
        """
        # DP group 0's master (tp_rank=0)
        result1 = self._generate_dist_init_method(
            "10.86.158.116:33333", 25001, nnodes=2, tp_rank=0, tp_size=16, dp_size=2
        )
        # DP group 1's master (tp_rank=0, different port)
        result2 = self._generate_dist_init_method(
            "10.86.158.116:33333", 25002, nnodes=2, tp_rank=0, tp_size=16, dp_size=2
        )

        self.assertEqual(result1, "tcp://0.0.0.0:25001")
        self.assertEqual(result2, "tcp://0.0.0.0:25002")
        self.assertNotEqual(result1, result2)

    def test_multi_node_ipv6_master(self):
        """Test IPv6 address with master binding."""
        result = self._generate_dist_init_method(
            "[2001:db8::1]:33333", 25001, nnodes=2, tp_rank=0, tp_size=16, dp_size=1
        )
        self.assertEqual(result, "tcp://0.0.0.0:25001")

    def test_multi_node_ipv6_non_master_different_node(self):
        """Test IPv6 address for non-master on different node."""
        result = self._generate_dist_init_method(
            "[2001:db8::1]:33333", 25001, nnodes=2, tp_rank=10, tp_size=16, dp_size=1
        )
        self.assertEqual(result, "tcp://[2001:db8::1]:25001")

    def test_four_nodes_two_dp_groups(self):
        """Test 4 nodes with dp=2, each DP group spanning 2 nodes.

        Scenario: 4 nodes, dp=2, tp=16 per DP group
        - DP group 0: nodes 0, 1 (tp_rank 0-15)
        - DP group 1: nodes 2, 3 (tp_rank 0-15)

        Each DP group has nnodes_per_dp_group = 2, tp_size_per_node = 8.
        """
        nnodes = 4
        dp_size = 2
        tp_size = 16  # per DP group

        # DP group's master (tp_rank=0) binds to all interfaces
        result_master = self._generate_dist_init_method(
            "10.0.0.1:33333",
            25001,
            nnodes=nnodes,
            tp_rank=0,
            tp_size=tp_size,
            dp_size=dp_size,
        )
        self.assertEqual(result_master, "tcp://0.0.0.0:25001")

        # tp_rank=5 is on same node as master (within first 8 ranks)
        result_same_node = self._generate_dist_init_method(
            "10.0.0.1:33333",
            25001,
            nnodes=nnodes,
            tp_rank=5,
            tp_size=tp_size,
            dp_size=dp_size,
        )
        self.assertEqual(result_same_node, "tcp://127.0.0.1:25001")

        # tp_rank=10 is on different node from master
        result_diff_node = self._generate_dist_init_method(
            "10.0.0.1:33333",
            25001,
            nnodes=nnodes,
            tp_rank=10,
            tp_size=tp_size,
            dp_size=dp_size,
        )
        self.assertEqual(result_diff_node, "tcp://10.0.0.1:25001")


if __name__ == "__main__":
    unittest.main()
