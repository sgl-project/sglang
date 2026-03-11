# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for topology_aware EPLB implementation.

This tests the topology-aware Expert Parallel Load Balancing algorithm
that distributes redundant expert copies across different nodes to
maximize NVLink-local expert availability and reduce inter-node RDMA traffic.
"""

import pytest
import torch

from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm, compute_algorithm
from sglang.srt.eplb.eplb_algorithms.deepseek import (
    rebalance_experts,
    rebalance_experts_hierarchical,
    rebalance_experts_hierarchical_topology_aware,
    replicate_experts_topology_aware,
)


class TestEplbAlgorithmEnum:
    """Tests for EplbAlgorithm enum."""

    def test_topology_aware_exists(self):
        """Test that topology_aware is in the EplbAlgorithm enum."""
        assert hasattr(
            EplbAlgorithm, "topology_aware"
        ), "topology_aware should be in EplbAlgorithm"


class TestComputeAlgorithm:
    """Tests for compute_algorithm function."""

    def test_multi_node_returns_topology_aware(self):
        """Test that compute_algorithm returns topology_aware for multi-node."""
        algo = compute_algorithm("auto", num_groups=16, num_nodes=2)
        assert (
            algo == EplbAlgorithm.topology_aware
        ), f"Expected topology_aware for multi-node, got {algo}"

    def test_single_node_returns_deepseek_hierarchical(self):
        """Test that compute_algorithm returns deepseek_hierarchical for single-node."""
        algo = compute_algorithm("auto", num_groups=16, num_nodes=1)
        assert (
            algo == EplbAlgorithm.deepseek_hierarchical
        ), f"Expected deepseek_hierarchical for single-node, got {algo}"

    def test_explicit_algorithm_overrides_auto(self):
        """Test that explicit algorithm choice overrides auto detection."""
        algo = compute_algorithm("deepseek_hierarchical", num_groups=16, num_nodes=2)
        assert (
            algo == EplbAlgorithm.deepseek_hierarchical
        ), f"Expected deepseek_hierarchical when explicitly set, got {algo}"


class TestReplicateExpertsTopologyAware:
    """Tests for replicate_experts_topology_aware function."""

    def test_basic_shapes(self):
        """Test basic output shapes."""
        num_layers = 2
        num_experts = 256
        num_phy = 512  # 2x replication
        num_nodes = 2

        weight = torch.rand(num_layers, num_experts)
        base_node = torch.zeros(num_layers, num_experts, dtype=torch.int64)
        base_node[:, num_experts // 2 :] = 1

        phy2log, phyrank, logcnt = replicate_experts_topology_aware(
            weight, num_phy, num_nodes, base_node
        )

        assert phy2log.shape == (
            num_layers,
            num_phy,
        ), f"phy2log shape mismatch: {phy2log.shape}"
        assert phyrank.shape == (
            num_layers,
            num_phy,
        ), f"phyrank shape mismatch: {phyrank.shape}"
        assert logcnt.shape == (
            num_layers,
            num_experts,
        ), f"logcnt shape mismatch: {logcnt.shape}"

    def test_replica_count(self):
        """Test that each expert has exactly 2 replicas."""
        num_layers = 2
        num_experts = 256
        num_phy = 512
        num_nodes = 2

        weight = torch.rand(num_layers, num_experts)
        base_node = torch.zeros(num_layers, num_experts, dtype=torch.int64)
        base_node[:, num_experts // 2 :] = 1

        phy2log, phyrank, logcnt = replicate_experts_topology_aware(
            weight, num_phy, num_nodes, base_node
        )

        assert (
            logcnt.min().item() == 2
        ), f"Min replicas should be 2, got {logcnt.min().item()}"
        assert (
            logcnt.max().item() == 2
        ), f"Max replicas should be 2, got {logcnt.max().item()}"

    def test_cross_node_coverage(self):
        """Test that topology_aware spreads redundant copies across nodes."""
        num_layers = 1
        num_experts = 16
        num_phy = 32  # 2x replication
        num_nodes = 2
        phy_per_node = num_phy // num_nodes

        weight = torch.ones(num_layers, num_experts)
        base_node = torch.zeros(num_layers, num_experts, dtype=torch.int64)

        phy2log, phyrank, logcnt = replicate_experts_topology_aware(
            weight, num_phy, num_nodes, base_node
        )

        node0_phy2log = phy2log[0, :phy_per_node]
        node1_phy2log = phy2log[0, phy_per_node:]

        node0_experts = set(node0_phy2log.tolist())
        node1_experts = set(node1_phy2log.tolist())

        overlap = node0_experts & node1_experts

        # With topology_aware, redundant copies should be on different nodes
        assert (
            len(overlap) > 0
        ), "Expected redundant copies to be placed on different nodes"


class TestRebalanceExpertsTopologyAware:
    """Tests for rebalance_experts with topology_aware=True."""

    def test_basic_shapes(self):
        """Test basic output shapes."""
        num_layers = 2
        num_experts = 256
        num_replicas = 512
        num_groups = 16
        num_nodes = 2
        num_gpus = 16

        weight = torch.rand(num_layers, num_experts)

        phy2log, log2phy, logcnt = rebalance_experts(
            weight,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
            enable_hierarchical=True,
            topology_aware=True,
        )

        assert phy2log.shape == (num_layers, num_replicas)
        assert logcnt.shape == (num_layers, num_experts)

    def test_direct_function_call(self):
        """Test rebalance_experts_hierarchical_topology_aware directly."""
        num_layers = 2
        num_experts = 256
        num_replicas = 512
        num_groups = 16
        num_nodes = 2
        num_gpus = 16

        weight = torch.rand(num_layers, num_experts)

        phy2log, phyrank, logcnt = rebalance_experts_hierarchical_topology_aware(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )

        assert phy2log.shape == (num_layers, num_replicas)
        assert phyrank.shape == (num_layers, num_replicas)
        assert logcnt.shape == (num_layers, num_experts)


class TestCompareAlgorithms:
    """Compare hierarchical vs topology_aware algorithms."""

    def test_cross_node_coverage_comparison(self):
        """Compare expert placement between hierarchical and topology_aware."""
        num_layers = 1
        num_experts = 64
        num_replicas = 128
        num_groups = 4
        num_nodes = 2
        num_gpus = 8
        phy_per_node = num_replicas // num_nodes

        weight = torch.ones(num_layers, num_experts)

        # Test hierarchical (original)
        phy2log_h, phyrank_h, logcnt_h = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )

        # Test topology_aware
        phy2log_ta, phyrank_ta, logcnt_ta = (
            rebalance_experts_hierarchical_topology_aware(
                weight, num_replicas, num_groups, num_nodes, num_gpus
            )
        )

        def count_cross_node_coverage(phy2log, phy_per_node):
            node0_experts = set(phy2log[0, :phy_per_node].tolist())
            node1_experts = set(phy2log[0, phy_per_node:].tolist())
            return len(node0_experts & node1_experts)

        coverage_h = count_cross_node_coverage(phy2log_h, phy_per_node)
        coverage_ta = count_cross_node_coverage(phy2log_ta, phy_per_node)

        # Topology-aware should have more cross-node coverage
        assert coverage_ta >= coverage_h, (
            f"Topology-aware should have >= cross-node coverage than hierarchical. "
            f"Got: topology_aware={coverage_ta}, hierarchical={coverage_h}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
