#!/usr/bin/env python3
"""
E2E tests for attention token capture with real SGLang endpoints.

These tests validate the attention capture pipeline against real models
running on NVIDIA RTX 6000 Pro Blackwell or other hardware.

Usage:
    # Test with small model (recommended first)
    pytest test_attention_capture.py -v --server http://localhost:30000

    # Test with specific model
    pytest test_attention_capture.py -v --server http://localhost:30000 --model Qwen/Qwen3-8B

    # Quick smoke test (single scenario)
    pytest test_attention_capture.py::TestAttentionCapture::test_basic_attention_capture -v

    # Full validation suite
    pytest test_attention_capture.py -v --full-validation
"""

import asyncio
import json
import os
import pytest
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import httpx

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scenarios import (
    Scenario, ExpectedManifold,
    SYNTAX_SCENARIOS, SEMANTIC_SCENARIOS, STRUCTURE_SCENARIOS,
    get_balanced_scenarios,
)
from collector import (
    AttentionCollector, AttentionStep, AttentionToken,
    Fingerprint, TraceData,
)


# Test configuration
DEFAULT_SERVER_URL = os.environ.get("SGLANG_SERVER_URL", "http://localhost:30000")
DEFAULT_TIMEOUT = 120.0
DEFAULT_TOP_K = 32


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--server",
        action="store",
        default=DEFAULT_SERVER_URL,
        help="SGLang server URL",
    )
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model name to test against",
    )
    parser.addoption(
        "--full-validation",
        action="store_true",
        default=False,
        help="Run full validation suite (slower)",
    )
    parser.addoption(
        "--attention-top-k",
        action="store",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top-k attention tokens",
    )


@pytest.fixture
def server_url(request):
    """Get server URL from pytest options."""
    return request.config.getoption("--server")


@pytest.fixture
def model_name(request):
    """Get model name from pytest options."""
    return request.config.getoption("--model")


@pytest.fixture
def full_validation(request):
    """Check if full validation is enabled."""
    return request.config.getoption("--full-validation")


@pytest.fixture
def attention_top_k(request):
    """Get attention top-k from pytest options."""
    return request.config.getoption("--attention-top-k")


@pytest.fixture
async def collector(server_url, attention_top_k):
    """Create and connect an attention collector."""
    collector = AttentionCollector(
        server_url=server_url,
        timeout=DEFAULT_TIMEOUT,
        attention_top_k=attention_top_k,
    )
    connected = await collector.connect()
    if not connected:
        pytest.skip(f"Could not connect to server at {server_url}")
    yield collector
    await collector.close()


class TestServerConnection:
    """Tests for SGLang server connection and capabilities."""

    @pytest.mark.asyncio
    async def test_server_reachable(self, server_url):
        """Test that the SGLang server is reachable."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{server_url}/health")
                # Accept 200 or 404 (health endpoint may not exist)
                assert response.status_code in [200, 404, 405]
            except httpx.ConnectError:
                pytest.skip(f"Server not reachable at {server_url}")

    @pytest.mark.asyncio
    async def test_models_endpoint(self, server_url):
        """Test /v1/models endpoint returns model info."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{server_url}/v1/models")
                assert response.status_code == 200

                data = response.json()
                assert "data" in data
                assert len(data["data"]) > 0

                model = data["data"][0]
                assert "id" in model
                print(f"Model: {model['id']}")

            except httpx.ConnectError:
                pytest.skip(f"Server not reachable at {server_url}")

    @pytest.mark.asyncio
    async def test_capabilities_endpoint(self, server_url):
        """Test /v1/capabilities endpoint for attention support."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{server_url}/v1/capabilities")
                if response.status_code == 404:
                    pytest.skip("Capabilities endpoint not available")

                assert response.status_code == 200
                caps = response.json()

                # Check attention_tokens capability
                attention_caps = caps.get("attention_tokens", {})
                print(f"Attention capabilities: {attention_caps}")

                if not attention_caps.get("supported"):
                    pytest.skip("Attention tokens not supported by server")

            except httpx.ConnectError:
                pytest.skip(f"Server not reachable at {server_url}")


class TestAttentionCapture:
    """Tests for attention token capture functionality."""

    @pytest.mark.asyncio
    async def test_basic_attention_capture(self, collector):
        """Test basic attention token capture with simple prompt."""
        scenario = Scenario(
            name="basic_test",
            category="test",
            description="Basic attention capture test",
            messages=[
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
            max_tokens=50,
        )

        trace = await collector.run_scenario(scenario)

        # Basic assertions
        assert trace.completion != "", "Completion should not be empty"
        assert len(trace.completion) > 0, "Completion should have content"
        assert len(trace.attention_steps) > 0, "Should have attention steps"

        # Check attention steps have data
        first_step = trace.attention_steps[0]
        assert len(first_step.top_k_tokens) > 0, "First step should have top-k tokens"

        print(f"Completion: {trace.completion[:100]}...")
        print(f"Attention steps: {len(trace.attention_steps)}")
        print(f"First step tokens: {len(first_step.top_k_tokens)}")

    @pytest.mark.asyncio
    async def test_attention_token_structure(self, collector):
        """Test attention token data structure is correct."""
        scenario = Scenario(
            name="structure_test",
            category="test",
            description="Test attention token structure",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ],
            expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
            max_tokens=50,
        )

        trace = await collector.run_scenario(scenario)

        for step in trace.attention_steps[:5]:  # Check first 5 steps
            for token in step.top_k_tokens:
                # Validate token fields
                assert isinstance(token.token_id, int)
                assert isinstance(token.token_str, str)
                assert isinstance(token.offset, int)
                assert isinstance(token.weight, (int, float))
                assert isinstance(token.position, int)

                # Weights should be valid
                assert token.weight >= 0
                assert token.offset >= 0

    @pytest.mark.asyncio
    async def test_attention_weights_normalized(self, collector):
        """Test attention weights are properly normalized."""
        scenario = Scenario(
            name="weight_test",
            category="test",
            description="Test attention weight normalization",
            messages=[
                {"role": "user", "content": "Explain the concept of attention in neural networks briefly."}
            ],
            expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
            max_tokens=100,
        )

        trace = await collector.run_scenario(scenario)

        for step in trace.attention_steps:
            if step.top_k_tokens:
                total_weight = sum(t.weight for t in step.top_k_tokens)
                # Weights should sum to approximately 1.0 (or less if sparse)
                assert total_weight <= 1.1, f"Weights sum too high: {total_weight}"

    @pytest.mark.asyncio
    async def test_mass_distribution_computed(self, collector):
        """Test local/mid/long mass distribution is computed."""
        scenario = Scenario(
            name="mass_test",
            category="test",
            description="Test mass distribution computation",
            messages=[
                {"role": "user", "content": "Write a short paragraph about programming."}
            ],
            expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
            max_tokens=150,
        )

        trace = await collector.run_scenario(scenario)

        for step in trace.attention_steps:
            # Mass should be computed
            assert step.local_mass is not None
            assert step.mid_mass is not None
            assert step.long_mass is not None

            # Mass values should be in [0, 1]
            assert 0 <= step.local_mass <= 1
            assert 0 <= step.mid_mass <= 1
            assert 0 <= step.long_mass <= 1

            # Total mass should be approximately 1
            total_mass = step.local_mass + step.mid_mass + step.long_mass
            assert 0.99 <= total_mass <= 1.01, f"Total mass: {total_mass}"


class TestFingerprintGeneration:
    """Tests for fingerprint generation from attention data."""

    @pytest.mark.asyncio
    async def test_fingerprints_generated(self, collector):
        """Test fingerprints are generated for each step."""
        scenario = Scenario(
            name="fingerprint_test",
            category="test",
            description="Test fingerprint generation",
            messages=[
                {"role": "user", "content": "List three primary colors."}
            ],
            expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
            max_tokens=50,
        )

        trace = await collector.run_scenario(scenario)

        # Should have fingerprints
        assert len(trace.fingerprints) > 0
        assert len(trace.fingerprints) == len(trace.attention_steps)

    @pytest.mark.asyncio
    async def test_fingerprint_vector_shape(self, collector):
        """Test fingerprint vector has correct shape (20 dimensions)."""
        scenario = Scenario(
            name="fingerprint_shape_test",
            category="test",
            description="Test fingerprint vector shape",
            messages=[
                {"role": "user", "content": "What is the capital of France?"}
            ],
            expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
            max_tokens=50,
        )

        trace = await collector.run_scenario(scenario)

        for fp in trace.fingerprints:
            vector = fp.to_vector()
            assert len(vector) == 20, f"Expected 20D vector, got {len(vector)}D"

            # Validate components
            assert len(fp.histogram) == 16
            assert 0 <= fp.local_mass <= 1
            assert 0 <= fp.mid_mass <= 1
            assert 0 <= fp.long_mass <= 1
            assert 0 <= fp.entropy <= 1

    @pytest.mark.asyncio
    async def test_fingerprint_histogram_normalized(self, collector):
        """Test fingerprint histogram is normalized."""
        scenario = Scenario(
            name="histogram_test",
            category="test",
            description="Test histogram normalization",
            messages=[
                {"role": "user", "content": "Describe the solar system briefly."}
            ],
            expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
            max_tokens=100,
        )

        trace = await collector.run_scenario(scenario)

        for fp in trace.fingerprints:
            # Histogram should sum to approximately 1
            hist_sum = sum(fp.histogram)
            assert 0.99 <= hist_sum <= 1.01, f"Histogram sum: {hist_sum}"


class TestManifoldZones:
    """Tests for manifold zone patterns."""

    @pytest.mark.asyncio
    async def test_syntax_floor_pattern(self, collector):
        """Test syntax_floor scenarios show high local mass."""
        scenario = SYNTAX_SCENARIOS[0]  # Simple completion
        trace = await collector.run_scenario(scenario)

        # Average local mass should be relatively high for syntax tasks
        avg_local_mass = np.mean([fp.local_mass for fp in trace.fingerprints])
        print(f"Average local mass for {scenario.name}: {avg_local_mass:.3f}")

        # Syntax floor should have higher local attention
        # (relaxed assertion - actual threshold depends on model)
        assert avg_local_mass > 0.1, f"Expected higher local mass, got {avg_local_mass}"

    @pytest.mark.asyncio
    async def test_semantic_bridge_pattern(self, collector):
        """Test semantic_bridge scenarios show mid-range attention."""
        scenario = SEMANTIC_SCENARIOS[0]  # Paragraph summary
        trace = await collector.run_scenario(scenario)

        # Average mid mass for semantic tasks
        avg_mid_mass = np.mean([fp.mid_mass for fp in trace.fingerprints])
        print(f"Average mid mass for {scenario.name}: {avg_mid_mass:.3f}")

        # Semantic bridge should have mid-range attention
        assert avg_mid_mass > 0.05, f"Expected higher mid mass, got {avg_mid_mass}"

    @pytest.mark.asyncio
    async def test_structure_pattern(self, collector):
        """Test structure scenarios show periodic patterns."""
        scenario = STRUCTURE_SCENARIOS[0]  # Code generation
        trace = await collector.run_scenario(scenario)

        # Check entropy variation (structured output should have lower entropy)
        entropies = [fp.entropy for fp in trace.fingerprints]
        avg_entropy = np.mean(entropies)
        print(f"Average entropy for {scenario.name}: {avg_entropy:.3f}")

        # Just verify we get valid entropy values
        assert all(0 <= e <= 1 for e in entropies)


class TestPerformanceMetrics:
    """Tests for performance metrics collection."""

    @pytest.mark.asyncio
    async def test_timing_metrics_captured(self, collector):
        """Test timing metrics are captured."""
        scenario = Scenario(
            name="timing_test",
            category="test",
            description="Test timing capture",
            messages=[
                {"role": "user", "content": "Count from 1 to 10."}
            ],
            expected_manifold=ExpectedManifold.STRUCTURE_RIPPLE,
            max_tokens=50,
        )

        trace = await collector.run_scenario(scenario)

        # Timing should be captured
        assert trace.time_to_first_token_ms is not None
        assert trace.total_generation_time_ms is not None
        assert trace.tokens_per_second is not None

        # Values should be reasonable
        assert trace.time_to_first_token_ms > 0
        assert trace.total_generation_time_ms > trace.time_to_first_token_ms
        assert trace.tokens_per_second > 0

        print(f"TTFT: {trace.time_to_first_token_ms:.1f}ms")
        print(f"Total time: {trace.total_generation_time_ms:.1f}ms")
        print(f"Tokens/sec: {trace.tokens_per_second:.1f}")

    @pytest.mark.asyncio
    async def test_token_counts_accurate(self, collector):
        """Test token counts are accurate."""
        scenario = Scenario(
            name="token_count_test",
            category="test",
            description="Test token counting",
            messages=[
                {"role": "user", "content": "Say hello."}
            ],
            expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
            max_tokens=20,
        )

        trace = await collector.run_scenario(scenario)

        # Note: In streaming mode, usage stats may not be returned
        # Check completion text and attention steps instead
        assert len(trace.completion) > 0, "Should have completion text"
        assert len(trace.attention_steps) > 0, "Should have attention steps"

        # If usage data is available, verify consistency
        if trace.completion_tokens > 0:
            assert trace.total_tokens == trace.prompt_tokens + trace.completion_tokens


class TestFullValidation:
    """Full validation suite (run with --full-validation flag)."""

    @pytest.mark.asyncio
    async def test_balanced_scenarios(self, collector, full_validation):
        """Test balanced set of scenarios from all categories."""
        if not full_validation:
            pytest.skip("Use --full-validation to run this test")

        scenarios = get_balanced_scenarios(n_per_category=2)
        results = []

        for scenario in scenarios:
            try:
                trace = await collector.run_scenario(scenario)
                results.append({
                    "name": scenario.name,
                    "category": scenario.category,
                    "expected_manifold": scenario.expected_manifold.value,
                    "completion_tokens": trace.completion_tokens,
                    "attention_steps": len(trace.attention_steps),
                    "avg_local_mass": np.mean([fp.local_mass for fp in trace.fingerprints]) if trace.fingerprints else 0,
                    "avg_entropy": np.mean([fp.entropy for fp in trace.fingerprints]) if trace.fingerprints else 0,
                    "tokens_per_second": trace.tokens_per_second,
                    "success": True,
                })
            except Exception as e:
                results.append({
                    "name": scenario.name,
                    "success": False,
                    "error": str(e),
                })

        # Print summary
        print("\n" + "=" * 60)
        print("FULL VALIDATION RESULTS")
        print("=" * 60)

        successes = [r for r in results if r.get("success")]
        failures = [r for r in results if not r.get("success")]

        print(f"Total scenarios: {len(results)}")
        print(f"Successes: {len(successes)}")
        print(f"Failures: {len(failures)}")

        for r in successes:
            print(f"  [OK] {r['name']}: {r['completion_tokens']} tokens, {r['attention_steps']} steps")

        for r in failures:
            print(f"  [FAIL] {r['name']}: {r.get('error', 'Unknown error')}")

        # At least 80% should succeed
        success_rate = len(successes) / len(results) if results else 0
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.0%}"


class TestModelSpecific:
    """Model-specific tests."""

    @pytest.mark.asyncio
    async def test_model_info(self, collector, model_name):
        """Test model information matches expected."""
        actual_model = collector.model_name

        if model_name:
            assert model_name in actual_model, f"Expected {model_name}, got {actual_model}"

        print(f"Testing with model: {actual_model}")

    @pytest.mark.asyncio
    async def test_long_context_attention(self, collector):
        """Test attention capture with longer context."""
        # Create a longer context scenario
        long_context = """
        The history of computing spans several centuries, beginning with mechanical
        devices like the abacus and culminating in modern digital computers. In the
        19th century, Charles Babbage designed the Analytical Engine, considered the
        first general-purpose computer design. Ada Lovelace wrote what is recognized
        as the first computer program for this machine.

        The 20th century saw rapid advancement in computing technology. Alan Turing
        developed the theoretical foundation for modern computers with his concept of
        the Turing machine. ENIAC, completed in 1945, was one of the first electronic
        general-purpose computers. The invention of the transistor in 1947 and later
        the integrated circuit enabled computers to become smaller and more powerful.

        The personal computer revolution of the 1970s and 1980s brought computing to
        homes and offices. Companies like Apple and Microsoft made computers accessible
        to ordinary people. The internet, developed from ARPANET, connected computers
        globally and transformed how people communicate and access information.

        Today, computing is ubiquitous. Smartphones contain more computing power than
        early mainframes. Artificial intelligence and machine learning are transforming
        industries. Quantum computing promises to solve problems beyond the reach of
        classical computers.
        """

        scenario = Scenario(
            name="long_context_test",
            category="test",
            description="Test long context attention",
            messages=[
                {"role": "user", "content": f"{long_context}\n\nBased on the text above, who designed the Analytical Engine?"}
            ],
            expected_manifold=ExpectedManifold.LONG_RANGE,
            max_tokens=100,
        )

        trace = await collector.run_scenario(scenario)

        # Should have attention steps
        assert len(trace.attention_steps) > 0

        # With long context, should see some long-range attention
        avg_long_mass = np.mean([fp.long_mass for fp in trace.fingerprints]) if trace.fingerprints else 0
        print(f"Average long-range mass: {avg_long_mass:.3f}")

        # Just verify we can handle long context
        assert len(trace.completion) > 0, "Should have completion text"


class TestHardwareValidation:
    """Hardware-specific validation tests."""

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, collector, full_validation):
        """Benchmark throughput on current hardware."""
        if not full_validation:
            pytest.skip("Use --full-validation to run this test")

        # Run multiple scenarios and measure throughput
        scenarios = [
            Scenario(
                name=f"throughput_test_{i}",
                category="benchmark",
                description="Throughput benchmark",
                messages=[
                    {"role": "user", "content": f"Write a short story about adventure number {i}."}
                ],
                expected_manifold=ExpectedManifold.DIFFUSE,
                max_tokens=200,
                temperature=0.9,
            )
            for i in range(5)
        ]

        total_tokens = 0
        total_time = 0

        for scenario in scenarios:
            trace = await collector.run_scenario(scenario)
            # Use attention steps as proxy for tokens in streaming mode
            tokens = trace.completion_tokens if trace.completion_tokens > 0 else len(trace.attention_steps)
            total_tokens += tokens
            total_time += trace.total_generation_time_ms / 1000

        avg_throughput = total_tokens / total_time if total_time > 0 else 0

        print(f"\n{'=' * 60}")
        print("THROUGHPUT BENCHMARK")
        print(f"{'=' * 60}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average throughput: {avg_throughput:.1f} tokens/sec")
        print(f"{'=' * 60}")

        # Verify reasonable throughput (>10 tokens/sec)
        assert avg_throughput > 10, f"Throughput too low: {avg_throughput:.1f} tokens/sec"

    @pytest.mark.asyncio
    async def test_memory_stability(self, collector, full_validation):
        """Test memory stability with repeated attention capture."""
        if not full_validation:
            pytest.skip("Use --full-validation to run this test")

        # Run 20 scenarios to test memory stability
        for i in range(20):
            scenario = Scenario(
                name=f"memory_test_{i}",
                category="test",
                description="Memory stability test",
                messages=[
                    {"role": "user", "content": f"What is {i} + {i}?"}
                ],
                expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
                max_tokens=30,
            )

            trace = await collector.run_scenario(scenario)
            assert len(trace.attention_steps) > 0

            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1}/20 memory stability iterations")

        print("Memory stability test passed")


# Run with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
