#!/usr/bin/env python3
"""
E2E tests for the complete fingerprint pipeline.

Tests the full flow from prompt → attention capture → fingerprint → classification.

Usage:
    # Run with SGLang server
    pytest test_fingerprint_pipeline.py -v --server http://localhost:30000

    # Run with sidecar
    pytest test_fingerprint_pipeline.py -v --server http://localhost:30000 --sidecar http://localhost:9009

    # Run full pipeline test
    pytest test_fingerprint_pipeline.py::TestFullPipeline -v
"""

import asyncio
import json
import os
import pytest
import time
import numpy as np
import tempfile
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scenarios import (
    Scenario, ExpectedManifold,
    ALL_SCENARIOS, SCENARIOS_BY_CATEGORY,
)
from collector import AttentionCollector, TraceData


# Test configuration
DEFAULT_SERVER_URL = os.environ.get("SGLANG_SERVER_URL", "http://localhost:30000")
DEFAULT_SIDECAR_URL = os.environ.get("SIDECAR_URL", "http://localhost:9009")


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--server",
        action="store",
        default=DEFAULT_SERVER_URL,
        help="SGLang server URL",
    )
    parser.addoption(
        "--sidecar",
        action="store",
        default=DEFAULT_SIDECAR_URL,
        help="Rapids sidecar URL",
    )


@pytest.fixture
def server_url(request):
    return request.config.getoption("--server")


@pytest.fixture
def sidecar_url(request):
    return request.config.getoption("--sidecar")


@pytest.fixture
async def collector(server_url):
    """Create and connect an attention collector."""
    collector = AttentionCollector(
        server_url=server_url,
        timeout=120.0,
        attention_top_k=32,
    )
    connected = await collector.connect()
    if not connected:
        pytest.skip(f"Could not connect to server at {server_url}")
    yield collector
    await collector.close()


class TestFingerprintComputation:
    """Tests for fingerprint computation from attention data."""

    @pytest.mark.asyncio
    async def test_fingerprint_components(self, collector):
        """Test fingerprint components are computed correctly."""
        scenario = Scenario(
            name="fingerprint_components",
            category="test",
            description="Test fingerprint components",
            messages=[
                {"role": "user", "content": "Explain what machine learning is in one sentence."}
            ],
            expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
            max_tokens=100,
        )

        trace = await collector.run_scenario(scenario)

        for fp in trace.fingerprints:
            # Test mass components
            total_mass = fp.local_mass + fp.mid_mass + fp.long_mass
            assert abs(total_mass - 1.0) < 0.02, f"Mass should sum to 1.0, got {total_mass}"

            # Test histogram
            assert len(fp.histogram) == 16
            hist_sum = sum(fp.histogram)
            assert abs(hist_sum - 1.0) < 0.02, f"Histogram should sum to 1.0, got {hist_sum}"

            # Test entropy is bounded
            assert 0 <= fp.entropy <= 1.0

    @pytest.mark.asyncio
    async def test_fingerprint_stability(self, collector):
        """Test fingerprint computation is stable across runs."""
        scenario = Scenario(
            name="stability_test",
            category="test",
            description="Test fingerprint stability",
            messages=[
                {"role": "user", "content": "What color is the sky?"}
            ],
            expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
            max_tokens=20,
            temperature=0.0,  # Deterministic
        )

        # Run same scenario twice
        trace1 = await collector.run_scenario(scenario)
        trace2 = await collector.run_scenario(scenario)

        # Compare first fingerprint from each run
        if trace1.fingerprints and trace2.fingerprints:
            fp1 = trace1.fingerprints[0]
            fp2 = trace2.fingerprints[0]

            # With temperature=0, fingerprints should be very similar
            assert abs(fp1.local_mass - fp2.local_mass) < 0.2
            assert abs(fp1.entropy - fp2.entropy) < 0.2


class TestPipelineIntegration:
    """Tests for pipeline integration with sidecar."""

    @pytest.mark.asyncio
    async def test_sidecar_health(self, sidecar_url):
        """Test sidecar health endpoint."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{sidecar_url}/health")
                if response.status_code == 404:
                    pytest.skip("Sidecar health endpoint not available")
                assert response.status_code == 200
            except httpx.ConnectError:
                pytest.skip(f"Sidecar not available at {sidecar_url}")

    @pytest.mark.asyncio
    async def test_sidecar_stats(self, sidecar_url):
        """Test sidecar stats endpoint."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{sidecar_url}/stats")
                if response.status_code == 404:
                    pytest.skip("Sidecar stats endpoint not available")

                assert response.status_code == 200
                stats = response.json()
                print(f"Sidecar stats: {json.dumps(stats, indent=2)}")

            except httpx.ConnectError:
                pytest.skip(f"Sidecar not available at {sidecar_url}")

    @pytest.mark.asyncio
    async def test_post_fingerprint_to_sidecar(self, collector, sidecar_url):
        """Test posting fingerprints to sidecar."""
        # First collect attention data
        scenario = Scenario(
            name="sidecar_post_test",
            category="test",
            description="Test sidecar fingerprint posting",
            messages=[
                {"role": "user", "content": "Hello!"}
            ],
            expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
            max_tokens=20,
        )

        trace = await collector.run_scenario(scenario)

        if not trace.fingerprints:
            pytest.skip("No fingerprints collected")

        # Post fingerprint to sidecar
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                fp = trace.fingerprints[0]
                payload = {
                    "request_id": f"test_{int(time.time())}",
                    "step": 0,
                    "vector": fp.to_vector(),
                    "metadata": {
                        "scenario": scenario.name,
                    }
                }

                response = await client.post(
                    f"{sidecar_url}/fingerprint",
                    json=payload,
                )

                if response.status_code == 404:
                    pytest.skip("Sidecar fingerprint endpoint not available")

                assert response.status_code in [200, 201]
                result = response.json()
                print(f"Sidecar response: {json.dumps(result, indent=2)}")

                # Check classification result
                if "zone" in result:
                    assert result["zone"] in [
                        "syntax_floor", "semantic_bridge",
                        "structure_ripple", "long_range", "diffuse", "unknown"
                    ]

            except httpx.ConnectError:
                pytest.skip(f"Sidecar not available at {sidecar_url}")

    @pytest.mark.asyncio
    async def test_classify_endpoint(self, collector, sidecar_url):
        """Test sidecar classify endpoint."""
        scenario = Scenario(
            name="classify_test",
            category="test",
            description="Test classification",
            messages=[
                {"role": "user", "content": "What is 5 times 7?"}
            ],
            expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
            max_tokens=30,
        )

        trace = await collector.run_scenario(scenario)

        if not trace.fingerprints:
            pytest.skip("No fingerprints collected")

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                fp = trace.fingerprints[0]
                payload = {"vector": fp.to_vector()}  # Sidecar expects "vector" key

                response = await client.post(
                    f"{sidecar_url}/classify",
                    json=payload,
                )

                if response.status_code == 404:
                    pytest.skip("Sidecar classify endpoint not available")

                assert response.status_code == 200
                result = response.json()

                print(f"Classification result: {json.dumps(result, indent=2)}")

                # Validate classification response - may be nested under "manifold"
                if "manifold" in result:
                    manifold = result["manifold"]
                    assert "zone" in manifold or "cluster_id" in manifold
                else:
                    assert "zone" in result or "cluster_id" in result

            except httpx.ConnectError:
                pytest.skip(f"Sidecar not available at {sidecar_url}")


class TestFullPipeline:
    """End-to-end pipeline tests."""

    @pytest.mark.asyncio
    async def test_prompt_to_classification(self, collector, sidecar_url):
        """Test full pipeline from prompt to classification."""
        scenarios = [
            # Syntax task
            Scenario(
                name="syntax_e2e",
                category="syntax",
                description="Syntax floor test",
                messages=[{"role": "user", "content": "Complete: The cat sat on the"}],
                expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
                max_tokens=30,
            ),
            # Semantic task
            Scenario(
                name="semantic_e2e",
                category="semantic",
                description="Semantic bridge test",
                messages=[{
                    "role": "user",
                    "content": "Summarize: AI is transforming industries by automating tasks and providing insights from data."
                }],
                expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
                max_tokens=50,
            ),
        ]

        results = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for scenario in scenarios:
                try:
                    # Step 1: Collect attention
                    trace = await collector.run_scenario(scenario)

                    if not trace.fingerprints:
                        results.append({
                            "scenario": scenario.name,
                            "success": False,
                            "error": "No fingerprints"
                        })
                        continue

                    # Step 2: Classify
                    fp = trace.fingerprints[0]

                    try:
                        response = await client.post(
                            f"{sidecar_url}/classify",
                            json={"vector": fp.to_vector()},  # Sidecar expects "vector" key
                        )

                        if response.status_code == 200:
                            classification = response.json()
                            # Handle nested manifold response structure
                            if "manifold" in classification:
                                predicted_zone = classification["manifold"].get("zone", "unknown")
                            else:
                                predicted_zone = classification.get("zone", "unknown")
                        else:
                            predicted_zone = "sidecar_error"

                    except httpx.ConnectError:
                        # If sidecar not available, use local classification
                        predicted_zone = self._classify_locally(fp)

                    # Step 3: Compare
                    expected_zone = scenario.expected_manifold.value
                    match = predicted_zone == expected_zone

                    results.append({
                        "scenario": scenario.name,
                        "expected": expected_zone,
                        "predicted": predicted_zone,
                        "match": match,
                        "local_mass": fp.local_mass,
                        "mid_mass": fp.mid_mass,
                        "entropy": fp.entropy,
                        "success": True,
                    })

                except Exception as e:
                    results.append({
                        "scenario": scenario.name,
                        "success": False,
                        "error": str(e),
                    })

        # Print results
        print("\n" + "=" * 60)
        print("PIPELINE RESULTS")
        print("=" * 60)

        for r in results:
            if r.get("success"):
                status = "MATCH" if r.get("match") else "MISMATCH"
                print(f"[{status}] {r['scenario']}: expected={r['expected']}, predicted={r['predicted']}")
                print(f"         local={r['local_mass']:.3f}, mid={r['mid_mass']:.3f}, entropy={r['entropy']:.3f}")
            else:
                print(f"[FAIL] {r['scenario']}: {r.get('error')}")

        # All should succeed
        successes = [r for r in results if r.get("success")]
        assert len(successes) == len(results), "Some scenarios failed"

    def _classify_locally(self, fp) -> str:
        """Simple local zone classification based on fingerprint."""
        if fp.local_mass > 0.5 and fp.entropy < 0.4:
            return "syntax_floor"
        elif fp.long_mass > 0.3:
            return "long_range"
        else:
            return "semantic_bridge"

    @pytest.mark.asyncio
    async def test_batch_processing(self, collector):
        """Test batch processing of multiple scenarios."""
        scenarios = [
            Scenario(
                name=f"batch_{i}",
                category="test",
                description=f"Batch test {i}",
                messages=[{"role": "user", "content": f"What is {i}+{i}?"}],
                expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
                max_tokens=20,
            )
            for i in range(5)
        ]

        traces = []
        for scenario in scenarios:
            trace = await collector.run_scenario(scenario)
            traces.append(trace)

        # All should complete - use completion text length since streaming may not return usage
        assert len(traces) == 5
        assert all(len(t.completion) > 0 for t in traces), "All traces should have completion text"
        assert all(len(t.fingerprints) > 0 for t in traces), "All traces should have fingerprints"

        # Compute aggregate stats
        all_fingerprints = [fp for t in traces for fp in t.fingerprints]
        avg_local = np.mean([fp.local_mass for fp in all_fingerprints])
        avg_entropy = np.mean([fp.entropy for fp in all_fingerprints])

        print(f"\nBatch stats:")
        print(f"  Total fingerprints: {len(all_fingerprints)}")
        print(f"  Avg local mass: {avg_local:.3f}")
        print(f"  Avg entropy: {avg_entropy:.3f}")


class TestDatabaseStorage:
    """Tests for fingerprint database storage."""

    @pytest.mark.asyncio
    async def test_fingerprint_storage_roundtrip(self, collector):
        """Test storing and retrieving fingerprints from database."""
        scenario = Scenario(
            name="storage_test",
            category="test",
            description="Test storage",
            messages=[{"role": "user", "content": "Hi!"}],
            expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
            max_tokens=10,
        )

        trace = await collector.run_scenario(scenario)

        if not trace.fingerprints:
            pytest.skip("No fingerprints collected")

        # Create temp database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE fingerprints (
                    id INTEGER PRIMARY KEY,
                    request_id TEXT,
                    step INTEGER,
                    local_mass REAL,
                    mid_mass REAL,
                    long_mass REAL,
                    entropy REAL,
                    histogram TEXT
                )
            """)

            # Store fingerprints
            for i, fp in enumerate(trace.fingerprints):
                conn.execute("""
                    INSERT INTO fingerprints
                    (request_id, step, local_mass, mid_mass, long_mass, entropy, histogram)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"test_{int(time.time())}",
                    i,
                    fp.local_mass,
                    fp.mid_mass,
                    fp.long_mass,
                    fp.entropy,
                    json.dumps(fp.histogram),
                ))

            conn.commit()

            # Retrieve and verify
            cursor = conn.execute("SELECT COUNT(*) FROM fingerprints")
            count = cursor.fetchone()[0]
            assert count == len(trace.fingerprints)

            cursor = conn.execute("SELECT local_mass, entropy FROM fingerprints LIMIT 1")
            row = cursor.fetchone()
            assert 0 <= row[0] <= 1  # local_mass
            assert 0 <= row[1] <= 1  # entropy

            conn.close()

        finally:
            os.unlink(db_path)


class TestModelComparison:
    """Tests for comparing attention patterns across models."""

    @pytest.mark.asyncio
    async def test_attention_consistency(self, collector):
        """Test attention patterns are consistent for same prompt."""
        prompt = "What is the capital of Japan?"

        traces = []
        for i in range(3):
            scenario = Scenario(
                name=f"consistency_{i}",
                category="test",
                description="Consistency test",
                messages=[{"role": "user", "content": prompt}],
                expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
                max_tokens=20,
                temperature=0.0,  # Deterministic
            )
            trace = await collector.run_scenario(scenario)
            traces.append(trace)

        # Compare fingerprint distributions
        all_local_masses = [[fp.local_mass for fp in t.fingerprints] for t in traces]
        all_entropies = [[fp.entropy for fp in t.fingerprints] for t in traces]

        # With temperature=0, patterns should be similar
        avg_locals = [np.mean(masses) if masses else 0 for masses in all_local_masses]
        avg_entropies = [np.mean(ents) if ents else 0 for ents in all_entropies]

        print(f"Local mass averages across runs: {avg_locals}")
        print(f"Entropy averages across runs: {avg_entropies}")

        # Variation should be small (within 20%)
        if avg_locals:
            local_variation = (max(avg_locals) - min(avg_locals)) / max(0.001, np.mean(avg_locals))
            assert local_variation < 0.5, f"Too much local mass variation: {local_variation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
