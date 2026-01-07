#!/usr/bin/env python3
"""
Long-running E2E test suite for attention exploration.

This test runs for ~60 minutes and:
1. Runs all diverse scenarios against the model
2. Collects attention patterns, fingerprints, MoE data
3. Analyzes manifold classification accuracy
4. Generates insights report for sinq implementation

Usage:
    # Full 60-minute test
    python test_long_running.py --server http://localhost:30000

    # Quick test (subset of scenarios)
    python test_long_running.py --server http://localhost:30000 --quick

    # Specific categories
    python test_long_running.py --server http://localhost:30000 --categories syntax semantic

    # With custom output directory
    python test_long_running.py --server http://localhost:30000 --output ./results
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from scenarios import (
    ALL_SCENARIOS, SCENARIOS_BY_CATEGORY,
    get_random_scenarios, get_balanced_scenarios, Scenario
)
from collector import AttentionCollector, CollectionRunner, CollectionRun
from analyzer import AttentionAnalyzer, RunAnalysis
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Main test runner for long-running E2E tests."""

    def __init__(
        self,
        server_url: str,
        output_dir: Path,
        timeout_minutes: int = 60,
        attention_top_k: int = 32,
    ):
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.timeout_minutes = timeout_minutes
        self.attention_top_k = attention_top_k

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Add file logging
        log_file = self.output_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    async def run_full_suite(self) -> RunAnalysis:
        """Run the full 60-minute test suite."""
        logger.info("=" * 60)
        logger.info("Starting full E2E attention exploration test suite")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Timeout: {self.timeout_minutes} minutes")
        logger.info("=" * 60)

        start_time = time.time()
        max_duration = self.timeout_minutes * 60

        # Build scenario list - multiple rounds
        scenarios = self._build_full_scenario_list(max_duration)
        logger.info(f"Planned {len(scenarios)} scenario runs")

        # Run collection
        run = await self._run_collection(scenarios, max_duration, start_time)

        # Analyze results
        analysis = self._analyze_results(run)

        # Generate report
        self._generate_report(run, analysis)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Test suite completed in {elapsed/60:.1f} minutes")
        logger.info(f"Total scenarios run: {len(run.traces)}")
        logger.info(f"Errors: {len(run.errors)}")
        logger.info("=" * 60)

        return analysis

    async def run_quick_test(self) -> RunAnalysis:
        """Run a quick subset for validation (~5 minutes)."""
        logger.info("Running quick validation test")

        # Get 2 scenarios per category
        scenarios = get_balanced_scenarios(n_per_category=2)
        logger.info(f"Running {len(scenarios)} scenarios")

        run = await self._run_collection(scenarios, max_duration=300)
        analysis = self._analyze_results(run)
        self._generate_report(run, analysis)

        return analysis

    async def run_categories(self, categories: List[str]) -> RunAnalysis:
        """Run specific categories of scenarios."""
        scenarios = []
        for cat in categories:
            if cat in SCENARIOS_BY_CATEGORY:
                scenarios.extend(SCENARIOS_BY_CATEGORY[cat])
            else:
                logger.warning(f"Unknown category: {cat}")

        if not scenarios:
            raise ValueError(f"No valid categories found: {categories}")

        logger.info(f"Running {len(scenarios)} scenarios from categories: {categories}")

        run = await self._run_collection(scenarios, max_duration=self.timeout_minutes * 60)
        analysis = self._analyze_results(run)
        self._generate_report(run, analysis)

        return analysis

    def _build_full_scenario_list(self, max_duration: int) -> List[Scenario]:
        """Build scenario list for full test, repeating to fill time."""
        # Estimate time per scenario (conservative: 30 seconds average)
        time_per_scenario = 30
        max_scenarios = max_duration // time_per_scenario

        scenarios = []

        # First pass: all scenarios once
        scenarios.extend(ALL_SCENARIOS)

        # Additional passes: random sampling to fill time
        while len(scenarios) < max_scenarios:
            # Add balanced random selection
            scenarios.extend(get_balanced_scenarios(n_per_category=1))

            # Add some pure random for variety
            scenarios.extend(get_random_scenarios(5))

        # Trim to max
        return scenarios[:max_scenarios]

    async def _run_collection(
        self,
        scenarios: List[Scenario],
        max_duration: int,
        start_time: Optional[float] = None,
    ) -> CollectionRun:
        """Run scenario collection."""
        if start_time is None:
            start_time = time.time()

        collector = AttentionCollector(
            server_url=self.server_url,
            timeout=120.0,
            attention_top_k=self.attention_top_k,
        )

        try:
            connected = await collector.connect()
            if not connected:
                raise RuntimeError("Failed to connect to server")

            runner = CollectionRunner(collector, self.output_dir)

            # Run with progress tracking
            run = await runner.run_scenarios(
                scenarios,
                progress_callback=lambda i, t, n, tr: self._progress_callback(
                    i, t, n, tr, start_time, max_duration
                ),
            )

            return run

        finally:
            await collector.close()

    def _progress_callback(
        self,
        current: int,
        total: int,
        scenario_name: str,
        trace,
        start_time: float,
        max_duration: int,
    ):
        """Progress callback for scenario runs."""
        elapsed = time.time() - start_time
        remaining = max_duration - elapsed

        # Log progress every 10 scenarios or if near timeout
        if current % 10 == 0 or remaining < 60:
            pct = 100 * current / total
            logger.info(
                f"Progress: {current}/{total} ({pct:.1f}%) - "
                f"Elapsed: {elapsed/60:.1f}m - Remaining: {remaining/60:.1f}m"
            )

        # Check timeout
        if remaining <= 0:
            logger.warning("Approaching timeout, stopping collection")
            raise TimeoutError("Test duration exceeded")

    def _analyze_results(self, run: CollectionRun) -> RunAnalysis:
        """Analyze collection results."""
        logger.info("Analyzing results...")
        analyzer = AttentionAnalyzer()
        return analyzer.analyze_run(run)

    def _generate_report(self, run: CollectionRun, analysis: RunAnalysis):
        """Generate analysis report."""
        logger.info("Generating report...")
        generator = ReportGenerator(self.output_dir)
        generator.generate_full_report(run, analysis)


async def main():
    parser = argparse.ArgumentParser(
        description="Long-running E2E attention exploration tests"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:30000",
        help="SGLang server URL",
    )
    parser.add_argument(
        "--output",
        default="./e2e_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Test timeout in minutes (default: 60)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation test (~5 minutes)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Run specific categories (syntax, semantic, long_range, structure, diffuse, mixed, math)",
    )
    parser.add_argument(
        "--attention-top-k",
        type=int,
        default=32,
        help="Number of top-k attention tokens to capture",
    )

    args = parser.parse_args()

    runner = TestRunner(
        server_url=args.server,
        output_dir=Path(args.output),
        timeout_minutes=args.timeout,
        attention_top_k=args.attention_top_k,
    )

    try:
        if args.quick:
            analysis = await runner.run_quick_test()
        elif args.categories:
            analysis = await runner.run_categories(args.categories)
        else:
            analysis = await runner.run_full_suite()

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Run ID: {analysis.run_id}")
        print(f"Total traces: {len(analysis.trace_analyses)}")
        print(f"\nManifold Accuracy:")
        for zone, acc in analysis.manifold_accuracy.items():
            print(f"  {zone}: {acc:.0%}")
        print(f"\nKey Findings:")
        for finding in analysis.key_findings[:5]:
            print(f"  - {finding}")
        print(f"\nSinq Recommendations:")
        for rec in analysis.sinq_recommendations[:3]:
            print(f"  - {rec}")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
