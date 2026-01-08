#!/usr/bin/env python3
"""
Prompt Harness for Manifold Discovery

Drives structured probes to map different attention "behaviors" in the model's
latent space. Each probe pack is designed to excite a specific attention program:

- syntax_floor: Local structure (JSON repair, bracket matching)
- semantic_bridge: Mid-range coreference and retrieval-like behavior
- structure_ripple: Long-range periodic patterns (counting, tables)

Usage:
    # Quick smoke test (5 min)
    python prompt_harness.py --duration 5 --server http://localhost:8000

    # Full 8-hour discovery run
    python prompt_harness.py --duration 480 --server http://localhost:8000

    # Generate report from captured data
    python prompt_harness.py --report --db fingerprints.db

The harness:
1. Rotates through probe packs to ensure behavior coverage
2. Tags each request with probe type for later analysis
3. Captures fingerprints via streaming API
4. Stores results in SQLite for discovery job consumption
"""

import argparse
import asyncio
import json
import logging
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# PROBE PACKS - Each designed to excite specific attention programs
# ============================================================================

@dataclass
class Probe:
    """A single probe with expected behavior label."""
    prompt: str
    expected_zone: str  # syntax_floor, semantic_bridge, structure_ripple
    pack: str           # Probe pack name for grouping
    max_tokens: int = 256


def generate_json_repair_probes() -> List[Probe]:
    """JSON/bracket repair - expected to produce syntax_floor patterns."""
    templates = [
        ('Fix this JSON: {"name": "Alice", "age": 30, "city": "NYC"', 50),
        ('Complete this JSON array: [1, 2, 3, {"nested": [4, 5,', 50),
        ('Fix the brackets: def foo(): return {"a": 1, "b": [2, 3}', 100),
        ('Parse and fix: [{"id": 1, "values": [10, 20, 30}, {"id": 2}]', 100),
        ('Complete: CREATE TABLE users (id INT, name VARCHAR(255', 50),
        ('Fix this YAML:\nname: test\nitems:\n  - item1\n  - item2\n  nested:\n    key: value', 100),
    ]
    return [
        Probe(prompt=p, expected_zone='syntax_floor', pack='json_repair', max_tokens=t)
        for p, t in templates
    ]


def generate_coref_probes() -> List[Probe]:
    """Coreference and retrieval - expected to produce semantic_bridge patterns."""
    templates = [
        ("Alice met Bob at the park. She gave him a book. Later, he thanked her for it. "
         "Who received the book?", 50),
        ("The cat sat on the mat. The dog lay near the door. It wagged its tail. "
         "Which animal wagged its tail?", 50),
        ("Dr. Smith examined the patient. She noted the symptoms. He described his pain. "
         "Who described the pain?", 50),
        ("John's car broke down. Mary offered to help. She drove him to the mechanic. "
         "Whose car needed repair?", 50),
        ("The red ball and the blue ball were on the table. Sarah picked up the larger one. "
         "The one she chose was the red ball. What color was the larger ball?", 100),
        ("In the story, the hero found a sword. Later, he used it to defeat the dragon. "
         "The weapon was magical. What did the hero use against the dragon?", 100),
    ]
    return [
        Probe(prompt=p, expected_zone='semantic_bridge', pack='coreference', max_tokens=t)
        for p, t in templates
    ]


def generate_counting_table_probes() -> List[Probe]:
    """Counting and tables - expected to produce structure_ripple patterns."""
    templates = [
        ("Count from 1 to 50, one number per line.", 300),
        ("Create a multiplication table from 1 to 10.", 500),
        ("List the first 20 Fibonacci numbers with their indices.", 300),
        ("Generate a CSV with columns: ID, Name, Score for 15 students.", 400),
        ("Create a weekly schedule table with hours 9AM-5PM and days Mon-Fri.", 400),
        ("Write a table of ASCII codes for letters A-Z (uppercase and lowercase).", 400),
    ]
    return [
        Probe(prompt=p, expected_zone='structure_ripple', pack='counting_tables', max_tokens=t)
        for p, t in templates
    ]


def generate_code_editing_probes() -> List[Probe]:
    """Code editing tasks - mixed patterns depending on task type."""
    templates = [
        ("Rename the variable 'x' to 'count' in this code:\n"
         "def sum_list(items):\n    x = 0\n    for item in items:\n        x += item\n    return x",
         150, 'syntax_floor'),
        ("Add type hints to this function:\n"
         "def calculate(a, b, operation):\n    if operation == 'add':\n        return a + b\n    return a - b",
         150, 'syntax_floor'),
        ("Refactor this code to use list comprehension:\n"
         "result = []\nfor i in range(10):\n    if i % 2 == 0:\n        result.append(i * 2)",
         100, 'semantic_bridge'),
        ("Add error handling to this code:\n"
         "def divide(a, b):\n    return a / b",
         150, 'semantic_bridge'),
    ]
    return [
        Probe(prompt=p, expected_zone=z, pack='code_editing', max_tokens=t)
        for p, t, z in templates
    ]


def generate_reasoning_probes() -> List[Probe]:
    """Multi-step reasoning - exercises deep attention patterns."""
    templates = [
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take "
         "100 machines to make 100 widgets? Think step by step.", 200),
        ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
         "How much does the ball cost? Show your reasoning.", 200),
        ("There are 3 boxes. One contains only apples, one contains only oranges, and one "
         "contains both. The boxes are labeled, but all labels are wrong. If you can only "
         "pick one fruit from one box, which box should you pick from to determine all "
         "contents? Explain your logic.", 300),
        ("A farmer needs to cross a river with a fox, a chicken, and a bag of grain. "
         "The boat can only carry the farmer and one item. If left alone, the fox will eat "
         "the chicken, and the chicken will eat the grain. How should the farmer cross?", 400),
    ]
    return [
        Probe(prompt=p, expected_zone='semantic_bridge', pack='reasoning', max_tokens=t)
        for p, t in templates
    ]


def generate_adversarial_probes() -> List[Probe]:
    """Edge cases and adversarial inputs - for detecting anomalies."""
    templates = [
        ("a" * 500, 50, 'diffuse'),  # Repetitive noise
        ("The the the the quick quick brown brown fox fox jumps jumps", 100, 'diffuse'),
        ("Explain quantum entanglement using only words starting with 'q'.", 200, 'semantic_bridge'),
        ("Write a story where every sentence contradicts the previous one.", 300, 'diffuse'),
        ("Translate to French: Hello. Now to Spanish. Now to German. Now back to English.", 200, 'diffuse'),
    ]
    return [
        Probe(prompt=p, expected_zone=z, pack='adversarial', max_tokens=t)
        for p, t, z in templates
    ]


def generate_natural_probes() -> List[Probe]:
    """Natural conversation - baseline distribution."""
    templates = [
        ("What's the weather like today?", 100),
        ("Tell me a joke about programming.", 150),
        ("Explain how photosynthesis works.", 200),
        ("What are the benefits of exercise?", 200),
        ("Summarize the plot of Romeo and Juliet.", 200),
        ("What's the difference between a virus and a bacterium?", 200),
        ("How do I make pasta from scratch?", 250),
        ("What happened during World War II?", 300),
    ]
    return [
        Probe(prompt=p, expected_zone='semantic_bridge', pack='natural', max_tokens=t)
        for p, t in templates
    ]


def get_all_probe_packs() -> Dict[str, List[Probe]]:
    """Get all probe packs with their probes."""
    return {
        'json_repair': generate_json_repair_probes(),
        'coreference': generate_coref_probes(),
        'counting_tables': generate_counting_table_probes(),
        'code_editing': generate_code_editing_probes(),
        'reasoning': generate_reasoning_probes(),
        'adversarial': generate_adversarial_probes(),
        'natural': generate_natural_probes(),
    }


# ============================================================================
# HARNESS - Runs probes and captures fingerprints
# ============================================================================

@dataclass
class ProbeResult:
    """Result of running a single probe."""
    probe: Probe
    request_id: str
    response: str
    fingerprint: Optional[Dict] = None
    manifold_zone: Optional[str] = None
    duration_ms: float = 0
    tokens_generated: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None


class ManifoldHarness:
    """Runs structured probes and captures fingerprints for manifold discovery."""

    def __init__(
        self,
        server_url: str,
        db_path: str = "fingerprints.db",
        concurrency: int = 1,
    ):
        self.server_url = server_url.rstrip('/')
        self.db_path = Path(db_path)
        self.concurrency = concurrency
        self.probe_packs = get_all_probe_packs()
        self.results: List[ProbeResult] = []
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for storing results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS probe_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                pack TEXT,
                expected_zone TEXT,
                actual_zone TEXT,
                prompt TEXT,
                response TEXT,
                fingerprint TEXT,  -- JSON
                duration_ms REAL,
                tokens_generated INTEGER,
                timestamp TEXT,
                error TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_probe_pack ON probe_results(pack)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_probe_timestamp ON probe_results(timestamp)
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")

    def _save_result(self, result: ProbeResult):
        """Save a probe result to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO probe_results
            (request_id, pack, expected_zone, actual_zone, prompt, response,
             fingerprint, duration_ms, tokens_generated, timestamp, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.request_id,
            result.probe.pack,
            result.probe.expected_zone,
            result.manifold_zone,
            result.probe.prompt[:500],  # Truncate long prompts
            result.response[:2000] if result.response else None,  # Truncate long responses
            json.dumps(result.fingerprint) if result.fingerprint else None,
            result.duration_ms,
            result.tokens_generated,
            result.timestamp,
            result.error,
        ))

        conn.commit()
        conn.close()

    async def _run_probe(
        self,
        session: aiohttp.ClientSession,
        probe: Probe,
        semaphore: asyncio.Semaphore,
    ) -> ProbeResult:
        """Run a single probe and capture fingerprint."""
        request_id = f"probe-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
        start_time = time.time()

        async with semaphore:
            try:
                payload = {
                    "model": "default",
                    "messages": [{"role": "user", "content": probe.prompt}],
                    "max_tokens": probe.max_tokens,
                    "stream": True,
                    "return_attention_tokens": True,
                    "top_k_attention": 10,
                }

                response_text = ""
                fingerprint = None
                manifold_zone = None
                tokens = 0

                async with session.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    async for line in resp.content:
                        line = line.decode('utf-8').strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                # Extract text
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                if content := delta.get("content"):
                                    response_text += content
                                    tokens += 1
                                # Extract fingerprint from attention data
                                attn = data.get("choices", [{}])[0].get("attention_tokens")
                                if attn and isinstance(attn, list) and len(attn) > 0:
                                    latest = attn[-1] if isinstance(attn[-1], dict) else attn[0]
                                    if "fingerprint" in latest:
                                        fingerprint = latest["fingerprint"]
                                    if "manifold_zone" in latest:
                                        manifold_zone = latest["manifold_zone"]
                            except json.JSONDecodeError:
                                continue

                duration_ms = (time.time() - start_time) * 1000

                result = ProbeResult(
                    probe=probe,
                    request_id=request_id,
                    response=response_text,
                    fingerprint=fingerprint,
                    manifold_zone=manifold_zone,
                    duration_ms=duration_ms,
                    tokens_generated=tokens,
                )

            except Exception as e:
                result = ProbeResult(
                    probe=probe,
                    request_id=request_id,
                    response="",
                    error=str(e),
                    duration_ms=(time.time() - start_time) * 1000,
                )
                logger.error(f"Probe failed: {e}")

        self._save_result(result)
        return result

    async def run_discovery(
        self,
        duration_minutes: int,
        probe_mix: Optional[Dict[str, float]] = None,
    ):
        """
        Run discovery for specified duration.

        Args:
            duration_minutes: How long to run
            probe_mix: Optional weights for probe packs (default: equal weight)
        """
        if probe_mix is None:
            # Default: 70% structured probes, 30% natural
            probe_mix = {
                'json_repair': 0.12,
                'coreference': 0.12,
                'counting_tables': 0.12,
                'code_editing': 0.12,
                'reasoning': 0.12,
                'adversarial': 0.10,
                'natural': 0.30,
            }

        # Build weighted probe list
        all_probes = []
        for pack_name, weight in probe_mix.items():
            if pack_name in self.probe_packs:
                pack_probes = self.probe_packs[pack_name]
                # Add each probe with weight adjustment
                all_probes.extend([(p, weight / len(pack_probes)) for p in pack_probes])

        # Normalize weights
        total_weight = sum(w for _, w in all_probes)
        all_probes = [(p, w / total_weight) for p, w in all_probes]

        logger.info(f"Starting discovery run for {duration_minutes} minutes")
        logger.info(f"Probe packs: {list(probe_mix.keys())}")
        logger.info(f"Total unique probes: {len(all_probes)}")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        semaphore = asyncio.Semaphore(self.concurrency)
        probes_run = 0

        async with aiohttp.ClientSession() as session:
            while time.time() < end_time:
                # Select probe based on weights
                r = random.random()
                cumulative = 0
                selected_probe = all_probes[0][0]
                for probe, weight in all_probes:
                    cumulative += weight
                    if r <= cumulative:
                        selected_probe = probe
                        break

                result = await self._run_probe(session, selected_probe, semaphore)
                self.results.append(result)
                probes_run += 1

                # Progress update every 10 probes
                if probes_run % 10 == 0:
                    elapsed = (time.time() - start_time) / 60
                    remaining = duration_minutes - elapsed
                    logger.info(
                        f"Progress: {probes_run} probes | "
                        f"Elapsed: {elapsed:.1f}m | Remaining: {remaining:.1f}m"
                    )

                # Small delay to avoid overwhelming server
                await asyncio.sleep(0.1)

        logger.info(f"Discovery complete. Total probes: {probes_run}")
        return self.results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(db_path: str) -> str:
    """Generate analysis report from captured data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MANIFOLD DISCOVERY REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Summary stats
    cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM probe_results")
    total, start, end = cursor.fetchone()
    report_lines.append(f"Total Probes: {total}")
    report_lines.append(f"Time Range: {start} to {end}")
    report_lines.append("")

    # Probes by pack
    report_lines.append("-" * 40)
    report_lines.append("PROBES BY PACK")
    report_lines.append("-" * 40)
    cursor.execute("""
        SELECT pack, COUNT(*), AVG(duration_ms), AVG(tokens_generated)
        FROM probe_results
        GROUP BY pack
        ORDER BY COUNT(*) DESC
    """)
    for pack, count, avg_duration, avg_tokens in cursor.fetchall():
        report_lines.append(f"  {pack:20s}: {count:5d} probes | {avg_duration:.0f}ms avg | {avg_tokens:.0f} tokens avg")
    report_lines.append("")

    # Zone distribution
    report_lines.append("-" * 40)
    report_lines.append("ZONE DISTRIBUTION")
    report_lines.append("-" * 40)
    cursor.execute("""
        SELECT actual_zone, COUNT(*) as cnt
        FROM probe_results
        WHERE actual_zone IS NOT NULL
        GROUP BY actual_zone
        ORDER BY cnt DESC
    """)
    for zone, count in cursor.fetchall():
        pct = (count / total) * 100
        bar = "#" * int(pct / 2)
        report_lines.append(f"  {zone:20s}: {count:5d} ({pct:5.1f}%) {bar}")
    report_lines.append("")

    # Confusion matrix (expected vs actual zone)
    report_lines.append("-" * 40)
    report_lines.append("ZONE CONFUSION MATRIX (Expected vs Actual)")
    report_lines.append("-" * 40)

    cursor.execute("""
        SELECT expected_zone, actual_zone, COUNT(*) as cnt
        FROM probe_results
        WHERE actual_zone IS NOT NULL
        GROUP BY expected_zone, actual_zone
        ORDER BY expected_zone, actual_zone
    """)
    confusion = {}
    zones = set()
    for exp, act, cnt in cursor.fetchall():
        confusion[(exp, act)] = cnt
        zones.add(exp)
        zones.add(act)

    zones = sorted(zones)
    # Header
    header = "Expected \\ Actual".ljust(20) + "".join(z[:8].center(10) for z in zones)
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for exp_zone in zones:
        row = exp_zone.ljust(20)
        exp_total = sum(confusion.get((exp_zone, z), 0) for z in zones)
        for act_zone in zones:
            cnt = confusion.get((exp_zone, act_zone), 0)
            pct = (cnt / exp_total * 100) if exp_total > 0 else 0
            row += f"{pct:5.0f}%    "
        report_lines.append(row)
    report_lines.append("")

    # Cluster purity by pack
    report_lines.append("-" * 40)
    report_lines.append("CLUSTER PURITY BY PACK")
    report_lines.append("-" * 40)

    cursor.execute("""
        SELECT pack,
               COUNT(*) as total,
               SUM(CASE WHEN expected_zone = actual_zone THEN 1 ELSE 0 END) as correct
        FROM probe_results
        WHERE actual_zone IS NOT NULL
        GROUP BY pack
        ORDER BY pack
    """)
    for pack, total, correct in cursor.fetchall():
        purity = (correct / total * 100) if total > 0 else 0
        bar = "#" * int(purity / 5)
        report_lines.append(f"  {pack:20s}: {purity:5.1f}% purity ({correct}/{total}) {bar}")
    report_lines.append("")

    # Error rate
    cursor.execute("""
        SELECT COUNT(*) FROM probe_results WHERE error IS NOT NULL
    """)
    errors = cursor.fetchone()[0]
    error_rate = (errors / total * 100) if total > 0 else 0
    report_lines.append(f"Error Rate: {error_rate:.1f}% ({errors}/{total} probes)")
    report_lines.append("")

    conn.close()

    return "\n".join(report_lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prompt harness for manifold discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (5 minutes)
  python prompt_harness.py --duration 5 --server http://localhost:8000

  # Full 8-hour discovery run
  python prompt_harness.py --duration 480 --server http://localhost:8000

  # Generate report from captured data
  python prompt_harness.py --report --db fingerprints.db
        """
    )

    parser.add_argument(
        "--server", "-s",
        default="http://localhost:8000",
        help="SGLang server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=5,
        help="Discovery duration in minutes (default: 5)"
    )
    parser.add_argument(
        "--db",
        default="fingerprints.db",
        help="SQLite database path (default: fingerprints.db)"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=1,
        help="Concurrent requests (default: 1)"
    )
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate report from existing data instead of running probes"
    )

    args = parser.parse_args()

    if args.report:
        # Generate report from existing data
        if not Path(args.db).exists():
            print(f"Database not found: {args.db}")
            return 1

        report = generate_report(args.db)
        print(report)

        # Also save to file
        report_path = Path(args.db).with_suffix('.report.txt')
        report_path.write_text(report)
        print(f"\nReport saved to: {report_path}")
        return 0

    # Run discovery
    harness = ManifoldHarness(
        server_url=args.server,
        db_path=args.db,
        concurrency=args.concurrency,
    )

    asyncio.run(harness.run_discovery(args.duration))

    # Generate and print summary report
    report = generate_report(args.db)
    print("\n" + report)

    return 0


if __name__ == "__main__":
    exit(main())
