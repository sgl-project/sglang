#!/usr/bin/env python3
"""
Manifold Discovery Quickstart Example

This script demonstrates the full attention fingerprint discovery workflow:
1. Initialize the database
2. Generate synthetic fingerprints (or collect from real inference)
3. Run discovery to create clusters and manifold embeddings
4. Start the sidecar for real-time classification
5. Query the manifold via API

Requirements:
    pip install numpy pandas pyarrow hdbscan umap-learn scikit-learn joblib requests

Usage:
    # Run the full example
    python examples/manifold_quickstart.py

    # Or run individual steps
    python examples/manifold_quickstart.py --step init
    python examples/manifold_quickstart.py --step generate --count 1000
    python examples/manifold_quickstart.py --step discover
    python examples/manifold_quickstart.py --step classify
"""

import argparse
import json
import random
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

# Configuration
DB_PATH = "./example_fingerprints.db"
OUTPUT_DIR = "./example_discovery"
SIDECAR_PORT = 9009
FINGERPRINT_DIM = 20


def pack_fingerprint(arr: np.ndarray) -> bytes:
    """Pack numpy array to fingerprint blob."""
    return struct.pack(f"<{FINGERPRINT_DIM}f", *arr.astype(np.float32))


def init_database():
    """Initialize the SQLite database with schema."""
    schema_path = Path(__file__).parent.parent / "discovery" / "schema.sql"

    if Path(DB_PATH).exists():
        print(f"Database {DB_PATH} already exists. Remove it to reinitialize.")
        return False

    print(f"Initializing database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.close()

    print("Database initialized successfully!")
    return True


def generate_fingerprint(zone: str) -> np.ndarray:
    """Generate a synthetic fingerprint for a specific zone pattern.

    Fingerprint layout (20 dimensions):
    - [0:3] local_mass, mid_mass, long_mass (attention distribution)
    - [3] entropy (attention entropy)
    - [4:12] histogram bins (attention pattern shape)
    - [12:20] layer entropies (per-layer variation)
    """
    if zone == "syntax_floor":
        # High local mass, low entropy - predictable local patterns
        # Examples: JSON repair, bracket matching, formatting
        local = random.uniform(0.6, 0.9)
        mid = random.uniform(0.1, 0.25)
        long = random.uniform(0.05, 0.15)
        entropy = random.uniform(0.5, 1.5)
        layer_base = 1.2
    elif zone == "semantic_bridge":
        # Balanced mid-range, moderate entropy
        # Examples: coreference resolution, context retrieval
        local = random.uniform(0.2, 0.4)
        mid = random.uniform(0.4, 0.6)
        long = random.uniform(0.1, 0.3)
        entropy = random.uniform(2.0, 3.5)
        layer_base = 2.5
    elif zone == "structure_ripple":
        # Periodic patterns, specific histogram shape
        # Examples: counting, tables, indentation tracking
        local = random.uniform(0.3, 0.5)
        mid = random.uniform(0.3, 0.5)
        long = random.uniform(0.1, 0.2)
        entropy = random.uniform(1.5, 2.5)
        layer_base = 1.8
    elif zone == "long_range":
        # High long-range attention
        # Examples: document-level context, multi-turn dialogue
        local = random.uniform(0.1, 0.25)
        mid = random.uniform(0.2, 0.35)
        long = random.uniform(0.5, 0.8)
        entropy = random.uniform(3.0, 4.5)
        layer_base = 3.0
    else:  # diffuse
        # Random/scattered attention
        # Examples: uncertain predictions, exploration
        local = random.uniform(0.2, 0.4)
        mid = random.uniform(0.2, 0.4)
        long = random.uniform(0.2, 0.4)
        entropy = random.uniform(3.5, 5.0)
        layer_base = 3.5

    # Generate histogram (8 bins) using Dirichlet distribution
    hist = np.random.dirichlet(np.ones(8) * 2)

    # Generate layer entropies (8 layers)
    layer_entropy = [layer_base + random.uniform(-0.5, 0.5) for _ in range(8)]

    return np.array(
        [local, mid, long, entropy] + list(hist) + layer_entropy, dtype=np.float32
    )


def generate_fingerprints(count: int = 500):
    """Generate synthetic fingerprints and store in database."""
    print(f"Generating {count} fingerprints...")

    if not Path(DB_PATH).exists():
        print("Database not found. Run 'init' step first.")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Use zones that match the schema CHECK constraint
    # Note: long_range and diffuse patterns are generated but stored as 'unknown'
    zones = [
        "syntax_floor",
        "semantic_bridge",
        "structure_ripple",
        "long_range",
        "diffuse",
    ]
    zone_to_db = {
        "syntax_floor": "syntax_floor",
        "semantic_bridge": "semantic_bridge",
        "structure_ripple": "structure_ripple",
        "long_range": "unknown",  # Not in schema constraint
        "diffuse": "unknown",  # Not in schema constraint
    }
    per_zone = count // len(zones)

    total_inserted = 0
    for zone in zones:
        for i in range(per_zone):
            request_id = f"example-{zone}-{i:04d}"

            # Generate 3-10 steps per request (simulating multi-token generation)
            n_steps = random.randint(3, 10)
            base_fp = generate_fingerprint(zone)

            for step in range(n_steps):
                # Add small variation per step
                fp = base_fp + np.random.uniform(-0.05, 0.05, FINGERPRINT_DIM).astype(
                    np.float32
                )
                fp = np.clip(fp, 0, 10)  # Keep values reasonable

                cursor.execute(
                    """
                    INSERT INTO fingerprints (request_id, step, fingerprint, manifold_zone)
                    VALUES (?, ?, ?, ?)
                """,
                    (request_id, step, pack_fingerprint(fp), zone_to_db[zone]),
                )
                total_inserted += 1

        print(f"  Generated {per_zone} requests for zone: {zone}")

    conn.commit()
    conn.close()

    print(f"Total fingerprints inserted: {total_inserted}")
    return True


def run_discovery():
    """Run the discovery job to create manifold embeddings."""
    print("Running discovery job...")

    if not Path(DB_PATH).exists():
        print("Database not found. Run 'init' and 'generate' steps first.")
        return False

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    discovery_script = Path(__file__).parent.parent / "discovery" / "discovery_job.py"

    cmd = [
        sys.executable,
        str(discovery_script),
        "--db",
        DB_PATH,
        "--output",
        OUTPUT_DIR,
        "--hours",
        "24",
        "--min-cluster-size",
        "10",
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    # Check outputs
    latest = Path(OUTPUT_DIR) / "latest"
    if latest.exists():
        manifest_path = latest / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"\nDiscovery complete!")
            print(f"  Run ID: {manifest.get('run_id')}")
            print(f"  Clusters: {manifest.get('cluster_count')}")
            print(f"  Fingerprints: {manifest.get('fingerprint_count')}")
            print(f"  Zone distribution: {manifest.get('zone_distribution')}")

    return True


def start_sidecar():
    """Start the sidecar service for real-time classification."""
    print("Starting sidecar service...")

    sidecar_script = Path(__file__).parent.parent / "rapids_sidecar.py"

    cmd = [
        sys.executable,
        str(sidecar_script),
        "--port",
        str(SIDECAR_PORT),
        "--db",
        DB_PATH,
        "--discovery-dir",
        OUTPUT_DIR,
        "--online",
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Sidecar will run on http://localhost:{SIDECAR_PORT}")
    print("Press Ctrl+C to stop.")

    subprocess.run(cmd)


def classify_fingerprints():
    """Demonstrate classification via API."""
    print("Testing classification API...")

    base_url = f"http://localhost:{SIDECAR_PORT}"

    # Check health
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            print(f"Sidecar not healthy: {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to sidecar at {base_url}")
        print(
            "Start the sidecar first: python examples/manifold_quickstart.py --step sidecar"
        )
        return False

    print("Sidecar is healthy!\n")

    # Check discovery status
    resp = requests.get(f"{base_url}/discovery/status")
    status = resp.json()
    print(f"Discovery status:")
    print(f"  Available: {status.get('available')}")
    print(f"  Run ID: {status.get('run_id')}")
    print(f"  Clusters: {status.get('cluster_count')}")
    print()

    # Test classification for each zone
    zones = ["syntax_floor", "semantic_bridge", "long_range", "diffuse"]

    print("Classifying example fingerprints:")
    for zone in zones:
        fp = generate_fingerprint(zone)

        resp = requests.post(
            f"{base_url}/classify",
            json={"vector": fp.tolist()},
        )
        result = resp.json()

        manifold = result.get("manifold", {})
        print(f"\n  Input zone: {zone}")
        print(f"  Classified zone: {manifold.get('zone')}")
        print(f"  Confidence: {manifold.get('confidence', 0):.2f}")
        print(f"  Cluster ID: {manifold.get('cluster_id')}")

    print("\n\nClassification API is working!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Manifold Discovery Quickstart Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all steps
    python manifold_quickstart.py

    # Run individual steps
    python manifold_quickstart.py --step init
    python manifold_quickstart.py --step generate --count 1000
    python manifold_quickstart.py --step discover
    python manifold_quickstart.py --step sidecar
    python manifold_quickstart.py --step classify
""",
    )
    parser.add_argument(
        "--step",
        choices=["all", "init", "generate", "discover", "sidecar", "classify"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of fingerprints to generate (default: 500)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Manifold Discovery Quickstart")
    print("=" * 60)
    print()

    if args.step == "all":
        # Run all steps except sidecar (which blocks)
        steps = ["init", "generate", "discover", "classify"]

        # Start sidecar in background for classify step
        print("Starting sidecar in background...")
        sidecar_script = Path(__file__).parent.parent / "rapids_sidecar.py"
        sidecar_proc = subprocess.Popen(
            [
                sys.executable,
                str(sidecar_script),
                "--port",
                str(SIDECAR_PORT),
                "--db",
                DB_PATH,
                "--discovery-dir",
                OUTPUT_DIR,
                "--online",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            for step in steps:
                print(f"\n{'='*60}")
                print(f"Step: {step.upper()}")
                print("=" * 60)

                if step == "init":
                    init_database()
                elif step == "generate":
                    generate_fingerprints(args.count)
                elif step == "discover":
                    run_discovery()
                elif step == "classify":
                    time.sleep(3)  # Wait for sidecar to start
                    classify_fingerprints()
        finally:
            print("\nStopping sidecar...")
            sidecar_proc.terminate()
            sidecar_proc.wait(timeout=5)
    else:
        if args.step == "init":
            init_database()
        elif args.step == "generate":
            generate_fingerprints(args.count)
        elif args.step == "discover":
            run_discovery()
        elif args.step == "sidecar":
            start_sidecar()
        elif args.step == "classify":
            classify_fingerprints()

    print("\nDone!")


if __name__ == "__main__":
    main()
