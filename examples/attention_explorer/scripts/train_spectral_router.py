#!/usr/bin/env python3
"""
Train a Spectral Router from fingerprint database.

Usage:
    python train_spectral_router.py --db exploration_fingerprints.db --output ./router_model

    # Then use in inference:
    python train_spectral_router.py --load ./router_model --fingerprint <vector>
"""

import argparse
import json
import logging
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery import RouterConfig, SpectralManifoldDiscovery, SpectralRouter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_fingerprints_from_db(db_path: str, limit: int = None) -> np.ndarray:
    """Load fingerprints from SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT fingerprint FROM fingerprints ORDER BY id"
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    fingerprints = []
    for (fp_blob,) in rows:
        fp = np.array(struct.unpack("<20f", fp_blob))
        fingerprints.append(fp)

    return np.array(fingerprints)


def train_router(args):
    """Train spectral router from database."""
    logger.info(f"Loading fingerprints from {args.db}")
    fingerprints = load_fingerprints_from_db(args.db, limit=args.limit)
    logger.info(f"Loaded {len(fingerprints)} fingerprints")

    # Configure router
    config = RouterConfig(
        high_coherence_threshold=args.high_threshold,
        low_coherence_threshold=args.low_threshold,
        small_model=args.small_model,
        medium_model=args.medium_model,
        large_model=args.large_model,
    )

    # Train router
    logger.info("Training spectral router...")
    router = SpectralRouter(config=config)
    router.fit(fingerprints)

    # Get spectral analysis stats
    analysis = router.spectral_discovery.analyze(fingerprints)
    logger.info(f"Spectral gap: {analysis.spectral_gap:.4f}")
    logger.info(f"Effective dimension: {analysis.effective_dimension}")
    logger.info(f"Graph connectivity: {analysis.graph_connectivity:.4f}")

    # Save router
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    router_path = output_path / "spectral_router.pkl"
    router.save(str(router_path))

    # Save metadata
    metadata = {
        "n_fingerprints": len(fingerprints),
        "spectral_gap": float(analysis.spectral_gap),
        "effective_dimension": int(analysis.effective_dimension),
        "graph_connectivity": float(analysis.graph_connectivity),
        "high_coherence_threshold": router.config.high_coherence_threshold,
        "low_coherence_threshold": router.config.low_coherence_threshold,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Router saved to {router_path}")

    # Run validation
    if args.validate:
        validate_router(router, fingerprints, args.validate_samples)


def validate_router(
    router: SpectralRouter, fingerprints: np.ndarray, n_samples: int = 100
):
    """Validate router on sample fingerprints."""
    logger.info(f"Validating on {n_samples} samples...")

    indices = np.random.choice(
        len(fingerprints), min(n_samples, len(fingerprints)), replace=False
    )

    decisions = []
    for idx in indices:
        decision = router.route(fingerprints[idx])
        decisions.append(decision)

    stats = router.get_routing_stats()

    logger.info("Routing Statistics:")
    logger.info(f"  Model distribution: {stats['model_distribution']}")
    logger.info(f"  CoT rate: {stats['cot_rate']:.2%}")
    logger.info(
        f"  Avg coherence: {stats['avg_coherence']:.3f} (+/- {stats['coherence_std']:.3f})"
    )
    logger.info(f"  Complexity distribution: {stats['complexity_distribution']}")


def route_fingerprint(args):
    """Route a single fingerprint using a trained router."""
    # Load router
    router = SpectralRouter.load(args.load)

    # Parse fingerprint
    if args.fingerprint:
        fp = np.array([float(x) for x in args.fingerprint.split(",")])
    elif args.fingerprint_file:
        fp = np.load(args.fingerprint_file)
    else:
        raise ValueError("Must provide --fingerprint or --fingerprint-file")

    # Route
    decision = router.route(fp)

    # Output
    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
    else:
        print(f"Model: {decision.model_size.value}")
        print(f"Use CoT: {decision.use_chain_of_thought}")
        print(f"Reason: {decision.reason}")
        print(f"Coherence: {decision.spectral_coherence:.3f}")
        print(f"Confidence: {decision.confidence:.3f}")
        print(f"Complexity: {decision.estimated_complexity}")
        print(f"Recommended model: {router.get_model_for_decision(decision)}")


def analyze_db(args):
    """Analyze fingerprints in database with spectral methods."""
    logger.info(f"Loading fingerprints from {args.db}")
    fingerprints = load_fingerprints_from_db(args.db, limit=args.limit)
    logger.info(f"Loaded {len(fingerprints)} fingerprints")

    # Run spectral analysis
    discovery = SpectralManifoldDiscovery()
    analysis = discovery.analyze(fingerprints)

    print("\n" + "=" * 60)
    print("SPECTRAL ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nFingerprints analyzed: {len(fingerprints)}")
    print(f"Spectral gap: {analysis.spectral_gap:.6f}")
    print(f"Effective dimension: {analysis.effective_dimension}")
    print(f"Graph connectivity: {analysis.graph_connectivity:.6f}")

    # Eigenvalue spectrum
    print(f"\nTop 10 eigenvalues:")
    for i, ev in enumerate(analysis.eigenvalues[:10]):
        print(f"  λ_{i+1} = {ev:.6f}")

    # Coherence distribution
    print(f"\nComputing coherence distribution...")
    n_samples = min(1000, len(fingerprints))
    indices = np.random.choice(len(fingerprints), n_samples, replace=False)

    coherences = []
    for idx in indices:
        coh = discovery.compute_spectral_coherence(fingerprints[idx])
        coherences.append(coh.coherence_score)

    coherences = np.array(coherences)
    print(f"\nCoherence statistics (n={n_samples}):")
    print(f"  Mean: {np.mean(coherences):.3f}")
    print(f"  Std: {np.std(coherences):.3f}")
    print(f"  Min: {np.min(coherences):.3f}")
    print(f"  Max: {np.max(coherences):.3f}")
    print(f"  25th percentile: {np.percentile(coherences, 25):.3f}")
    print(f"  50th percentile: {np.percentile(coherences, 50):.3f}")
    print(f"  75th percentile: {np.percentile(coherences, 75):.3f}")

    # Save analysis
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save eigenvalues
        np.save(output_path / "eigenvalues.npy", analysis.eigenvalues)

        # Save embeddings
        np.save(output_path / "spectral_embeddings.npy", analysis.embeddings)

        # Save coherence samples
        np.save(output_path / "coherence_samples.npy", coherences)

        # Save metadata
        metadata = {
            "n_fingerprints": len(fingerprints),
            "spectral_gap": float(analysis.spectral_gap),
            "effective_dimension": int(analysis.effective_dimension),
            "graph_connectivity": float(analysis.graph_connectivity),
            "coherence_mean": float(np.mean(coherences)),
            "coherence_std": float(np.std(coherences)),
        }
        with open(output_path / "spectral_analysis.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Analysis saved to {args.output}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train and use Spectral Router for query routing"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train spectral router")
    train_parser.add_argument("--db", required=True, help="Fingerprint database path")
    train_parser.add_argument(
        "--output", required=True, help="Output directory for router"
    )
    train_parser.add_argument("--limit", type=int, help="Limit number of fingerprints")
    train_parser.add_argument(
        "--high-threshold", type=float, default=0.7, help="High coherence threshold"
    )
    train_parser.add_argument(
        "--low-threshold", type=float, default=0.3, help="Low coherence threshold"
    )
    train_parser.add_argument(
        "--small-model", default="Qwen/Qwen3-4B", help="Small model identifier"
    )
    train_parser.add_argument(
        "--medium-model", default="Qwen/Qwen3-14B", help="Medium model identifier"
    )
    train_parser.add_argument(
        "--large-model", default="Qwen/Qwen3-72B", help="Large model identifier"
    )
    train_parser.add_argument(
        "--validate", action="store_true", help="Run validation after training"
    )
    train_parser.add_argument(
        "--validate-samples", type=int, default=100, help="Number of validation samples"
    )

    # Route command
    route_parser = subparsers.add_parser("route", help="Route a fingerprint")
    route_parser.add_argument("--load", required=True, help="Path to trained router")
    route_parser.add_argument(
        "--fingerprint", help="Comma-separated fingerprint values"
    )
    route_parser.add_argument(
        "--fingerprint-file", help="Path to .npy fingerprint file"
    )
    route_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze database with spectral methods"
    )
    analyze_parser.add_argument("--db", required=True, help="Fingerprint database path")
    analyze_parser.add_argument("--output", help="Output directory for analysis")
    analyze_parser.add_argument(
        "--limit", type=int, help="Limit number of fingerprints"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_router(args)
    elif args.command == "route":
        route_fingerprint(args)
    elif args.command == "analyze":
        analyze_db(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
