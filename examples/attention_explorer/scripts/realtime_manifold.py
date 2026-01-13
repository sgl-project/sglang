#!/usr/bin/env python3
"""
Real-time Manifold Visualization

Watch attention fingerprints stream in and see the manifold grow in real-time.

Usage:
    python scripts/realtime_manifold.py --url http://localhost:30000

Controls:
    - Fingerprints appear as they're generated
    - Colors indicate manifold zone (syntax_floor=blue, semantic_bridge=purple, structure_ripple=red)
    - Close window to stop
"""

import argparse
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from sklearn.decomposition import PCA
except ImportError:
    print("pip install matplotlib scikit-learn")
    sys.exit(1)

# Zone colors
ZONE_COLORS = {
    "syntax_floor": "#3498db",  # Blue
    "semantic_bridge": "#9b59b6",  # Purple
    "structure_ripple": "#e74c3c",  # Red
    "unknown": "#95a5a6",  # Gray
}

# Sample prompts for variety
PROMPTS = [
    "What is 2 + 2?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "def fibonacci(n):",
    "The capital of France is",
    "List the planets in order from the sun.",
    "Why is the sky blue?",
    "Translate 'hello world' to Spanish.",
    "What are the benefits of exercise?",
    "How does photosynthesis work?",
]


class RealtimeManifold:
    def __init__(self, base_url: str, max_points: int = 2000):
        self.base_url = base_url
        self.max_points = max_points

        # Data storage
        self.fingerprints: List[np.ndarray] = []
        self.zones: List[str] = []
        self.coords_2d: Optional[np.ndarray] = None
        self.pca: Optional[PCA] = None

        # Threading
        self.lock = threading.Lock()
        self.running = True
        self.request_thread: Optional[threading.Thread] = None

        # Stats
        self.total_collected = 0
        self.zone_counts = {z: 0 for z in ZONE_COLORS}

    def fetch_fingerprints(self, prompt: str) -> List[Dict]:
        """Send request and get fingerprints."""
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "return_attention_tokens": True,
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                # attention_tokens is directly in choices[0], not meta_info
                return data["choices"][0].get("attention_tokens", [])
        except Exception as e:
            print(f"Request error: {e}")
        return []

    def add_fingerprints(self, tokens: List[Dict]):
        """Add fingerprints to the collection."""
        with self.lock:
            for token in tokens:
                fp = token.get("fingerprint")
                zone = token.get("manifold", "unknown")

                if fp and len(fp) >= 20:
                    self.fingerprints.append(np.array(fp[:20]))
                    self.zones.append(zone)
                    self.zone_counts[zone] = self.zone_counts.get(zone, 0) + 1
                    self.total_collected += 1

            # Trim if too many
            if len(self.fingerprints) > self.max_points:
                excess = len(self.fingerprints) - self.max_points
                self.fingerprints = self.fingerprints[excess:]
                self.zones = self.zones[excess:]

            # Recompute PCA
            if len(self.fingerprints) >= 10:
                fp_array = np.array(self.fingerprints)
                if self.pca is None:
                    self.pca = PCA(n_components=2)
                    self.coords_2d = self.pca.fit_transform(fp_array)
                else:
                    self.coords_2d = self.pca.transform(fp_array)

    def request_loop(self):
        """Background thread that sends requests."""
        prompt_idx = 0
        while self.running:
            prompt = PROMPTS[prompt_idx % len(PROMPTS)]
            tokens = self.fetch_fingerprints(prompt)
            if tokens:
                self.add_fingerprints(tokens)
            prompt_idx += 1
            time.sleep(0.5)  # Small delay between requests

    def start_requests(self):
        """Start the background request thread."""
        self.request_thread = threading.Thread(target=self.request_loop, daemon=True)
        self.request_thread.start()

    def stop(self):
        """Stop the visualization."""
        self.running = False

    def get_plot_data(self):
        """Get current data for plotting."""
        with self.lock:
            if self.coords_2d is None or len(self.coords_2d) == 0:
                return None, None, None
            return (
                self.coords_2d.copy(),
                self.zones.copy(),
                dict(self.zone_counts),
            )


def run_visualization(base_url: str):
    """Run the real-time visualization."""
    print(f"Connecting to {base_url}...")

    # Check server
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code != 200:
            print("Server not healthy")
            return
    except Exception as e:
        print(f"Cannot connect: {e}")
        return

    print("Starting real-time manifold visualization...")
    print("Close the window to stop.\n")

    manifold = RealtimeManifold(base_url)
    manifold.start_requests()

    # Setup plot
    fig, (ax_main, ax_stats) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.suptitle("Real-time Attention Manifold", fontsize=14, fontweight="bold")

    # Initialize scatter
    scatter = ax_main.scatter([], [], c=[], s=20, alpha=0.6)
    ax_main.set_xlabel("PCA Dimension 1")
    ax_main.set_ylabel("PCA Dimension 2")
    ax_main.set_title("Fingerprint Manifold (colored by zone)")
    ax_main.grid(True, alpha=0.3)

    # Stats bars
    zone_names = list(ZONE_COLORS.keys())
    bars = ax_stats.barh(
        zone_names, [0] * len(zone_names), color=[ZONE_COLORS[z] for z in zone_names]
    )
    ax_stats.set_xlabel("Count")
    ax_stats.set_title("Zone Distribution")
    ax_stats.set_xlim(0, 100)

    # Text for total
    total_text = ax_stats.text(
        0.95, 0.95, "", transform=ax_stats.transAxes, ha="right", va="top", fontsize=12
    )

    def update(frame):
        coords, zones, counts = manifold.get_plot_data()

        if coords is not None and len(coords) > 0:
            # Update scatter
            colors = [ZONE_COLORS.get(z, "#95a5a6") for z in zones]
            scatter.set_offsets(coords)
            scatter.set_facecolors(colors)

            # Auto-scale
            margin = 0.5
            ax_main.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
            ax_main.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)

        if counts:
            # Update bars
            max_count = max(counts.values()) if counts.values() else 100
            ax_stats.set_xlim(0, max(100, max_count * 1.1))
            for bar, zone in zip(bars, zone_names):
                bar.set_width(counts.get(zone, 0))

            # Update total
            total = sum(counts.values())
            total_text.set_text(f"Total: {total}")

        return scatter, *bars, total_text

    def on_close(event):
        manifold.stop()

    fig.canvas.mpl_connect("close_event", on_close)

    ani = FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    manifold.stop()
    print(f"\nCollected {manifold.total_collected} fingerprints total.")


def main():
    parser = argparse.ArgumentParser(description="Real-time manifold visualization")
    parser.add_argument("--url", default="http://localhost:30000", help="Server URL")
    args = parser.parse_args()

    run_visualization(args.url)


if __name__ == "__main__":
    main()
