#!/usr/bin/env python3
"""
Layer Discovery: Analyze which transformer layers encode semantic vs syntactic information.

This script runs the same prompt through all layers and compares attention patterns
to discover layer-specific behaviors.

Usage:
    python layer_discovery.py --prompt "The capital of France is" --num-layers 28
"""

import argparse
import json
import math
import requests
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class LayerAnalysis:
    """Analysis results for a single layer."""
    layer_id: int
    # Which input tokens get the most attention
    top_attended_positions: List[int]
    top_attended_tokens: List[str]
    top_attended_scores: List[float]
    # Attention distribution metrics
    entropy: float  # Higher = more distributed attention
    max_score: float  # Highest single attention score
    prompt_vs_output_ratio: float  # How much attention goes to prompt vs generated tokens


@dataclass
class SemanticTest:
    """A test case for semantic attention discovery."""
    name: str
    prompt: str
    expected_semantic_tokens: List[str]  # Tokens we expect to be semantically important
    description: str


# Pre-defined semantic test cases
SEMANTIC_TESTS = [
    SemanticTest(
        name="capital_city",
        prompt="The capital of France is",
        expected_semantic_tokens=["France", "capital"],
        description="Test if layers attend to 'France' when generating 'Paris'"
    ),
    SemanticTest(
        name="arithmetic",
        prompt="5 + 3 =",
        expected_semantic_tokens=["5", "3", "+"],
        description="Test if layers attend to operands and operator for math"
    ),
    SemanticTest(
        name="synonym",
        prompt="Happy is the opposite of",
        expected_semantic_tokens=["Happy", "opposite"],
        description="Test if layers attend to 'Happy' when generating 'sad'"
    ),
    SemanticTest(
        name="continuation",
        prompt="Once upon a time, there was a",
        expected_semantic_tokens=["Once", "upon", "time"],
        description="Test attention patterns in story continuation"
    ),
    SemanticTest(
        name="translation_context",
        prompt="In French, 'hello' is",
        expected_semantic_tokens=["French", "hello"],
        description="Test if layers attend to language and word to translate"
    ),
]


def get_model_info(api_base: str) -> dict:
    """Get model information including number of layers."""
    try:
        response = requests.get(f"{api_base}/v1/models")
        return response.json()
    except:
        return {}


def tokenize_prompt(prompt: str, api_base: str) -> Tuple[List[int], List[str]]:
    """Tokenize prompt and return token IDs and texts."""
    tok_response = requests.post(
        f"{api_base}/v1/tokenize",
        json={"model": "default", "prompt": prompt}
    )
    token_ids = tok_response.json().get("tokens", [])

    # Get individual token texts
    token_texts = []
    for tid in token_ids:
        detok = requests.post(
            f"{api_base}/v1/detokenize",
            json={"model": "default", "tokens": [tid]}
        )
        token_texts.append(detok.json().get("text", f"[{tid}]"))

    return token_ids, token_texts


def analyze_layer(
    prompt: str,
    layer_id: int,
    prompt_tokens: List[str],
    api_base: str,
    max_tokens: int = 5,
    top_k: int = 15
) -> Optional[LayerAnalysis]:
    """Analyze attention patterns for a specific layer."""

    try:
        response = requests.post(
            f"{api_base}/v1/completions",
            json={
                "model": "default",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "return_attention_tokens": True,
                "top_k_attention": top_k,
                "attention_capture_layer_id": layer_id,
            },
            timeout=30
        )

        if response.status_code != 200:
            print(f"  Layer {layer_id}: API error {response.status_code}")
            return None

        data = response.json()
        attention_tokens = data["choices"][0].get("attention_tokens", [])

        if not attention_tokens:
            return None

        prompt_len = len(prompt_tokens)

        # Aggregate attention across all output tokens
        position_scores = defaultdict(float)
        position_counts = defaultdict(int)
        total_prompt_attention = 0.0
        total_output_attention = 0.0
        all_scores = []

        for entry in attention_tokens:
            positions = entry["token_positions"]
            scores = entry["attention_scores"]

            for pos, score in zip(positions, scores):
                position_scores[pos] += score
                position_counts[pos] += 1
                all_scores.append(score)

                if pos < prompt_len:
                    total_prompt_attention += score
                else:
                    total_output_attention += score

        # Calculate entropy of attention distribution
        if all_scores:
            total = sum(all_scores)
            if total > 0:
                probs = [s / total for s in all_scores]
                entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
            else:
                entropy = 0.0
            max_score = max(all_scores)
        else:
            entropy = 0.0
            max_score = 0.0

        # Get top attended positions (only prompt tokens)
        prompt_positions = [(pos, score) for pos, score in position_scores.items() if pos < prompt_len]
        prompt_positions.sort(key=lambda x: -x[1])

        top_positions = [pos for pos, _ in prompt_positions[:5]]
        top_tokens = [prompt_tokens[pos] if pos < len(prompt_tokens) else f"[{pos}]" for pos in top_positions]
        top_scores = [position_scores[pos] for pos in top_positions]

        # Calculate prompt vs output ratio
        total_attention = total_prompt_attention + total_output_attention
        ratio = total_prompt_attention / total_attention if total_attention > 0 else 0.5

        return LayerAnalysis(
            layer_id=layer_id,
            top_attended_positions=top_positions,
            top_attended_tokens=top_tokens,
            top_attended_scores=top_scores,
            entropy=entropy,
            max_score=max_score,
            prompt_vs_output_ratio=ratio,
        )

    except Exception as e:
        print(f"  Layer {layer_id}: Error - {e}")
        return None


def run_layer_discovery(
    prompt: str,
    num_layers: int,
    api_base: str,
    expected_tokens: Optional[List[str]] = None,
    max_tokens: int = 5,
    layer_step: int = 1
) -> List[LayerAnalysis]:
    """Run attention analysis across all layers."""

    print(f"\n{'='*70}")
    print(f"LAYER DISCOVERY ANALYSIS")
    print(f"{'='*70}")
    print(f"Prompt: \"{prompt}\"")
    if expected_tokens:
        print(f"Expected semantic tokens: {expected_tokens}")
    print()

    # Tokenize prompt
    _, prompt_tokens = tokenize_prompt(prompt, api_base)
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
    print()

    # Analyze each layer
    results = []
    layers_to_test = list(range(0, num_layers, layer_step))

    # Always include the last layer
    if num_layers - 1 not in layers_to_test:
        layers_to_test.append(num_layers - 1)

    print(f"Analyzing {len(layers_to_test)} layers (step={layer_step})...")

    for layer_id in layers_to_test:
        analysis = analyze_layer(prompt, layer_id, prompt_tokens, api_base, max_tokens)
        if analysis:
            results.append(analysis)
            # Progress indicator
            print(f"  Layer {layer_id:3d}: top tokens = {analysis.top_attended_tokens[:3]}, "
                  f"entropy={analysis.entropy:.2f}, prompt_ratio={analysis.prompt_vs_output_ratio:.1%}")

    return results


def find_semantic_layers(
    results: List[LayerAnalysis],
    expected_tokens: List[str]
) -> Tuple[List[int], List[int]]:
    """Find layers that attend to semantic tokens vs structural tokens."""

    semantic_layers = []
    structural_layers = []

    for analysis in results:
        # Check if top attended tokens include expected semantic tokens
        top_tokens_lower = [t.lower().strip() for t in analysis.top_attended_tokens]
        expected_lower = [t.lower() for t in expected_tokens]

        semantic_matches = sum(1 for t in expected_lower if any(t in tt for tt in top_tokens_lower))

        if semantic_matches > 0:
            semantic_layers.append(analysis.layer_id)
        else:
            # Check for structural patterns (common tokens like punctuation, articles)
            structural_tokens = {',', '.', ':', ';', 'the', 'a', 'an', 'is', 'are', 'was', 'were'}
            structural_matches = sum(1 for t in top_tokens_lower if t.strip() in structural_tokens)
            if structural_matches > 0:
                structural_layers.append(analysis.layer_id)

    return semantic_layers, structural_layers


def print_layer_summary(results: List[LayerAnalysis], expected_tokens: Optional[List[str]] = None):
    """Print a summary of layer analysis results."""

    if not results:
        print("No results to analyze")
        return

    print(f"\n{'='*70}")
    print("LAYER SUMMARY")
    print(f"{'='*70}")

    # Find layers with highest/lowest entropy
    by_entropy = sorted(results, key=lambda x: x.entropy)
    print("\nLowest entropy (most focused attention):")
    for a in by_entropy[:3]:
        print(f"  Layer {a.layer_id:3d}: entropy={a.entropy:.3f}, top={a.top_attended_tokens[:2]}")

    print("\nHighest entropy (most distributed attention):")
    for a in by_entropy[-3:]:
        print(f"  Layer {a.layer_id:3d}: entropy={a.entropy:.3f}, top={a.top_attended_tokens[:2]}")

    # Find layers with highest prompt attention ratio
    by_prompt_ratio = sorted(results, key=lambda x: x.prompt_vs_output_ratio, reverse=True)
    print("\nMost prompt-focused layers:")
    for a in by_prompt_ratio[:3]:
        print(f"  Layer {a.layer_id:3d}: {a.prompt_vs_output_ratio:.1%} prompt attention")

    # Semantic analysis if expected tokens provided
    if expected_tokens:
        semantic_layers, structural_layers = find_semantic_layers(results, expected_tokens)

        print(f"\n{'='*70}")
        print("SEMANTIC VS STRUCTURAL LAYERS")
        print(f"{'='*70}")
        print(f"\nLayers attending to semantic tokens ({expected_tokens}):")
        if semantic_layers:
            print(f"  {semantic_layers}")
        else:
            print("  None found")

        print(f"\nLayers attending to structural tokens:")
        if structural_layers:
            print(f"  {structural_layers}")
        else:
            print("  None found")

    # Layer behavior by position
    print(f"\n{'='*70}")
    print("LAYER BEHAVIOR BY POSITION")
    print(f"{'='*70}")

    early_layers = [a for a in results if a.layer_id < len(results) // 3]
    mid_layers = [a for a in results if len(results) // 3 <= a.layer_id < 2 * len(results) // 3]
    late_layers = [a for a in results if a.layer_id >= 2 * len(results) // 3]

    def avg_metrics(layer_list):
        if not layer_list:
            return 0, 0
        return (
            sum(a.entropy for a in layer_list) / len(layer_list),
            sum(a.prompt_vs_output_ratio for a in layer_list) / len(layer_list)
        )

    early_entropy, early_ratio = avg_metrics(early_layers)
    mid_entropy, mid_ratio = avg_metrics(mid_layers)
    late_entropy, late_ratio = avg_metrics(late_layers)

    print(f"\n  Position       Avg Entropy  Avg Prompt Ratio")
    print(f"  -----------    -----------  ----------------")
    print(f"  Early layers   {early_entropy:11.3f}  {early_ratio:15.1%}")
    print(f"  Middle layers  {mid_entropy:11.3f}  {mid_ratio:15.1%}")
    print(f"  Late layers    {late_entropy:11.3f}  {late_ratio:15.1%}")


def run_all_semantic_tests(num_layers: int, api_base: str, layer_step: int = 4):
    """Run all pre-defined semantic tests."""

    print("\n" + "="*70)
    print("RUNNING ALL SEMANTIC TESTS")
    print("="*70)

    all_results = {}

    for test in SEMANTIC_TESTS:
        print(f"\n\n{'#'*70}")
        print(f"# TEST: {test.name}")
        print(f"# {test.description}")
        print(f"{'#'*70}")

        results = run_layer_discovery(
            prompt=test.prompt,
            num_layers=num_layers,
            api_base=api_base,
            expected_tokens=test.expected_semantic_tokens,
            layer_step=layer_step
        )

        print_layer_summary(results, test.expected_semantic_tokens)
        all_results[test.name] = results

        # Small delay between tests
        time.sleep(0.5)

    # Cross-test analysis
    print("\n\n" + "="*70)
    print("CROSS-TEST ANALYSIS")
    print("="*70)

    # Find layers that consistently attend to semantic tokens across tests
    layer_semantic_count = defaultdict(int)
    for test_name, results in all_results.items():
        test = next(t for t in SEMANTIC_TESTS if t.name == test_name)
        semantic_layers, _ = find_semantic_layers(results, test.expected_semantic_tokens)
        for layer_id in semantic_layers:
            layer_semantic_count[layer_id] += 1

    print("\nLayers with semantic attention across multiple tests:")
    for layer_id, count in sorted(layer_semantic_count.items(), key=lambda x: -x[1]):
        if count > 1:
            print(f"  Layer {layer_id}: semantic in {count}/{len(SEMANTIC_TESTS)} tests")


def main():
    parser = argparse.ArgumentParser(description="Layer Discovery Analysis")
    parser.add_argument("--prompt", type=str, help="Custom prompt to analyze")
    parser.add_argument("--num-layers", type=int, default=28, help="Number of model layers")
    parser.add_argument("--api-base", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--max-tokens", type=int, default=5, help="Max tokens to generate")
    parser.add_argument("--layer-step", type=int, default=1, help="Step between layers (for faster testing)")
    parser.add_argument("--run-all-tests", action="store_true", help="Run all semantic tests")
    parser.add_argument("--expected-tokens", nargs="+", help="Expected semantic tokens")

    args = parser.parse_args()

    if args.run_all_tests:
        run_all_semantic_tests(args.num_layers, args.api_base, args.layer_step)
    elif args.prompt:
        results = run_layer_discovery(
            prompt=args.prompt,
            num_layers=args.num_layers,
            api_base=args.api_base,
            expected_tokens=args.expected_tokens,
            max_tokens=args.max_tokens,
            layer_step=args.layer_step
        )
        print_layer_summary(results, args.expected_tokens)
    else:
        # Default: run capital city test
        results = run_layer_discovery(
            prompt="The capital of France is",
            num_layers=args.num_layers,
            api_base=args.api_base,
            expected_tokens=["France", "capital"],
            max_tokens=args.max_tokens,
            layer_step=args.layer_step
        )
        print_layer_summary(results, ["France", "capital"])


if __name__ == "__main__":
    main()
