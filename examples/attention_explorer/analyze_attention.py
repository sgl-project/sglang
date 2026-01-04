#!/usr/bin/env python3
"""
Analyze attention patterns to find influential input tokens.

Features:
- Uses local tokenizer (no per-token HTTP calls)
- True probability scoring for influence ranking (not just renormalized top-k)
- Decodes generated tokens (not just prompt tokens)
- Layer selection support for semantic vs syntax analysis
- Proper error handling with timeouts

Usage:
    python analyze_attention.py "Your prompt here" --max-tokens 20
    python analyze_attention.py "def factorial(n):" --layer-id 16
    python analyze_attention.py "Hello world" --layer-id -1  # last layer
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import requests

# Try to use local tokenizer for efficiency
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class AttentionAnalyzer:
    """Analyze attention patterns with efficient tokenization."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000",
        model: str = "default",
        timeout: int = 60,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.local_tokenizer = None
        self._model_path = None

    def _get_model_info(self) -> Optional[str]:
        """Get actual model path from server."""
        try:
            resp = requests.get(
                f"{self.api_base}/v1/models",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id")
        except Exception as e:
            print(f"Warning: Could not get model info: {e}")
        return None

    def _init_local_tokenizer(self):
        """Initialize local tokenizer if available."""
        if not HAS_TRANSFORMERS:
            return

        if self._model_path is None:
            self._model_path = self._get_model_info()

        if self._model_path:
            try:
                self.local_tokenizer = AutoTokenizer.from_pretrained(
                    self._model_path,
                    trust_remote_code=True,
                )
                print(f"Using local tokenizer: {self._model_path}")
            except Exception as e:
                print(f"Warning: Could not load local tokenizer: {e}")

    def tokenize(self, text: str) -> Tuple[List[int], List[str]]:
        """
        Tokenize text, returning (token_ids, token_strings).

        Uses local tokenizer if available, falls back to server API.
        """
        if self.local_tokenizer is None:
            self._init_local_tokenizer()

        if self.local_tokenizer is not None:
            # Fast path: local tokenizer
            token_ids = self.local_tokenizer.encode(text, add_special_tokens=False)
            token_strings = [
                self.local_tokenizer.decode([tid]) for tid in token_ids
            ]
            return token_ids, token_strings

        # Fallback: server API (batch, not per-token)
        try:
            resp = requests.post(
                f"{self.api_base}/v1/tokenize",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            token_ids = resp.json().get("tokens", [])

            # Try batch detokenize if available
            try:
                detok_resp = requests.post(
                    f"{self.api_base}/v1/detokenize",
                    json={"model": self.model, "tokens": token_ids},
                    timeout=self.timeout,
                )
                detok_resp.raise_for_status()
                # Server returns concatenated text, not per-token
                # Fall back to position markers
                token_strings = [f"[{i}]" for i in range(len(token_ids))]
            except:
                token_strings = [f"[{i}]" for i in range(len(token_ids))]

            return token_ids, token_strings

        except Exception as e:
            print(f"Error tokenizing: {e}")
            return [], []

    def get_completion_with_attention(
        self,
        prompt: str,
        max_tokens: int = 20,
        top_k_attention: int = 10,
        temperature: float = 0.0,
        layer_id: Optional[int] = None,
    ) -> Dict:
        """Get completion with attention token capture."""
        try:
            request_body = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "return_attention_tokens": True,
                "top_k_attention": top_k_attention,
            }
            # Add layer selection if specified
            if layer_id is not None:
                request_body["attention_capture_layer_id"] = layer_id

            resp = requests.post(
                f"{self.api_base}/v1/completions",
                json=request_body,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            print(f"Error: Request timed out after {self.timeout}s")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON response: {e}")
            sys.exit(1)

    def compute_true_probability(
        self,
        topk_logit: float,
        logsumexp_candidates: Optional[float],
    ) -> Optional[float]:
        """
        Compute true attention probability from logit and logsumexp.

        Returns None if logsumexp_candidates is not available.
        """
        if logsumexp_candidates is None:
            return None
        try:
            return math.exp(topk_logit - logsumexp_candidates)
        except (OverflowError, ValueError):
            return None

    def analyze(
        self,
        prompt: str,
        max_tokens: int = 20,
        top_k_attention: int = 10,
        max_output_breakdown: int = 5,
        layer_id: Optional[int] = None,
    ) -> Dict:
        """
        Analyze attention patterns for a prompt.

        Returns analysis dict with:
        - prompt_tokens: List of prompt token strings
        - output_tokens: List of generated token strings
        - attention_steps: Number of recorded attention steps
        - influence_ranking: Top tokens by attention received (ranked by true prob if available)
        - output_breakdown: Per-output-token attention analysis
        - captured_mass: Estimated total attention mass captured
        """
        # Get completion
        data = self.get_completion_with_attention(
            prompt, max_tokens, top_k_attention, layer_id=layer_id
        )

        choice = data.get("choices", [{}])[0]
        text = choice.get("text", "")
        attention_tokens = choice.get("attention_tokens", [])

        # Tokenize prompt
        prompt_token_ids, prompt_tokens = self.tokenize(prompt)
        prompt_len = len(prompt_token_ids)

        # Tokenize generated text for proper labeling
        if text:
            _, output_tokens = self.tokenize(text)
        else:
            output_tokens = []

        # Combined token list for position lookup
        all_tokens = prompt_tokens + output_tokens

        # Calculate influence scores
        position_total_score = defaultdict(float)
        position_count = defaultdict(int)
        position_max_score = defaultdict(float)
        position_true_prob_sum = defaultdict(float)

        total_captured_mass = 0.0
        mass_samples = 0

        for entry in attention_tokens:
            # Handle multi-layer format
            if "layers" in entry:
                # Use last layer for backward compat
                layer_id = entry.get("layer_id", max(entry["layers"].keys()))
                layer_data = entry["layers"].get(str(layer_id), entry)
            else:
                layer_data = entry

            positions = layer_data.get("token_positions", [])
            scores = layer_data.get("attention_scores", [])
            logits = layer_data.get("topk_logits", [])
            logsumexp = layer_data.get("logsumexp_candidates")

            # Compute captured mass for this step
            if logits and logsumexp is not None:
                step_mass = 0
                for l in logits:
                    if l is not None:
                        prob = self.compute_true_probability(l, logsumexp)
                        if prob is not None:
                            step_mass += prob
                total_captured_mass += step_mass
                mass_samples += 1

            for i, (pos, score) in enumerate(zip(positions, scores)):
                if pos < prompt_len:
                    position_total_score[pos] += score
                    position_count[pos] += 1
                    position_max_score[pos] = max(position_max_score[pos], score)

                    # Track true probability if available
                    if i < len(logits) and logsumexp is not None and logits[i] is not None:
                        true_prob = self.compute_true_probability(logits[i], logsumexp)
                        if true_prob is not None:
                            position_true_prob_sum[pos] += true_prob

        # Build influence ranking
        # Prefer true_prob_sum for ranking when available, fall back to total_score
        has_true_probs = any(position_true_prob_sum.values())

        influence_ranking = []
        for pos in position_total_score.keys():
            tok_text = all_tokens[pos] if pos < len(all_tokens) else f"[{pos}]"
            true_prob = position_true_prob_sum.get(pos, 0)
            influence_ranking.append({
                "position": pos,
                "token": tok_text,
                "total_score": position_total_score[pos],
                "count": position_count[pos],
                "max_score": position_max_score[pos],
                "true_prob_sum": true_prob,
                "is_prompt": pos < prompt_len,
            })

        # Sort by true_prob_sum if available, otherwise by total_score
        if has_true_probs:
            influence_ranking.sort(key=lambda x: -x["true_prob_sum"])
        else:
            influence_ranking.sort(key=lambda x: -x["total_score"])

        # Build output breakdown
        output_breakdown = []
        for i, entry in enumerate(attention_tokens[:max_output_breakdown]):
            if "layers" in entry:
                layer_id = entry.get("layer_id", max(entry["layers"].keys()))
                layer_data = entry["layers"].get(str(layer_id), entry)
            else:
                layer_data = entry

            positions = layer_data.get("token_positions", [])
            scores = layer_data.get("attention_scores", [])
            logits = layer_data.get("topk_logits", [])
            logsumexp = layer_data.get("logsumexp_candidates")

            prompt_attn = sum(s for p, s in zip(positions, scores) if p < prompt_len)
            output_attn = sum(s for p, s in zip(positions, scores) if p >= prompt_len)

            top_attended = []
            for j, (pos, score) in enumerate(zip(positions[:5], scores[:5])):
                # Use combined token list for proper labeling
                tok_text = all_tokens[pos] if pos < len(all_tokens) else f"[{pos}]"
                true_prob = None
                if j < len(logits) and logsumexp is not None and logits[j] is not None:
                    true_prob = self.compute_true_probability(logits[j], logsumexp)

                top_attended.append({
                    "position": pos,
                    "token": tok_text,
                    "score": score,
                    "true_prob": true_prob,
                    "is_prompt": pos < prompt_len,
                })

            output_breakdown.append({
                "step": i,
                "prompt_attention": prompt_attn,
                "output_attention": output_attn,
                "top_attended": top_attended,
            })

        avg_captured_mass = total_captured_mass / mass_samples if mass_samples > 0 else None

        # Get layer info from first attention entry if available
        captured_layer_id = None
        if attention_tokens:
            captured_layer_id = attention_tokens[0].get("layer_id")

        return {
            "prompt": prompt,
            "output_text": text,
            "prompt_len": prompt_len,
            "output_len": len(output_tokens),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "attention_steps": len(attention_tokens),
            "influence_ranking": influence_ranking,
            "output_breakdown": output_breakdown,
            "avg_captured_mass": avg_captured_mass,
            "layer_id": captured_layer_id,
            "ranked_by": "true_prob" if has_true_probs else "attention_score",
        }

    def print_analysis(self, analysis: Dict):
        """Pretty print analysis results."""
        print(f"Prompt: \"{analysis['prompt']}\"")
        print(f"Output: \"{analysis['output_text'][:100]}{'...' if len(analysis['output_text']) > 100 else ''}\"")
        print(f"Prompt tokens: {analysis['prompt_len']}, Output tokens: {analysis.get('output_len', '?')}")
        print(f"Recorded attention steps: {analysis['attention_steps']} (stride/max limits may apply)")
        if analysis.get('layer_id') is not None:
            print(f"Captured from layer: {analysis['layer_id']}")
        if analysis['avg_captured_mass'] is not None:
            print(f"Avg captured mass (top-k): {analysis['avg_captured_mass']:.1%}")
        print(f"Ranking method: {analysis.get('ranked_by', 'attention_score')}")
        print()

        # Print prompt tokens
        print(f"Prompt tokens ({analysis['prompt_len']}):")
        for i, tok in enumerate(analysis['prompt_tokens'][:20]):
            print(f"  {i:3d}: {repr(tok)}")
        if len(analysis['prompt_tokens']) > 20:
            print(f"  ... ({len(analysis['prompt_tokens']) - 20} more)")

        # Print output tokens if available
        output_tokens = analysis.get('output_tokens', [])
        if output_tokens:
            print(f"\nOutput tokens ({len(output_tokens)}):")
            for i, tok in enumerate(output_tokens[:10]):
                print(f"  {analysis['prompt_len'] + i:3d}: {repr(tok)}")
            if len(output_tokens) > 10:
                print(f"  ... ({len(output_tokens) - 10} more)")
        print()

        # Print influence ranking
        print("=" * 70)
        ranked_by = analysis.get('ranked_by', 'attention_score')
        if ranked_by == 'true_prob':
            print("TOKEN INFLUENCE (ranked by TRUE PROBABILITY - stable across k)")
        else:
            print("TOKEN INFLUENCE (ranked by attention score - renormalized over top-k)")
        print("=" * 70)
        print(f"{'Pos':>4} {'Type':<6} {'Token':<20} {'Score':>8} {'Count':>6} {'TrueProb':>10}")
        print("-" * 70)

        for item in analysis['influence_ranking'][:15]:
            true_prob_str = f"{item['true_prob_sum']:.4f}" if item['true_prob_sum'] > 0 else "-"
            pos_type = "prompt" if item.get('is_prompt', True) else "output"
            print(
                f"{item['position']:4d} {pos_type:<6} {repr(item['token']):<20} "
                f"{item['total_score']:8.3f} {item['count']:6d} "
                f"{true_prob_str:>10}"
            )

        print()
        print("=" * 70)
        print("OUTPUT TOKEN ATTENTION BREAKDOWN")
        print("=" * 70)

        for breakdown in analysis['output_breakdown']:
            print(f"\nOutput token {breakdown['step'] + 1}:")
            print(f"  Prompt attention: {breakdown['prompt_attention']:.1%}")
            print(f"  Output attention: {breakdown['output_attention']:.1%}")
            print(f"  Top attended:")
            for att in breakdown['top_attended']:
                if att['true_prob'] is not None:
                    print(
                        f"    {att['position']:3d} {repr(att['token']):<15} "
                        f"score={att['score']:.3f} true_prob={att['true_prob']:.4f}"
                    )
                else:
                    print(
                        f"    {att['position']:3d} {repr(att['token']):<15} "
                        f"score={att['score']:.3f}"
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention patterns to find influential tokens"
    )
    parser.add_argument("prompt", help="The prompt to analyze")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max output tokens")
    parser.add_argument("--top-k-attention", type=int, default=10,
                        help="Number of top attention positions to capture")
    parser.add_argument("--max-output-breakdown", type=int, default=5,
                        help="Number of output tokens to show detailed breakdown")
    parser.add_argument("--layer-id", type=int, default=None,
                        help="Layer to capture attention from (-1 for last layer, default: server default)")
    parser.add_argument("--model", default="default", help="Model name for API calls")
    parser.add_argument("--api-base", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    analyzer = AttentionAnalyzer(
        api_base=args.api_base,
        model=args.model,
        timeout=args.timeout,
    )

    analysis = analyzer.analyze(
        args.prompt,
        max_tokens=args.max_tokens,
        top_k_attention=args.top_k_attention,
        max_output_breakdown=args.max_output_breakdown,
        layer_id=args.layer_id,
    )

    if args.json:
        # Convert for JSON serialization
        print(json.dumps(analysis, indent=2, default=str))
    else:
        analyzer.print_analysis(analysis)


if __name__ == "__main__":
    main()
