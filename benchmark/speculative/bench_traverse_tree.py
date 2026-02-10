"""
Benchmark: Python fallback vs C++ native traverse_tree implementation.

Usage:
    python benchmark/speculative/bench_traverse_tree.py
    python benchmark/speculative/bench_traverse_tree.py --spec-steps 5 --spec-topk 4 --spec-draft-tokens 16
"""

import argparse
import time
from typing import Tuple

import torch
import xgrammar as xgr
from transformers import AutoTokenizer, PreTrainedTokenizer
from xgrammar import allocate_token_bitmask

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.srt.speculative.spec_utils import (
    _traverse_draft_tree_native,
    traverse_tree_fallback,
)


def compile_grammar(tokenizer: PreTrainedTokenizer) -> xgr.CompiledGrammar:
    """Compile JSON grammar once (expensive operation)."""
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar = xgr.Grammar.builtin_json_grammar()
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    return compiled


def create_grammar(compiled: xgr.CompiledGrammar) -> XGrammarGrammar:
    """Create new GrammarMatcher and XGrammarGrammar (cheap operation)."""
    matcher = xgr.GrammarMatcher(compiled)
    return XGrammarGrammar(matcher, compiled.tokenizer_info.vocab_size, None, None)


def create_tree(
    spec_draft_tokens: int,
    spec_topk: int,
    spec_steps: int,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a tree with valid JSON tokens at each node."""
    retrieve_next_token = torch.full((spec_draft_tokens,), -1, dtype=torch.int64)
    retrieve_next_sibling = torch.full((spec_draft_tokens,), -1, dtype=torch.int64)
    draft_tokens = torch.zeros(spec_draft_tokens, dtype=torch.int64)

    # Generate valid JSON token sequences for each path in the tree
    # Example JSON: {"a":1,"b":2,"c":3,...}
    json_str = '{"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8}'
    token_ids = tokenizer.encode(json_str, add_special_tokens=False)

    # Build tree: node 0 -> topk children -> topk grandchildren...
    idx = 1
    level_nodes = [0]
    draft_tokens[0] = token_ids[0]  # First token: "{"

    for step in range(spec_steps):
        if idx >= spec_draft_tokens:
            break
        next_level = []
        for parent in level_nodes:
            if idx >= spec_draft_tokens:
                break
            retrieve_next_token[parent] = idx
            for k in range(spec_topk):
                if idx >= spec_draft_tokens:
                    break
                next_level.append(idx)
                if k < spec_topk - 1 and idx + 1 < spec_draft_tokens:
                    retrieve_next_sibling[idx] = idx + 1
                draft_tokens[idx] = token_ids[step + 1]
                idx += 1
        level_nodes = next_level

    return retrieve_next_token, retrieve_next_sibling, draft_tokens


def benchmark(
    spec_steps: int = 5,
    spec_topk: int = 4,
    spec_draft_tokens: int = 16,
    num_warmup: int = 5,
    num_iter: int = 50,
    tokenizer_path: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> None:
    print(
        f"Config: spec_steps={spec_steps}, spec_topk={spec_topk}, spec_draft_tokens={spec_draft_tokens}"
    )
    print(f"        warmup={num_warmup}, iter={num_iter}")
    print(f"Tokenizer: {tokenizer_path}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)

    # Compile grammar once (expensive)
    compiled = compile_grammar(tokenizer)

    # Prepare data
    retrieve_next_token, retrieve_next_sibling, draft_tokens = create_tree(
        spec_draft_tokens, spec_topk, spec_steps, tokenizer
    )
    first_token = int(draft_tokens[0].item())

    # Benchmark Python fallback
    for _ in range(num_warmup):
        grammar = create_grammar(compiled)
        grammar.accept_token(first_token)  # Accept first draft token before traversal
        bitmask = allocate_token_bitmask(spec_draft_tokens, vocab_size)
        traverse_tree_fallback(
            retrieve_next_token, retrieve_next_sibling, draft_tokens, grammar, bitmask
        )

    python_times = []
    for _ in range(num_iter):
        grammar = create_grammar(compiled)
        grammar.accept_token(first_token)  # Accept first draft token before traversal
        bitmask = allocate_token_bitmask(spec_draft_tokens, vocab_size)
        start = time.perf_counter()
        traverse_tree_fallback(
            retrieve_next_token, retrieve_next_sibling, draft_tokens, grammar, bitmask
        )
        python_times.append((time.perf_counter() - start) * 1000)

    python_avg = sum(python_times) / len(python_times)
    python_bitmask = bitmask.clone()

    # Benchmark C++ native
    if _traverse_draft_tree_native is None:
        print("C++ native not available!")
        return

    for _ in range(num_warmup):
        grammar = create_grammar(compiled)
        grammar.accept_token(first_token)
        bitmask = allocate_token_bitmask(spec_draft_tokens, vocab_size)
        _traverse_draft_tree_native(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar.matcher,
            bitmask,
        )

    native_times = []
    for _ in range(num_iter):
        grammar = create_grammar(compiled)
        grammar.accept_token(first_token)
        bitmask = allocate_token_bitmask(spec_draft_tokens, vocab_size)
        start = time.perf_counter()
        _traverse_draft_tree_native(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar.matcher,
            bitmask,
        )
        native_times.append((time.perf_counter() - start) * 1000)

    native_avg = sum(native_times) / len(native_times)
    native_bitmask = bitmask.clone()

    # Results
    print("=" * 50)
    print(f"Python Fallback: {python_avg:.3f} ms")
    print(f"C++ Native:      {native_avg:.3f} ms")
    print(f"Speedup:         {python_avg / native_avg:.2f}x")
    print("=" * 50)

    # Correctness
    if torch.equal(python_bitmask, native_bitmask):
        print("✓ Results match!")
    else:
        print("✗ Results differ!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec-steps", type=int, default=5, help="Number of speculative steps"
    )
    parser.add_argument(
        "--spec-topk", type=int, default=4, help="Top-k candidates per step"
    )
    parser.add_argument(
        "--spec-draft-tokens", type=int, default=16, help="Total number of draft tokens"
    )
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iter", type=int, default=50)
    parser.add_argument(
        "--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    args = parser.parse_args()

    benchmark(
        args.spec_steps,
        args.spec_topk,
        args.spec_draft_tokens,
        args.num_warmup,
        args.num_iter,
        args.tokenizer,
    )
