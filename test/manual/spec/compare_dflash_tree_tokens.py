"""Does DFLASH tree verify emit the same tokens as the chain it degenerates to?

At tree width 1 the beam keeps one node per depth, which is the chain DFLASH has
always drafted, so the two paths must produce identical token ids. That equality is
the gate for the whole tree wiring: positions, the custom mask, the tree links and
the accept all have a width-1 answer that is already known to be right.

Run the server twice and diff the dumps:

    ./run_dflash2_tree.sh baseline &                       # chain path
    python3 test/manual/spec/compare_dflash_tree_tokens.py --out /tmp/chain.json
    ./run_dflash2_tree.sh force-tree &                     # tree path, width 1
    python3 test/manual/spec/compare_dflash_tree_tokens.py --out /tmp/tree.json
    python3 test/manual/spec/compare_dflash_tree_tokens.py \
        --compare /tmp/chain.json /tmp/tree.json

Requests go out one at a time so the batch is always size 1: batch composition
changes matmul reduction order, and that difference alone would swamp the signal.

`--concurrency 4` exists but is **not** a gate. Measured 2026-08-24: the same server
compared against itself at concurrency 4 diverges on 2 of 8 prompts, at the same
token indices and the same logprob magnitudes (~1e-2) as a chain-vs-tree comparison
does. Which requests share a step depends on arrival timing, so the batch differs
between runs and nothing about the tree can be concluded from it. Keep it for smoke
value only.

Asserts nothing. It prints, and a divergence needs reading rather than a red mark:
attaching a custom mask switches the attention kernel to its masked path, so a
handful of near-tie argmax flips is a different finding from a wiring bug. The
logprob diff separates them -- structural errors are not subtle.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import requests

# Deterministic, varied enough to reach different accept lengths, short enough that
# a full pass is a couple of minutes on one GPU.
PROMPTS = [
    "Explain in three sentences why speculative decoding speeds up inference.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is 17 * 23? Show the steps.",
    "List four differences between a linked list and an array.",
    "Summarize the plot of Hamlet in one paragraph.",
    "Translate to French: The weather is unusually warm for October.",
    "Name the first five prime numbers and explain what makes a number prime.",
    "Describe what a KV cache stores during autoregressive decoding.",
]


def _generate(port: int, prompt: str, max_new_tokens: int) -> dict:
    response = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            # The only way to get token ids back; also gives the logprobs the
            # divergence triage needs.
            "return_logprob": True,
        },
        timeout=900,
    )
    response.raise_for_status()
    payload = response.json()
    logprobs = payload["meta_info"]["output_token_logprobs"]
    return {
        "prompt": prompt,
        "ids": [entry[1] for entry in logprobs],
        "logprobs": [entry[0] for entry in logprobs],
        "accept_length": payload["meta_info"].get("spec_accept_length"),
        "text": payload["text"],
    }


def collect(*, port: int, max_new_tokens: int, concurrency: int) -> list:
    if concurrency == 1:
        return [_generate(port, prompt, max_new_tokens) for prompt in PROMPTS]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(
            pool.map(lambda prompt: _generate(port, prompt, max_new_tokens), PROMPTS)
        )


def compare(chain: list, tree: list) -> bool:
    """Print a per-prompt verdict; return whether every token matched."""
    all_equal = True
    for baseline, candidate in zip(chain, tree):
        ids_a, ids_b = baseline["ids"], candidate["ids"]
        shared = min(len(ids_a), len(ids_b))
        first_diff = next(
            (i for i in range(shared) if ids_a[i] != ids_b[i]),
            None if len(ids_a) == len(ids_b) else shared,
        )
        # Over the shared prefix only: past a divergence the sequences are different
        # continuations and the logprob gap stops meaning anything.
        compared = shared if first_diff is None else first_diff
        max_logprob_diff = max(
            (
                abs(baseline["logprobs"][i] - candidate["logprobs"][i])
                for i in range(compared)
            ),
            default=0.0,
        )
        verdict = "same" if first_diff is None else f"DIVERGES at token {first_diff}"
        print(
            f"[{verdict}] len={len(ids_a)}/{len(ids_b)} "
            f"accept_len={baseline['accept_length']}/{candidate['accept_length']} "
            f"max|dlogprob|={max_logprob_diff:.2e}  {baseline['prompt'][:44]!r}"
        )
        if first_diff is not None:
            all_equal = False
            lo, hi = max(0, first_diff - 3), first_diff + 3
            print(f"    chain ids[{lo}:{hi}] = {ids_a[lo:hi]}")
            print(f"    tree  ids[{lo}:{hi}] = {ids_b[lo:hi]}")
    print("\nALL TOKENS IDENTICAL" if all_equal else "\nDIVERGENCE FOUND")
    if not all_equal:
        print(
            "Triage: max|dlogprob| above ~1e-3 with a structural pattern means the "
            "wiring is wrong (check positions -> custom_mask -> accept_index[:, 0] == "
            "bs_idx * N -> retrieve_* links, in that order). Values below that, at a "
            "near-tie argmax, are the masked attention kernel's own numerics."
        )
    return all_equal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="1 keeps the batch at size 1; >1 cross-checks with a fixed batch.",
    )
    parser.add_argument("--out", help="Write this run's dump here.")
    parser.add_argument(
        "--compare", nargs=2, metavar=("CHAIN", "TREE"), help="Diff two dumps."
    )
    args = parser.parse_args()

    if args.compare:
        with open(args.compare[0]) as chain_file, open(args.compare[1]) as tree_file:
            compare(json.load(chain_file), json.load(tree_file))
        return

    dump = collect(
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        concurrency=args.concurrency,
    )
    for record in dump:
        print(
            f"{len(record['ids']):4d} tokens  accept_len={record['accept_length']}  "
            f"{record['prompt'][:56]!r}"
        )
    if args.out:
        with open(args.out, "w") as out_file:
            json.dump(dump, out_file)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
