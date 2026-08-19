"""Compare a CP-enabled KDA server against a non-CP baseline (manual E2E).

Sends identical greedy requests to both servers and reports:
* generated token-id agreement (exact prefix length),
* prompt-token logprob deltas (prefill-only signal, robust to greedy
  argmax cascades on near-ties: CP vs non-CP differ by re-chunking bf16
  noise, so a long shared prefix + tiny logprob deltas = pass).

Usage: python3 kda_cp_server_compare.py <baseline_port> <cp_port>
"""

import json
import sys
import urllib.request

BASE = int(sys.argv[1]) if len(sys.argv) > 1 else 30500
CP = int(sys.argv[2]) if len(sys.argv) > 2 else 30502

LONG_DOC = (
    "The study of linear attention mechanisms has advanced rapidly. "
    "Delta-rule updates let a fixed-size state track key-value associations, "
    "and per-channel gating decides how quickly old associations decay. "
    "Chunked algorithms split the sequence into blocks, solve a small "
    "triangular system inside each block, and carry a compact recurrent "
    "state between blocks, which makes long-context processing practical. "
) * 60

CASES = [
    ("short", "The capital of France is", 24),
    ("long", LONG_DOC + " In summary, the key idea of these methods is", 48),
]


def post(port, payload):
    req = urllib.request.Request(
        f"http://localhost:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    all_ok = True
    for name, text, max_new in CASES:
        payload = {
            "text": text,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        rb = post(BASE, payload)
        rc = post(CP, payload)

        ids_b, ids_c = rb["output_ids"], rc["output_ids"]
        prefix = common_prefix_len(ids_b, ids_c)

        lp_b = [
            x[0] for x in rb["meta_info"]["input_token_logprobs"] if x[0] is not None
        ]
        lp_c = [
            x[0] for x in rc["meta_info"]["input_token_logprobs"] if x[0] is not None
        ]
        deltas = [abs(a - b) for a, b in zip(lp_b, lp_c)]
        mean_d = sum(deltas) / max(len(deltas), 1)
        max_d = max(deltas) if deltas else 0.0

        prompt_toks = rb["meta_info"]["prompt_tokens"]
        # A fully identical greedy generation is the strongest signal; else
        # require a long shared prefix (greedy near-tie cascades are benign)
        # plus prefill logprob deltas at the re-chunking noise level.
        ok = prefix == max_new or (prefix >= max_new // 2 and mean_d < 0.05)
        all_ok &= ok
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name}: prompt_tokens={prompt_toks} "
            f"gen_match={prefix}/{max_new} "
            f"prompt_logprob mean|d|={mean_d:.4f} max|d|={max_d:.4f}"
        )
        print(f"  base: {rb['text'][:90]!r}")
        print(f"  cp  : {rc['text'][:90]!r}")
    print("E2E_ALL_PASS" if all_ok else "E2E_SOME_FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
