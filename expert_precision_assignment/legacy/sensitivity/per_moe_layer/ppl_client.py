"""WikiText-2 perplexity via sglang /generate with return_logprob=True."""
import argparse
import json
import math
import random
import time

import requests
from transformers import AutoTokenizer
from datasets import load_dataset


def load_samples(tokenizer_path: str, nsamples: int, seqlen: int, seed: int):
    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt")
    ids = enc.input_ids[0]
    out = []
    for _ in range(nsamples):
        i = random.randint(0, ids.shape[0] - seqlen - 1)
        out.append(ids[i:i + seqlen].tolist())
    return out


def ppl_for_server(
    endpoint: str, tokenizer_path: str,
    nsamples: int, seqlen: int, seed: int,
) -> dict:
    samples = load_samples(tokenizer_path, nsamples, seqlen, seed)
    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()
    for idx, input_ids in enumerate(samples):
        r = requests.post(
            f"{endpoint}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=600,
        )
        r.raise_for_status()
        payload = r.json()
        lps = payload["meta_info"]["input_token_logprobs"]
        nll_sum = 0.0
        n_tok = 0
        for entry in lps:
            logp = entry[0]
            if logp is None:
                continue
            nll_sum += -logp
            n_tok += 1
        total_nll += nll_sum
        total_tokens += n_tok
        if (idx + 1) % 10 == 0:
            partial_ppl = math.exp(total_nll / max(total_tokens, 1))
            print(f"  [{idx+1}/{len(samples)}] partial_ppl={partial_ppl:.4f} "
                  f"({total_tokens} tokens)")

    avg_nll = total_nll / max(total_tokens, 1)
    return {
        "perplexity": math.exp(avg_nll),
        "nll_sum": total_nll,
        "tokens": total_tokens,
        "samples": len(samples),
        "elapsed_s": time.time() - t0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:30210")
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True,
                    help="JSON file to write results.")
    args = ap.parse_args()

    result = ppl_for_server(
        args.endpoint, args.tokenizer_path,
        args.nsamples, args.seqlen, args.seed,
    )
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
