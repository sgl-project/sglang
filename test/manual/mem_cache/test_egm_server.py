"""Exercise an EGM host hit on a tiny model server with a 1024-token KV pool.

Start a server with HiCache EGM enabled, layer_first/direct I/O, a 1.2 host
ratio, a 1024-token device pool and tokenizer initialization disabled. The
model must accept input token IDs 3..255 and have at least 256 context tokens.
The test flushes the server's cache and issues generation requests.
"""

import argparse

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:34192")
    args = parser.parse_args()
    requests.post(f"{args.url}/flush_cache", timeout=30).raise_for_status()

    def generate(ids):
        response = requests.post(
            f"{args.url}/generate",
            json={
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    prompt = list(range(3, 67))
    first = generate(prompt)
    for i in range(15):
        generate([((i * 11 + 80 + j * 3) % 253) + 3 for j in range(64)])
    restored = generate(prompt)
    assert first["output_ids"] == restored["output_ids"]
    hit = restored["meta_info"]["cached_tokens_details"]["host"]
    assert hit > 0, restored["meta_info"]
    print(f"host_hit_tokens={hit} output_ids={restored['output_ids']}")


if __name__ == "__main__":
    main()
