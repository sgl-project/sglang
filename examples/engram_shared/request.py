"""Send one request through the SGLang P/D router started by launch.sh."""

import argparse
import json
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", default="http://127.0.0.1:31002")
    parser.add_argument(
        "--prompt", default="What is 17 times 19? Return only the number."
    )
    args = parser.parse_args()
    payload = {
        "text": args.prompt,
        "sampling_params": {
            "temperature": 0,
            "min_new_tokens": 32,
            "max_new_tokens": 32,
            "ignore_eos": True,
        },
    }
    request = Request(
        f"{args.router_url}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as response:
        result = json.load(response)
    output_ids = result.get("output_ids", [])
    if len(output_ids) != 32:
        raise RuntimeError(f"expected 32 output tokens, got {len(output_ids)}")
    print(result.get("text", ""))
    print(json.dumps({"output_tokens": len(output_ids)}))


if __name__ == "__main__":
    main()
