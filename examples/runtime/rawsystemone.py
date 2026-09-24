"""Score supplied completions with an existing SGLang HTTP server.

python examples/runtime/rawsystemone.py --url http://127.0.0.1:30000
Set SGLANG_API_KEY if the server requires authentication.
"""

import argparse
import json
import os
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--prefix",
        default="Customer: Please move my appointment to Friday.\n\nThe requested operation is",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        dest="suffixes",
        help="Repeat once per suffix; whitespace is preserved.",
    )
    parser.add_argument("--token-logprobs", action="store_true")
    args = parser.parse_args()
    payload = {
        "prefix": args.prefix,
        "suffixes": args.suffixes or [" booking.", " cancellation.", " rescheduling."],
        "return_token_logprobs": args.token_logprobs,
    }
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SGLANG_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SGLANG_API_KEY"]
    req = Request(
        args.url.rstrip("/") + "/v1/rawsystemone",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urlopen(req, timeout=310) as response:
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
