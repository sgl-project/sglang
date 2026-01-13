#!/usr/bin/env python3
"""
Batch script to send multiple generation requests to the SGLang server.
"""

import argparse
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, Optional

import requests


def send_request(
    url: str,
    text: str,
    sampling_params: Dict[str, Any],
    request_id: Optional[int] = None,
) -> None:
    """
    Send a single generation request to the server and consume the streaming response.

    Args:
        url: The endpoint URL
        text: The input text prompt
        sampling_params: Sampling parameters dictionary
        request_id: Optional request ID for tracking
    """
    payload = {
        "text": text,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 4,
        "token_ids_logprob": None,
        "return_text_in_logprobs": False,
        "stream": True,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=None, stream=True
        )
        response.raise_for_status()

        # Consume the streaming response
        for line in response.iter_lines():
            if line:
                # Decode the line if it's bytes
                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                # print(line)
                # input()
                # Stream is consumed but not printed to avoid cluttering output
                # The stream will finish when the server closes the connection
    except requests.exceptions.RequestException as e:
        print(f"Error sending request {request_id}: {e}", file=sys.stderr)
        if e.response is not None:
            try:
                print(f"Response: {e.response.text}", file=sys.stderr)
            except Exception:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Send batch generation requests to SGLang server"
    )
    parser.add_argument(
        "-n", type=int, required=True, help="Number of requests to send"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1000,
        help="Input text length in characters (random text will be generated)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=100000,
        help="Number of new tokens (sets both min_new_tokens and max_new_tokens)",
    )

    args = parser.parse_args()

    url = "http://127.0.0.1:30000/generate"
    # Generate random text of specified length
    text = "".join(
        random.choices(string.ascii_letters + string.digits + " ", k=args.input_len)
    )
    sampling_params = {
        "temperature": 0,
        "min_new_tokens": args.output_len,
        "max_new_tokens": args.output_len,
    }

    print(f"Sending {args.n} concurrent requests...")

    success_count = 0
    error_count = 0
    lock = Lock()

    def send_with_tracking(request_id: int):
        nonlocal success_count, error_count
        try:
            send_request(
                url=url,
                text=text,
                sampling_params=sampling_params,
                request_id=request_id,
            )
            with lock:
                success_count += 1
                print(f"Request {request_id}/{args.n} completed")
            return True
        except Exception as e:
            with lock:
                error_count += 1
                print(f"Request {request_id} failed: {e}", file=sys.stderr)
            return False

    with ThreadPoolExecutor(max_workers=args.n) as executor:
        futures = {
            executor.submit(send_with_tracking, i + 1): i + 1 for i in range(args.n)
        }

        # Wait for all requests to complete
        for future in as_completed(futures):
            future.result()

    print(f"\nSummary: {success_count} succeeded, {error_count} failed")

    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()