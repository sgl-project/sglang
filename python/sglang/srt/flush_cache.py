"""
Flush the KV cache.

Usage:
python3 -m sglang.srt.flush_cache --url http://localhost:30000
"""

import argparse

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:30000")
    args = parser.parse_args()

    response = requests.get(args.url + "/flush_cache")
    assert response.status_code == 200
