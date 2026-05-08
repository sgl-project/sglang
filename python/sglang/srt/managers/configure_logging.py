"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Configure the logging settings of a server.

Usage:
python3 -m sglang.srt.managers.configure_logging --url http://localhost:30000
"""

import argparse

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:30000")
    parser.add_argument("--log-requests", action="store_true")
    parser.add_argument("--log-requests-level", type=int, default=3)
    parser.add_argument(
        "--dump-requests-folder", type=str, default="/tmp/sglang_request_dump"
    )
    parser.add_argument("--dump-requests-threshold", type=int, default=1000)
    args = parser.parse_args()

    response = requests.post(
        args.url + "/configure_logging",
        json={
            "log_requests": args.log_requests,
            "log_requests_level": args.log_requests_level,  # Log full requests
            "dump_requests_folder": args.dump_requests_folder,
            "dump_requests_threshold": args.dump_requests_threshold,
        },
    )
    assert response.status_code == 200
