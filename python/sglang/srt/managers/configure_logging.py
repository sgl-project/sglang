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
Configure the logging settings of a server. Only the options passed are sent;
the server keeps its current value for every other setting.

Usage:
python3 -m sglang.srt.managers.configure_logging --url http://localhost:30000 --log-level debug
"""

import argparse

import requests


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:30000")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set runtime log level",
    )
    parser.add_argument(
        "--log-requests", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--log-requests-level", type=int, default=None)
    parser.add_argument("--dump-requests-folder", type=str, default=None)
    parser.add_argument("--dump-requests-threshold", type=int, default=None)
    parser.add_argument(
        "--dump-requests-exclude-meta-keys",
        type=str,
        default=None,
        help=(
            "Comma-separated meta_info keys to strip from each dumped request "
            "(e.g. 'routed_experts,hidden_states'). Pass an empty string to "
            "keep all keys. If not set, the server default is used."
        ),
    )
    return parser.parse_args(argv)


def _build_payload(args: argparse.Namespace) -> dict:
    payload = {
        "log_requests": args.log_requests,
        "log_requests_level": args.log_requests_level,
        "dump_requests_folder": args.dump_requests_folder,
        "dump_requests_threshold": args.dump_requests_threshold,
        "log_level": args.log_level,
    }
    if args.dump_requests_exclude_meta_keys is not None:
        payload["dump_requests_exclude_meta_keys"] = [
            k.strip()
            for k in args.dump_requests_exclude_meta_keys.split(",")
            if k.strip()
        ]
    return {key: value for key, value in payload.items() if value is not None}


if __name__ == "__main__":
    args = _parse_args()
    response = requests.post(args.url + "/configure_logging", json=_build_payload(args))
    assert response.status_code == 200
