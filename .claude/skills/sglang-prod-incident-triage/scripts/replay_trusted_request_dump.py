#!/usr/bin/env python3
"""Replay a trusted SGLang request dump directly over HTTP.

Use this only for locally captured or otherwise trusted dump files.
It uses plain pickle loading to bypass SafeUnpickler restrictions that may block
the stock replay helper on newer SGLang builds.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import requests

Record = tuple[object, dict[str, Any], float, float]


def normalize_mm_data_item(item: Any) -> Any:
    if isinstance(item, dict) and "url" in item:
        return item["url"]
    return item


def normalize_mm_data(data: Any) -> Any:
    if data is None:
        return None
    if isinstance(data, list):
        return [
            (
                [normalize_mm_data_item(item) for item in sublist]
                if isinstance(sublist, list)
                else normalize_mm_data_item(sublist)
            )
            for sublist in data
        ]
    return normalize_mm_data_item(data)


def normalize_request_data(json_data: dict[str, Any]) -> dict[str, Any]:
    for field in ["image_data", "video_data", "audio_data"]:
        if field in json_data and json_data[field] is not None:
            json_data[field] = normalize_mm_data(json_data[field])
    return json_data


def to_plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    dict_method = getattr(obj, "dict", None)
    if callable(dict_method):
        dumped = dict_method()
        if isinstance(dumped, dict):
            return dumped

    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        return {
            key: value for key, value in obj_dict.items() if not key.startswith("_")
        }

    raise TypeError(f"Unsupported request object type: {type(obj)!r}")


def request_to_json_data(req: Any) -> dict[str, Any]:
    json_data = normalize_request_data(to_plain_dict(req))
    sampling_params = json_data.get("sampling_params")
    if sampling_params is not None and not isinstance(sampling_params, dict):
        json_data["sampling_params"] = to_plain_dict(sampling_params)
    return json_data


def load_records(path: Path) -> list[Record]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict) and "requests" in payload:
        return payload["requests"]
    return payload


def iter_files(args: argparse.Namespace) -> Sequence[Path]:
    if args.input_file:
        return [Path(args.input_file)]
    if args.input_folder:
        return [
            Path(p)
            for p in sorted(glob.glob(f"{args.input_folder}/*.pkl"))[: args.file_number]
        ]
    raise SystemExit("Either --input-file or --input-folder must be provided.")


def run_one_request(
    record: Record,
    args: argparse.Namespace,
    replay_init_time: float,
    base_time: float,
    idx: int,
) -> None:
    req, output, start_time, end_time = record
    relative_start = start_time - base_time
    delay = max(0.0, (relative_start - (time.time() - replay_init_time)) / args.speed)
    if delay:
        time.sleep(delay)

    json_data = request_to_json_data(req)
    if args.ignore_eos:
        json_data.setdefault("sampling_params", {})["ignore_eos"] = True
        completion_tokens = output.get("meta_info", {}).get("completion_tokens")
        if completion_tokens:
            json_data["sampling_params"]["max_new_tokens"] = completion_tokens

    t0 = time.time()
    response = requests.post(
        f"http://{args.host}:{args.port}/generate",
        json=json_data,
        timeout=args.timeout,
        stream=bool(json_data.get("stream")),
    )
    elapsed = time.time() - t0

    if json_data.get("stream"):
        last = None
        for chunk in response.iter_lines(decode_unicode=False):
            decoded = chunk.decode("utf-8")
            if decoded and decoded.startswith("data:"):
                if decoded == "data: [DONE]":
                    break
                last = json.loads(decoded[5:].strip())
        result = last or {}
    else:
        result = response.json()

    meta = result.get("meta_info", {})
    print(
        json.dumps(
            {
                "idx": idx,
                "status_code": response.status_code,
                "elapsed_seconds": round(elapsed, 3),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "rid": meta.get("id"),
            },
            ensure_ascii=False,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a trusted SGLang request dump or crash dump directly over HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--input-folder", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--file-number", type=int, default=1)
    parser.add_argument("--req-number", type=int, default=1_000_000)
    parser.add_argument("--req-start", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    files = iter_files(args)
    print(f"Replay files: {[str(p) for p in files]}")

    records: list[Record] = []
    for path in files:
        records.extend(load_records(path))

    if not records:
        print("No requests found.")
        return 0

    records.sort(key=lambda x: x[-2])
    records = records[args.req_start : args.req_start + args.req_number]
    print(f"Replay requests: {len(records)}")
    base_time = records[0][-2]
    print(
        "Base time: " + datetime.fromtimestamp(base_time).strftime("%Y-%m-%d %H:%M:%S")
    )

    replay_init_time = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for idx, record in enumerate(records):
            futures.append(
                executor.submit(
                    run_one_request, record, args, replay_init_time, base_time, idx
                )
            )
        for future in futures:
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
