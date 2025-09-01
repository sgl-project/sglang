"""
Usage:
# replay from a folder
python3 replay_request_dump.py --file-number 100 --parallel 512 --input-folder /data/lianmin/sglang_request_dump/grok-mini-0220-engine-5756f8f94-28bm6/

# replay from a single file
python3 replay_request_dump.py --parallel 512 --input-file /data/sglang_crash_dump/memx-cti-34-sr1.xpop.twttr.net/crash_dump_2025-06-04_20-13-18.pkl
"""

import argparse
import glob
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime

import requests

from sglang.bench_serving import set_ulimit
from sglang.utils import get_exception_traceback


def read_records(files):
    records = []
    for f in files:
        tmp = pickle.load(open(f, "rb"))
        if isinstance(tmp, dict) and "requests" in tmp:
            records.extend(tmp["requests"])
        else:
            records.extend(tmp)

    return records


def run_one_request_internal(record):
    (req, output, replay_init_time, start_time, end_time, idx) = record
    time.sleep(max(0, (start_time - (time.time() - replay_init_time)) / args.speed))

    if "completion_tokens" in output.get("meta_info", {}):
        recorded_completion_tokens = output["meta_info"]["completion_tokens"]
    else:
        recorded_completion_tokens = ""

    json_data = asdict(req)
    stream = json_data["stream"]

    if args.ignore_eos:
        json_data["sampling_params"]["ignore_eos"] = True
        if recorded_completion_tokens:
            json_data["sampling_params"]["max_new_tokens"] = recorded_completion_tokens

    response = requests.post(
        f"http://{args.host}:{args.port}/generate",
        json=json_data,
        stream=stream,
    )

    if stream:
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                ret = json.loads(chunk[5:].strip("\n"))
    else:
        ret = response.json()

    prompt_tokens = ret["meta_info"]["prompt_tokens"]
    completion_tokens = ret["meta_info"]["completion_tokens"]
    print(
        f"{idx=}, {start_time=:.2f}, {prompt_tokens=}, "
        f"{completion_tokens=}, {recorded_completion_tokens=}"
    )


def run_one_request(record):
    # global success_ct, error_ct

    try:
        run_one_request_internal(record)
        # success_ct += 1
    except Exception:
        # error_ct += 1
        traceback = get_exception_traceback()
        print(f"Hit an exception: {traceback}")


def main(records):
    if len(records) == 0:
        return

    base_time = records[0][-2]
    base_time_str = datetime.fromtimestamp(base_time).strftime("%y-%m-%d %H:%M:%S")
    print(f"{base_time_str=}")
    replay_init_time = time.time()

    for i in range(len(records)):
        req, output, start_time, end_time = records[i]
        start_time -= base_time
        records[i] = (req, output, replay_init_time, start_time, end_time, i)

    with ThreadPoolExecutor(args.parallel) as executor:
        executor.map(run_one_request, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--input-folder", type=str, default=None, help="Folder containing pickle files"
    )
    parser.add_argument(
        "--input-file", type=str, default=None, help="Single pickle file to process"
    )
    parser.add_argument("--file-number", type=int, default=1)
    parser.add_argument("--req-number", type=int, default=1000000)
    parser.add_argument("--req-start", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=512)
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--speed", type=float, default=1)
    args = parser.parse_args()

    set_ulimit()

    files = []
    if args.input_file:
        files = [args.input_file]
        if args.file_number > 1:
            print("Warning: --file-number is ignored when --input-file is provided.")
    elif args.input_folder:
        files = glob.glob(f"{args.input_folder}/*.pkl")
        files = files[: args.file_number]
    else:
        print("Error: Either --input-folder or --input-file must be provided.")
        exit(1)
    print(f"{files=}")

    records = read_records(files)
    # Sort by the receive time, before filtering
    records.sort(key=lambda x: x[-2])
    records = records[args.req_start :]
    if args.idx:
        records = [records[args.idx]]
        print(f"testing {args.idx=}")
        print(f"{records[0]}")
    print(f"{len(records)=}")
    main(records)
