import glob
import json
import os
import re
import sys

from tqdm import tqdm

sys.path.append("../../")
from fix_corrupted_json import clean_json_file

dirpath = "/Users/ying"
output_file_prefix = "analyzed_log"

time = {}
tot_time = {}
size = {}

os.system(f"rm {output_file_prefix}*")

for dirname in glob.glob(os.path.join(dirpath, "trace*")):
    print(dirname)
    trace_name = dirname.split("/")[-1]
    time[trace_name] = {}
    size[trace_name] = {}
    total_time = 0
    for filename in tqdm(glob.glob(os.path.join(dirname, "*.json"))):
        step_name = filename.split("/")[-1].split(".")[0]
        step_name = "_".join(step_name.split("_")[1:])
        if "prefill" not in filename and "decode" not in filename:
            continue

        match = re.search(r"(prefill|decode)_step_(\d+)\.json", filename)
        if match:
            phase = match.group(1)
            step = match.group(2)
        else:
            raise Exception(f"Cannot parse {filename}")

        try:
            with open(filename, "r") as f:
                trace = json.load(f)
        except:
            clean_json_file(filename, filename)
            with open(filename, "r") as f:
                trace = json.load(f)

        for event in trace["traceEvents"]:
            name = event["name"]
            if name in ["profile_prefill_step", "profile_decode_step"]:
                dur = event["dur"] / 1e3
                time[trace_name][step_name] = dur
                break
        total_time += dur

        step = int(step_name.split("_")[-1])
        with open(os.path.join(dirname, f"size_{step}.json"), "r") as f:
            size_info = json.load(f)
        size[trace_name][step_name] = size_info["size"]

    tot_time[trace_name] = total_time
    time[trace_name] = dict(
        sorted(time[trace_name].items(), key=lambda x: int(x[0].split("_")[-1]))
    )
    size[trace_name] = dict(
        sorted(size[trace_name].items(), key=lambda x: int(x[0].split("_")[-1]))
    )

    with open(f"{output_file_prefix}_{trace_name}", "a") as f:
        for k, v in time[trace_name].items():
            size_v = size[trace_name][k]
            print(f"{k:>15}{v:10.2f}\t{size_v}")
            f.write(f"{k:>15}{v:10.2f}\t{size_v}\n")

with open(f"{output_file_prefix}_total_time", "w") as f:
    print(tot_time)
    json.dump(tot_time, f)
