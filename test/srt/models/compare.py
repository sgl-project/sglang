"""
used for debug using tensor comparison
dump {name: tensor} into "log_hf.jsonl" and "log_srt.jsonl"
use the same name for two tensors that supposed to be close
recommend name like: "layer 2 after mlp"
"""

import json
import sys

import torch

if len(sys.argv) > 1:
    assert sys.argv[1] == "base"
    hf_log = "base_log_hf.jsonl"
    srt_log = "base_log_srt.jsonl"
else:
    hf_log = "log_hf.jsonl"
    srt_log = "log_srt.jsonl"


def load_data(filepath):
    tensors = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            for k, v in data.items():
                tensors[k] = torch.tensor(v)
    return tensors


hf_tensors = load_data(hf_log)
srt_tensors = load_data(srt_log)


def get_diff(t1, t2):
    t1 = t1.reshape(t2.shape)
    max_diff = torch.max(abs(t1.reshape(t2.shape) - t2))
    l2_dis = torch.dist(t1, t2, p=2)
    return l2_dis, max_diff


for k, _ in srt_tensors.items():
    l2_dis, max_diff = get_diff(hf_tensors[k], srt_tensors[k])
    print(f"{k} {l2_dis=} {max_diff=}")
    if k == "layer 1 attn":
        print(hf_tensors[k])
        print(srt_tensors[k])
    if k == "layer 0 prefill k":
        print(srt_tensors[k].shape)
        print(hf_tensors[k].shape)
