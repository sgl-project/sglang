# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://github.com/OpenBMB/InfiniteBench/blob/main/src/eval_utils.py

import os
import sys

sys.path.append(os.path.dirname(__file__))

import json
from pathlib import Path

DATA_NAME_TO_PATH = {
    # Retrieval tasks
    "passkey": "passkey.jsonl",
    "kv_retrieval": "kv_retrieval.jsonl",
}

DATA_NAME_TO_MAX_NEW_TOKENS = {
    "passkey": 6,
    "kv_retrieval": 50,
}


def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r") as fin:
        for line in fin:
            if i == cnt:
                break
            yield json.loads(line)
            i += 1


def load_json(fname):
    return json.load(open(fname))


def dump_jsonl(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")


def dump_json(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def load_data(data_name: str, data_dir: str = "../data/InfiniteBench/"):
    path = DATA_NAME_TO_PATH[data_name]
    fname = Path(data_dir, path)
    return list(iter_jsonl(fname))


def create_prompt(eg: dict, data_name: str, data_dir) -> str:
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """
    data_dir = Path(data_dir)

    templates = {
        "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",  # noqa
        "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    }

    if "content" in eg:
        content = eg["content"]
        del eg["content"]
        eg["context"] = content

    format_dict = {
        "context": eg["context"],
        "input": eg["input"],
    }
    prompt = templates[data_name].format(**format_dict)
    return prompt


def get_answer(eg: dict, data_name: str):
    if data_name in ["code_debug", "longbook_choice_eng"]:
        OPTIONS = "ABCD"
        if isinstance(eg["answer"], str):
            ret = [eg["answer"], OPTIONS[eg["options"].index(eg["answer"])]]
        elif isinstance(eg["answer"], list):
            if len(eg["answer"]) == 1:
                ret = [eg["answer"][0], OPTIONS[eg["options"].index(eg["answer"][0])]]
            elif len(eg["answer"]) == 2 and eg["answer"][1] in ["A", "B", "C", "D"]:
                ret = eg["answer"]
            else:
                raise ValueError
        else:
            raise ValueError
        return ret

    return eg["answer"]


def truncate_input(input, max_length, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        return input[0 : max_length // 2] + input[-max_length // 2 :]
    else:
        return None
