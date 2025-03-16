#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import itertools
import logging
import os
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from statistics import mean

from ifeval_utils import instructions_registry

def test_instruction_following_strict(inp, response):
    """Tests response to see if instrutions are followed."""
    instruction_list = inp['instruction_id_list']
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp['kwargs'][index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp['prompt'])

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return {
        "strict_prompt_acc": all(is_following_list),
        "strict_instruction_acc": is_following_list
    }
def compute_scores(jobs, cache_path):
    for job in jobs:
        assert len(job["gen"]) == 1
        gen = job['gen'][0]
        job.update(test_instruction_following_strict(job, gen))
    save_cache(jobs, cache_path)
    return mean(x['strict_prompt_acc'] for x in jobs)

def save_cache(jobs, cache_path):
    with open(cache_path, "w") as g:
        for job in jobs:
            g.write(json.dumps(job, ensure_ascii=False) + "\n")
            g.flush()

