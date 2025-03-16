from pathlib import Path
from collections import defaultdict
from datetime import datetime

import os
import hashlib
import json
import logging
import multiprocessing
from multiprocessing.pool import ThreadPool
import numpy as np
from statistics import mean
from tqdm import tqdm
import copy

from livecodebench_v5_utils.compute_code_generation_metrics import _temp_run

LIVECODEBENCH_TESTS = os.getenv("LIVECODEBENCH_TESTS", "data/livecodebench_v5_tests")

def _extract_code(text: str) -> str:
    outputlines = text.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1:indexlines[-1]])

def preprocess(job):
    tests = job['tests']
    raw_gen = job['gen'] if isinstance(job['gen'], str) else job['gen'][0]
    gen_code = _extract_code(raw_gen)

    return tests, gen_code

def work(job):
    tests, generation = preprocess(job)
    res = check_correctness(
        tests=tests,
        generation=generation,
    )
    assert res['md5'] == tests['md5'], "test md5 mismatched"
    return res, job

def compute_scores(jobs, cache_path):
    with ThreadPool(max(1, int(os.cpu_count() * 0.5))) as pool:
        for res, job in tqdm(pool.imap_unordered(work, jobs), total=len(jobs)):
            extraction_failed = 0
            ispass = res['ispass']
            metadata = res['metadata']
            extraction_failed = metadata.get("error_code", 0) == -1
            results = res['results']

            job.update({
                "pass-1": ispass,
                "results": results,
                "metadata": metadata,
                "extraction_failed": extraction_failed,
            })
            save_cache(job, cache_path)
    with open(cache_path, "r") as f:
        jobs = [json.loads(l) for l in f]

    # Retry all timeout jobs sequentially (without using multiprocessing)
    new_jobs = []
    for job in tqdm(jobs, desc="Processing jobs"):
        error_code = job["metadata"].get("error_code", 0)
        if error_code == -3:
            res, job = work(job)
            job.update(res)
            new_jobs.append(job)
            save_cache(job, cache_path.replace(".jsonl", "_try2.jsonl"))
        else:
            new_jobs.append(job)

    return mean(x['pass-1'] for x in new_jobs)
def check_correctness(tests: dict, generation: str, timeout: int = 30, debug: bool = False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    tests_path = Path(LIVECODEBENCH_TESTS) / tests['fname']
    with open(tests_path, "r") as f:
        sample = json.load(f)

    md5 = calculate_string_md5(json.dumps(sample))

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        metadata_list = [{"error_code": -3}]
        if debug:
            print(f"global timeout")

    res, metadata = result[0], metadata_list[0]
    fixed = []
    for e in res:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        if isinstance(e, np.bool_):
            e = bool(e)
        if e != True and e != False:
            e = False
        fixed.append(e)
    res = fixed
    # print(res)
    if not np.all(res):
        print("fail")
        return dict(ispass=0, md5=md5, results=res, metadata=metadata)
    else:
        print("pass")
        return dict(ispass=1, md5=md5, results=res, metadata=metadata)

def calculate_string_md5(input_string: str):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    return md5.hexdigest()

def save_cache(job, cache_path):
    with open(cache_path, "a") as g:
        g.write(json.dumps(job, ensure_ascii=False) + "\n")
        g.flush()
