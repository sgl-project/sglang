"""
DIAGNOSTIC ONLY (not for merge): root-cause the low GSM8K scores of
neuralmagic/Mistral-7B-Instruct-v0.3-FP8 and neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8
on AMD. For each config it runs:
  1. the legacy scorer used by this nightly (5-shot prompt in one chat message,
     max_tokens 2048, no stop, last number in the response),
     plus re-scores of the same responses truncated at the next "Question:" block;
  2. the 5-shot completion API with stop=["Question"] (few_shot_gsm8k style);
  3. sgl-eval gsm8k, which is what the NVIDIA nightly baselines on H100
     (27.82% Mistral-7B-FP8, 40.33% Mixtral-8x7B-FP8).
"""

import os
import re
import time
import traceback
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval_once
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.simple_eval_mixed_prefix_gsm8k import GSM8KEval, get_answer_value
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

register_amd_ci(est_time=3600, suite="nightly-amd", nightly=True)

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

LEGACY = ("legacy", "completion_stop_question")
SGL = ("sgl_eval",)
MIXTRAL_FP8 = "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8"
# (model, tp, env overrides, extra server args, evals)
CONFIGS = [
    (
        "neuralmagic/Mistral-7B-Instruct-v0.3-FP8",
        1,
        {"SGLANG_USE_AITER": "1"},
        [],
        LEGACY,
    ),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", 2, {"SGLANG_USE_AITER": "1"}, [], LEGACY),
    (MIXTRAL_FP8, 2, {"SGLANG_USE_AITER": "1", "SGLANG_MOE_PADDING": "0"}, [], LEGACY),
    (MIXTRAL_FP8, 2, {"SGLANG_USE_AITER": "1", "SGLANG_MOE_PADDING": "1"}, [], SGL),
    (
        MIXTRAL_FP8,
        2,
        {"SGLANG_USE_AITER": "0", "SGLANG_MOE_PADDING": "0"},
        ["--disable-cuda-graph"],
        SGL,
    ),
]


def truncate_at_next_question(text):
    m = re.search(r"\n\s*Question:", text)
    return text[: m.start()] if m else text


def legacy_eval(base_url, model):
    args = SimpleNamespace(
        base_url=base_url, model=model, eval_name="gsm8k", num_threads=1024
    )
    eval_obj = GSM8KEval(num_examples=None, num_threads=1024, num_shots=5)
    result, _, _ = run_eval_once(args, base_url, eval_obj)
    n = len(eval_obj._lines)
    official = truncated = continued = 0
    samples = []
    for i in range(n):
        resp = result.convos[i][-1]["content"] or ""
        gold = get_answer_value(eval_obj._lines[i]["answer"])
        ok = get_answer_value(resp) == gold
        cut = truncate_at_next_question(resp)
        ok_cut = get_answer_value(cut) == gold
        official += ok
        truncated += ok_cut
        if cut != resp:
            continued += 1
        if not ok and len(samples) < 3:
            samples.append((gold, resp))
    return dict(
        legacy_official=official / n,
        legacy_truncated=truncated / n,
        legacy_continued_frac=continued / n,
        legacy_mean_len=sum(len(c[-1]["content"] or "") for c in result.convos) / n,
        samples=samples,
    )


def completion_eval(base_url, model):
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="gsm8k",
        num_threads=1024,
        api="completion",
    )
    eval_obj = GSM8KEval(num_examples=None, num_threads=1024, num_shots=5)
    result, _, _ = run_eval_once(args, base_url, eval_obj)
    return result.score


def sgl_eval(base_url, model):
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="gsm8k",
        num_examples=None,
        num_threads=1024,
    )
    return run_sgl_eval(args)["score"]


class TestDiagGsm8kMistral(unittest.TestCase):
    def test_diag(self):
        rows = []
        for model, tp, env, extra_args, evals in CONFIGS:
            label = f"{model} tp={tp} {env} {extra_args}"
            print(f"\n{'=' * 70}\nDIAG {label}\n{'=' * 70}", flush=True)
            os.environ.update(env)
            other_args = ["--trust-remote-code", "--log-level-http", "warning"]
            if tp == 2:
                other_args += ["--tp", "2"]
            other_args += extra_args
            process = None
            row = {"label": label}
            try:
                t = time.time()
                process = popen_launch_server(
                    model, DEFAULT_URL_FOR_TEST, timeout=3600, other_args=other_args
                )
                row["startup_s"] = round(time.time() - t)
                for name, fn in (
                    ("legacy", legacy_eval),
                    ("completion_stop_question", completion_eval),
                    ("sgl_eval", sgl_eval),
                ):
                    if name not in evals:
                        continue
                    try:
                        out = fn(DEFAULT_URL_FOR_TEST, model)
                        if isinstance(out, dict):
                            samples = out.pop("samples")
                            row.update({k: round(v, 4) for k, v in out.items()})
                            for gold, resp in samples:
                                print(f"--- WRONG sample, gold={gold}\n{resp[:800]}\n")
                        else:
                            row[name] = round(out, 4)
                    except Exception:
                        row[name] = "ERROR"
                        traceback.print_exc()
            except Exception:
                row["launch"] = "ERROR"
                traceback.print_exc()
            finally:
                if process is not None:
                    kill_process_tree(process.pid)
            print(f"DIAG RESULT {row}", flush=True)
            rows.append(row)
        print("\nDIAG SUMMARY")
        for row in rows:
            print(f"DIAG SUMMARY {row}")


if __name__ == "__main__":
    unittest.main()
