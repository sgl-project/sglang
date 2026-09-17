"""Additional checks against an already-running, otherwise idle ReplaySSM server.

OPENAI_API_KEY=EMPTY REPLAYSSM_BASE_URL=http://127.0.0.1:31316 MODEL_PATH=/path/to/model \
GSM8K_DATA_PATH=/path/to/test.jsonl REPLAYSSM_RESULT_DIR=/path/to/results \
  python -m pytest -s test/ppu/e2e/test_kda_pla_replayssm_extended.py

Run separately for the original and PLA paths. This client never launches or
stops the server. Do not run it concurrently with performance measurements.
GSM8K uses the existing 5-shot completion scorer, excluding prefix examples
from the evaluated questions. Scores are measurements, not a model-specific
acceptance threshold; compare the saved question-level results between paths.
"""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

pytestmark = pytest.mark.skipif(
    not os.getenv("REPLAYSSM_BASE_URL"), reason="requires an idle ReplaySSM server"
)


@pytest.fixture(scope="module")
def endpoint():
    url = os.environ["REPLAYSSM_BASE_URL"].rstrip("/")
    response = requests.get(url + "/server_info", timeout=30)
    response.raise_for_status()
    info = response.json()
    assert info["enable_linear_replayssm_spec"]
    assert info["speculative_algorithm"] in ("EAGLE", "DSPARK")
    assert info["speculative_num_draft_tokens"] > 1
    output = Path(os.environ["REPLAYSSM_RESULT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    return url, output


def flush(url):
    response = requests.get(url + "/flush_cache", timeout=30)
    response.raise_for_status()


def test_chat_and_recycled_slots(endpoint):
    from transformers import AutoTokenizer

    from sglang.test.simple_eval_mixed_prefix_gsm8k import get_answer_value

    url, output = endpoint
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["MODEL_PATH"], trust_remote_code=True
    )
    assert tokenizer.chat_template, "use the checkpoint's actual chat template"
    prompts = [
        "Compute 17 multiplied by 19. End with ANSWER: followed by the integer.",
        "A box has 48 pens. Seven are removed and twelve are added. How many pens "
        "remain? End with ANSWER: followed by the integer.",
        "Describe the water cycle and explain why rain forms in three sentences.",
        "Explain the difference between a stack and a queue using an example.",
    ]
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    records = []
    flush(url)
    # Revisit the same questions after freeing/reusing slots and changing batch size.
    for indices in ([0, 1, 2, 3], [0, 1], [0], [1]):
        response = requests.post(
            url + "/generate",
            json={
                "input_ids": [inputs[i] for i in indices],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1024},
            },
            timeout=1200,
        )
        response.raise_for_status()
        results = response.json()
        assert len(results) == len(indices)
        for index, result in zip(indices, results):
            meta = result["meta_info"]
            records.append(
                {"prompt_index": index, "batch_size": len(indices), **result}
            )
            (output / "chat-slots.json").write_text(
                json.dumps(
                    {"prompts": prompts, "input_ids": inputs, "results": records},
                    indent=2,
                )
            )
            assert result["text"].strip()
            assert 0 < meta["completion_tokens"] <= 1024
            assert meta["spec_verify_ct"] > 0
            assert meta["finish_reason"]["type"] in ("stop", "length")
    for record in records:
        expected = {0: 323, 1: 53}.get(record["prompt_index"])
        if expected is not None:
            assert record["meta_info"]["finish_reason"]["type"] == "stop", record
            assert (
                get_answer_value(record["text"].split("</think>")[-1]) == expected
            ), record


@pytest.mark.parametrize("mixed_prefix", [False, True], ids=["gsm8k", "mixed-prefix"])
def test_gsm8k_measurement(endpoint, mixed_prefix):
    from sglang.test.run_eval import run_eval_once
    from sglang.test.simple_eval_mixed_prefix_gsm8k import (
        GSM8KEval,
        MixedPrefixGSM8KEval,
        get_answer_value,
    )

    url, output = endpoint
    dataset = Path(os.environ["GSM8K_DATA_PATH"])
    count = 64 if mixed_prefix else 200
    kwargs = dict(
        num_examples=count, num_threads=8, num_shots=5, data_path=str(dataset)
    )
    if mixed_prefix:
        evaluation = MixedPrefixGSM8KEval(**kwargs, secondary_pool_size=15, seed=37)
    else:
        evaluation = GSM8KEval(**kwargs)
    assert (
        len(evaluation._lines) == count
    ), "dataset must contain all requested questions"
    args = SimpleNamespace(
        api="completion",
        model=os.environ["MODEL_PATH"],
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
    )
    flush(url)
    result, latency, sampler = run_eval_once(args, url + "/v1", evaluation)
    records = []
    for row, conversation in zip(evaluation._lines, result.convos):
        text = conversation[-1]["content"]
        label = get_answer_value(row["answer"])
        prediction = get_answer_value(text)
        records.append(
            dict(
                question=row["question"],
                label=label,
                prediction=prediction,
                correct=label == prediction,
                conversation=conversation,
            )
        )
    name = "mixed-prefix" if mixed_prefix else "gsm8k"
    payload = dict(
        score=result.score,
        metrics=result.metrics,
        latency=latency,
        completion_tokens=sampler._completion_tokens,
        records=records,
        data_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        num_shots=5,
        count=count,
        max_tokens=512,
        concurrency=8,
        temperature=0.0,
        prefix_pool_size=20 if mixed_prefix else 5,
        prompt_style="completion; existing SGLang GSM8K scorer",
    )
    (output / (name + ".json")).write_text(json.dumps(payload, indent=2))
    assert len(records) == count
    assert len(sampler._completion_tokens) == count, "all requests must complete"
    assert all(x["conversation"][-1]["content"].strip() for x in records)
    assert 0 <= result.score <= 1
    assert sum(x["correct"] for x in records) / count == pytest.approx(result.score)
    print(
        json.dumps(
            {
                "evaluation": name,
                "count": count,
                "score": result.score,
                "latency": latency,
            }
        )
    )
