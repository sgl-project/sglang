import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pandas
import pytest

from sglang.test.run_eval import (
    validate_per_example_evidence_args,
    write_per_example_evidence,
)
from sglang.test.simple_eval_common import SingleEvalResult, aggregate_results
from sglang.test.simple_eval_gpqa import GPQAEval


class FakeSampler:
    def _pack_message(self, *, content, role):
        return {"content": content, "role": role}

    def __call__(self, _messages):
        return "reasoning\nAnswer: A"


def synthetic_gpqa_rows(count: int) -> list[dict]:
    return [
        {
            "Question": f"question {index}",
            "Correct Answer": f"correct {index}",
            "Incorrect Answer 1": f"wrong {index} one",
            "Incorrect Answer 2": f"wrong {index} two",
            "Incorrect Answer 3": f"wrong {index} three",
        }
        for index in range(count)
    ]


def test_gpqa_retains_non_raw_per_example_evidence(tmp_path: Path):
    dataset = tmp_path / "gpqa.csv"
    pandas.DataFrame(synthetic_gpqa_rows(2)).to_csv(dataset, index=False)

    result = GPQAEval(str(dataset), num_examples=2, num_threads=1)(FakeSampler())

    assert len(result.examples) == 2
    assert {example["source_row_index"] for example in result.examples} == {0, 1}
    for example in result.examples:
        assert example["parsed_answer"] == "A"
        assert "response" not in example
        assert (
            example["response_sha256"]
            == hashlib.sha256(b"reasoning\nAnswer: A").hexdigest()
        )


def test_gpqa_198_question_ids_are_unique_and_ordered_stably(tmp_path: Path):
    dataset = tmp_path / "gpqa.csv"
    pandas.DataFrame(synthetic_gpqa_rows(198)).to_csv(dataset, index=False)
    first_eval = GPQAEval(str(dataset), num_examples=198, num_threads=8)
    second_eval = GPQAEval(str(dataset), num_examples=198, num_threads=8)

    first = first_eval(FakeSampler())
    second = second_eval(FakeSampler())
    first_ids = [example["question_id"] for example in first.examples]
    second_ids = [example["question_id"] for example in second.examples]

    assert len(first_ids) == 198
    assert len(set(first_ids)) == 198
    assert first_ids == second_ids
    assert [example["source_row_index"] for example in first.examples] == [
        row["source_row_index"] for row in first_eval.examples
    ]


def test_source_row_hash_ignores_extra_nan_columns(tmp_path: Path):
    rows = synthetic_gpqa_rows(2)
    first_dataset = tmp_path / "first.csv"
    second_dataset = tmp_path / "second.csv"
    pandas.DataFrame(rows).assign(extra_metadata=["first", "second"]).to_csv(
        first_dataset, index=False
    )
    pandas.DataFrame(rows).assign(extra_metadata=[None, None]).to_csv(
        second_dataset, index=False
    )

    first = GPQAEval(str(first_dataset), num_examples=2, num_threads=1)(FakeSampler())
    second = GPQAEval(str(second_dataset), num_examples=2, num_threads=1)(FakeSampler())

    assert [item["source_row_sha256"] for item in first.examples] == [
        item["source_row_sha256"] for item in second.examples
    ]


def test_per_example_writer_requires_explicit_private_path(tmp_path: Path):
    response = "private reasoning\nAnswer: A"
    response_sha256 = hashlib.sha256(response.encode()).hexdigest()
    example = {
        "question_id": "gpqa-000-deadbeefdeadbeef",
        "response_sha256": response_sha256,
        "parsed_answer": "A",
        "correct": True,
    }
    result = aggregate_results(
        [
            SingleEvalResult(
                score=1.0,
                convo=[
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": response},
                ],
                example=example,
            )
        ]
    )
    public_path = tmp_path / "public.json"
    private_path = tmp_path / "private.jsonl"

    write_per_example_evidence(
        result=result,
        metrics={"score": 1.0},
        eval_name="gpqa",
        model="model",
        output_path=str(public_path),
        private_responses_path=str(private_path),
    )

    public_payload = json.loads(public_path.read_text())
    assert public_payload["examples"] == [example]
    assert public_payload["summary"]["score"] == sum(
        item["correct"] for item in public_payload["examples"]
    ) / len(public_payload["examples"])
    assert response not in public_path.read_text()
    private_payload = json.loads(private_path.read_text())
    assert private_payload["response"] == response
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(public_path.stat().st_mode) == 0o444


def test_writer_refuses_overwrite_and_arm_paths_do_not_conflict(tmp_path: Path):
    response = "Answer: A"
    example = {
        "question_id": "gpqa-000-deadbeefdeadbeef",
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "parsed_answer": "A",
        "correct": True,
    }
    result = aggregate_results(
        [
            SingleEvalResult(
                score=1.0,
                convo=[{"role": "assistant", "content": response}],
                example=example,
            )
        ]
    )
    baseline = tmp_path / "baseline/gpqa_per_example.json"
    candidate = tmp_path / "candidate/gpqa_per_example.json"
    for output in (baseline, candidate):
        write_per_example_evidence(
            result=result,
            metrics={"score": 1.0},
            eval_name="gpqa",
            model="model",
            output_path=str(output),
        )
    assert baseline.is_file() and candidate.is_file()
    with pytest.raises(FileExistsError):
        write_per_example_evidence(
            result=result,
            metrics={"score": 1.0},
            eval_name="gpqa",
            model="model",
            output_path=str(baseline),
        )


def test_private_responses_require_public_evidence_path(tmp_path: Path):
    args = SimpleNamespace(
        eval_name="gpqa",
        repeat=1,
        per_example_output=None,
        per_example_private_responses=str(tmp_path / "private.jsonl"),
    )
    with pytest.raises(
        ValueError,
        match="--per-example-private-responses requires --per-example-output",
    ):
        validate_per_example_evidence_args(args)


@pytest.mark.parametrize(
    ("eval_name", "repeat", "message"),
    [
        ("mmlu", 1, "supported only for GPQA"),
        ("gpqa", 2, "requires --repeat 1"),
    ],
)
def test_per_example_evidence_rejects_unsupported_runs(
    tmp_path: Path, eval_name: str, repeat: int, message: str
):
    args = SimpleNamespace(
        eval_name=eval_name,
        repeat=repeat,
        per_example_output=str(tmp_path / "public.json"),
        per_example_private_responses=None,
    )
    with pytest.raises(ValueError, match=message):
        validate_per_example_evidence_args(args)


@pytest.mark.parametrize("invalid_value", [None, "", "   "])
def test_gpqa_rejects_invalid_semantic_fields(tmp_path: Path, invalid_value):
    rows = synthetic_gpqa_rows(2)
    rows[1]["Incorrect Answer 2"] = invalid_value
    dataset = tmp_path / "gpqa.csv"
    pandas.DataFrame(rows).to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="must contain non-empty strings"):
        GPQAEval(str(dataset), num_examples=2, num_threads=1)
