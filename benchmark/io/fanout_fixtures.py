"""Synthetic native output streams for the CPU multi-worker fanout benchmark."""

import math
from array import array

import msgspec
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from sglang.srt.managers import io_struct as io
from sglang.srt.observability.req_time_stats import (
    ReqTimeStatsBase,
    SchedulerReqTimeStats,
)


def make_tokenizer(path):
    """Save a tiny genuine tokenizer locally, without downloading model files."""
    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]")
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(path)
    return tokenizer


def make_stream(count, rich, seed, endpoints):
    """Build three incremental token batches with production timing objects."""
    batches = []
    for step in range(3):
        fields = dict.fromkeys(io.BatchTokenIDOutput.__struct_fields__)
        fields.update(
            rids=[f"rid-{seed}-{i}" for i in range(count)],
            http_worker_ipcs=[endpoints[i % len(endpoints)] for i in range(count)],
            finished_reasons=[
                None if step < 2 else {"type": "length", "length": 3}
                for _ in range(count)
            ],
            decoded_texts=[""] * count,
            decode_ids=[array("q", [step + 1]) for _ in range(count)],
            read_offsets=[0] * count,
            output_ids=[array("q", [step + 1]) for _ in range(count)],
            skip_special_tokens=[True] * count,
            spaces_between_special_tokens=[True] * count,
            no_stop_trim=[False] * count,
            prompt_tokens=[10] * count,
            reasoning_tokens=[0] * count,
            completion_tokens=[step + 1] * count,
            cached_tokens=[0] * count,
        )
        if rich:
            stats = []
            for i in range(count):
                stat = SchedulerReqTimeStats(enable_metrics=True)
                stat.wait_queue_entry_time = 10 + i / 1000
                stat.forward_entry_time = 12 + i / 1000
                stat.prefill_finished_time = 14 + i / 1000
                stat.queue_duration_s = 0.25 + i / 1000
                stats.append(stat)
            fields.update(
                time_stats=io.wrap_as_pickle(stats),
                customized_info=io.wrap_as_pickle(
                    {
                        "metric": [[float(i), 0.5] for i in range(count)],
                        "nested": [[{"x": [i], "label": "λ"}] for i in range(count)],
                    }
                ),
            )
        batches.append(io.BatchTokenIDOutput(**fields))
    return batches


def canonical(value):
    """Compare complete payloads with timing values in a shared wall clock."""
    if isinstance(value, io.PickleWrapper):
        return canonical(io.unwrap_from_pickle(value))
    if isinstance(value, ReqTimeStatsBase):
        state = value.__getstate__()
        offset = state.get("diff_realtime_monotonic", 0)
        return {
            key: item + offset if key.endswith("time") and item else canonical(item)
            for key, item in state.items()
            if key != "diff_realtime_monotonic"
        }
    if isinstance(value, msgspec.Struct):
        return {
            name: canonical(getattr(value, name)) for name in value.__struct_fields__
        }
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, array)):
        return [canonical(item) for item in value]
    return value


def assert_equal(actual, expected, path="output"):
    """Permit clock-rebasing precision only within timing metadata."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (item, wanted) in enumerate(zip(actual, expected)):
            assert_equal(item, wanted, f"{path}[{index}]")
    elif isinstance(expected, float) and ".time_stats" in path:
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=0.000005), path
    else:
        assert actual == expected, (path, actual, expected)
