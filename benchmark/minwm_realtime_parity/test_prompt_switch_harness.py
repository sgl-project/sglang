from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_minwm_message,
    load_cases,
    prompt_switch_boundary,
    prompt_switch_event,
)


CASES = Path(__file__).with_name("cases_prompt_switch_kv_roll_832x480.json")


def test_prompt_switch_maps_chunk_one_to_shared_frame_boundary() -> None:
    manifest = load_cases(CASES)
    case = manifest["cases"][1]
    contract = manifest["contract"]

    assert prompt_switch_boundary(case, contract) == 17
    assert prompt_switch_event(case) == {
        "type": "event",
        "kind": "prompt",
        "payload": case["prompt_switch"]["prompt"],
        "event_id": 1101,
    }

    message = build_minwm_message(case, contract, Path("/tmp/first-frame.png"))
    controls = message["messages"][1]["controls"]
    prompt_control = next(
        control for control in controls if control["type"] == "text_prompt_interval"
    )
    assert prompt_control["segments"] == [
        {"start": 0, "end": 17, "text": case["prompt"]},
        {
            "start": 17,
            "end": 129,
            "text": case["prompt_switch"]["prompt"],
        },
    ]


def test_prompt_switch_rejects_timing_dependent_later_chunk(
    tmp_path: Path,
) -> None:
    manifest = json.loads(CASES.read_text(encoding="utf-8"))
    invalid = copy.deepcopy(manifest)
    invalid["cases"][1]["prompt_switch"]["target_chunk"] = 2
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match="requires target_chunk=1"):
        load_cases(path)
