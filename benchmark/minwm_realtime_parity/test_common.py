from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    action_label_sequence,
    build_minwm_message,
    load_cases,
    materialize_first_frame,
)


DRAGON_CASES = Path(__file__).with_name("cases_dragon_ride_60s_832x480.json")


def test_dragon_ride_contract_is_exactly_sixty_generated_seconds() -> None:
    manifest = load_cases(DRAGON_CASES)
    contract = manifest["contract"]
    case = manifest["cases"][0]

    assert contract["generated_pixel_frames"] == 1440
    assert contract["fps"] == 24
    assert contract["generated_pixel_frames"] / contract["fps"] == 60
    assert contract["chunks"] == 90
    labels = action_label_sequence(case, contract)
    assert labels == (
        [9] * 30
        + [18] * 30
        + [0] * 30
        + [9] * 30
        + [18] * 30
        + [0] * 210
    )

    message = build_minwm_message(case, contract, Path("/tmp/dragon.png"))
    actions = message["messages"][1]["controls"][0]["actions"]
    assert actions[0] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert actions[119] == actions[0]
    assert actions[120] == [0, 0, 1, 0, 0, 0, 0, 0]
    assert actions[239] == actions[120]
    assert actions[240] == [0] * 8
    assert actions[359] == actions[240]
    assert actions[360] == actions[0]
    assert actions[480] == actions[120]
    assert actions[600] == [0] * 8
    assert actions[-1] == [0] * 8


def test_materialize_http_first_frame_verifies_sha256(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"stable-reference-image"
    expected = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    path = materialize_first_frame(
        {
            "id": "http-fixture",
            "first_frame": "https://example.invalid/reference.png",
            "first_frame_sha256": expected,
        },
        tmp_path,
    )

    assert path.read_bytes() == payload


def test_materialize_first_frame_rejects_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"changed"),
    )

    with pytest.raises(ValueError, match="does not match"):
        materialize_first_frame(
            {
                "id": "http-fixture",
                "first_frame": "https://example.invalid/reference.png",
                "first_frame_sha256": "0" * 64,
            },
            tmp_path,
        )
