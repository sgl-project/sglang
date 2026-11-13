"""The launch-record comparison of the test-only MoE LoRA launch recorder
fails closed: it reports two records as the same only when it compared them.

Records here are written by hand in the exported format, so the comparison is
tested without a GPU: no records, unreadable or malformed files, records
without evidence and records taken under another fixture, device, library
set, format or check kind are refused with a non-zero exit and are never
printed as SAME; a different revision alone is not a reason to refuse.
"""

from __future__ import annotations

import json

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.moe_lora_launch_recorder import (
    LaunchRecorder,
    NotComparable,
    first_difference,
    main,
    not_comparable,
    save_artifact,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _call(kernel: str = "kernel_a", **fields) -> dict:
    call = dict(
        kind="triton",
        kernel=kernel,
        grid=[4, 1],
        constants=[[3, 128], ["BLOCK", 64]],
        operands=[[0, [8, 16], "torch.bfloat16", [16, 1], "hidden", 0, None]],
        boundary={},
    )
    return call | fields


def _record(case: str = "case", **fields) -> dict:
    record = dict(
        format=1,
        case=case,
        check="structure",
        fixture=dict(seed=20261109, tokens=257),
        revision="1111111111aaaaaaaaaa",
        device=dict(name="NVIDIA GB300", capability=[10, 3], sms=152),
        environment=dict(torch="2.14.1+cu130", triton="3.8.0"),
        coverage={},
        calls=[_call(), _call("kernel_b")],
        outputs={},
    )
    return record | fields


def _numeric(**fields) -> dict:
    calls = [
        _call(boundary=dict(hidden="aa")),
        _call("kernel_b", boundary=dict(hidden="aa", gateup="bb")),
    ]
    return _record(check="numeric", calls=calls, outputs=dict(output="cc")) | fields


def _write(path, *records) -> str:
    for record in records:
        save_artifact(path, record)
    return str(path)


def _lines(capsys, prefix: str) -> list[str]:
    return [
        line for line in capsys.readouterr().out.splitlines() if line.startswith(prefix)
    ]


def test_paths_without_records_are_refused(tmp_path, capsys) -> None:
    empty_a, empty_b = tmp_path / "a", tmp_path / "b"
    empty_a.mkdir()
    empty_b.mkdir()
    assert main([str(empty_a), str(empty_b)]) == 3
    out = capsys.readouterr().out
    assert out.startswith("INVALID") and "no launch records" in out
    assert "SAME" not in out
    good = _write(tmp_path / "good.json", _record())
    assert main([good, str(empty_b)]) == 3
    assert main([str(empty_a), good]) == 3
    assert main([good, str(tmp_path / "absent.json")]) == 3
    no_cases = tmp_path / "no_cases.json"
    no_cases.write_text("{}\n")
    assert main([good, str(no_cases)]) == 3
    not_json = tmp_path / "not_json.json"
    not_json.write_text("SAME case\n")
    assert main([good, str(not_json)]) == 3
    a_list = tmp_path / "a_list.json"
    a_list.write_text(json.dumps([_record()]))
    assert main([good, str(a_list)]) == 3
    out = capsys.readouterr().out
    assert "SAME" not in out and out.count("INVALID") == 6
    assert main([]) == 2 and main([good]) == 2


def test_only_the_revision_may_differ(tmp_path, capsys) -> None:
    before = _write(tmp_path / "before.json", _record(), _numeric(case="numeric"))
    later = "2222222222bbbbbbbbbb"
    after = _write(
        tmp_path / "after.json",
        _record(revision=later),
        _numeric(case="numeric", revision=later),
    )
    assert main([before, after]) == 0
    same = _lines(capsys, "SAME")
    assert len(same) == 2 and all("1111111111 -> 2222222222" in line for line in same)
    assert first_difference(_record(), _record(revision=later)) is None
    # A directory of artifact files is read as one set of cases.
    folder = tmp_path / "records"
    folder.mkdir()
    _write(folder / "one.launches.json", _record())
    _write(folder / "two.launches.json", _numeric(case="numeric"))
    assert main([str(folder), after]) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("fixture", dict(seed=20261109, tokens=129)),
        ("device", dict(name="NVIDIA H200", capability=[9, 0], sms=132)),
        ("environment", dict(torch="2.14.1+cu130", triton="3.9.0")),
        ("format", 2),
        ("check", "numeric"),
    ],
)
def test_another_context_is_not_comparable(tmp_path, capsys, field, value) -> None:
    other = _numeric() if field == "check" else _record(**{field: value})
    assert any(reason.startswith(field) for reason in not_comparable(_record(), other))
    with pytest.raises(NotComparable, match=field):
        first_difference(_record(), other)
    before = _write(tmp_path / "before.json", _record())
    after = _write(tmp_path / "after.json", other)
    assert main([before, after]) == 3
    out = capsys.readouterr().out
    assert f"NOT-COMPARABLE case: {field}" in out and "SAME" not in out


@pytest.mark.parametrize(
    "broken,reason",
    [
        (_record(calls=[]), "no launches recorded"),
        (
            _numeric(calls=[_call(), _call("kernel_b")], outputs={}),
            "without any digest",
        ),
        ({k: v for k, v in _record().items() if k != "device"}, "not a launch record"),
        (dict(case="case"), "not a launch record"),
    ],
)
def test_a_record_without_evidence_is_not_comparable(
    tmp_path, capsys, broken, reason
) -> None:
    whole = _numeric() if broken.get("check") == "numeric" else _record()
    for pair in ((whole, broken), (broken, whole), (broken, broken)):
        assert any(reason in text for text in not_comparable(*pair))
        with pytest.raises(NotComparable, match=reason):
            first_difference(*pair)
    before = _write(tmp_path / "before.json", broken)
    after = _write(tmp_path / "after.json", broken)
    assert main([before, after]) == 3
    out = capsys.readouterr().out
    assert "NOT-COMPARABLE case" in out and "SAME" not in out


def test_a_difference_or_a_missing_case_exits_1(tmp_path, capsys) -> None:
    changed = _record(calls=[_call(), _call("kernel_c")])
    found = first_difference(_record(), changed)
    assert (found["call"], found["field"]) == (1, "kernel")
    before = _write(tmp_path / "before.json", _record(), _record("only_before"))
    after = _write(tmp_path / "after.json", changed)
    assert main([before, after]) == 1
    out = capsys.readouterr().out
    assert "DIFFERS case" in out and '"call": 1' in out
    assert "MISSING only_before: only in before" in out and "SAME" not in out
    # A numeric record parts at the first buffer, then at an output.
    moved = _numeric()
    moved["calls"][1]["boundary"]["gateup"] = "dd"
    found = first_difference(_numeric(), moved)
    assert (found["call"], found["field"], found["buffer"]) == (1, "buffer", "gateup")
    found = first_difference(_numeric(), _numeric(outputs=dict(output="ee")))
    assert (found["field"], found["output"]) == ("output", "output")
    shorter = _record(calls=[_call()])
    assert first_difference(_record(), shorter)["field"] == "launch count"


def test_one_case_that_cannot_be_compared_outranks_the_others(tmp_path, capsys) -> None:
    before = _write(
        tmp_path / "before.json",
        _record("same"),
        _record("differs"),
        _record("other_box"),
    )
    after = _write(
        tmp_path / "after.json",
        _record("same"),
        _record("differs", calls=[_call("kernel_z"), _call("kernel_b")]),
        _record(
            "other_box", device=dict(name="NVIDIA B300", capability=[10, 3], sms=152)
        ),
    )
    assert main([before, after]) == 3
    out = capsys.readouterr().out
    assert out.count("SAME ") == 1 and "SAME same" in out
    assert "DIFFERS differs" in out and "NOT-COMPARABLE other_box: device" in out


def test_a_recorder_without_evidence_exports_nothing() -> None:
    recorder = LaunchRecorder()
    with pytest.raises(ValueError, match="no launch was recorded"):
        recorder.artifact("case", check="structure", fixture={})
    recorder.launches.append(_call())
    with pytest.raises(ValueError, match="a declared buffer or a named output"):
        recorder.artifact("case", check="numeric", fixture={})
    with pytest.raises(ValueError, match="unknown check"):
        recorder.artifact("case", check="approximate", fixture={})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
