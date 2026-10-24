"""Regression tests for the invalid suite that escaped /rerun-test in #44938.

Run with: python -m pytest scripts/ci/test_rerun_suite_validation.py
Only the slash handler's dependencies (PyGithub and PyYAML) are needed.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_HANDLER_PATH = Path(__file__).parent / "utils/slash_command_handler.py"
_spec = importlib.util.spec_from_file_location("slash_command_handler", _HANDLER_PATH)
handler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(handler)


def write_test(tmp_path, monkeypatch, registration):
    monkeypatch.chdir(tmp_path)
    path = Path("test/registered/test_example.py")
    path.parent.mkdir(parents=True)
    path.write_text(registration)
    return str(path)


@pytest.mark.parametrize("also_valid", [False, True])
def test_invalid_suite_is_reported_without_dispatch(tmp_path, monkeypatch, also_valid):
    """A valid H100 runner must not hide an undefined stage/runner combination."""
    registration = (
        'register_cuda_ci(est_time=300, stage="extra-a", runner_config="4-gpu-h100")\n'
    )
    if also_valid:
        registration += 'register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-h100")\n'
    path = write_test(tmp_path, monkeypatch, registration)
    repo, pr, comment = MagicMock(), MagicMock(), MagicMock()
    with (
        patch.object(handler, "_check_rebase_gate", return_value=(True, None)),
        patch.object(
            handler,
            "_dispatch_batch",
            side_effect=AssertionError("Invalid suite dispatched"),
        ),
    ):
        result = handler.handle_rerun_test(
            repo, pr, comment, {}, [path], "token", skip_permission_check=True
        )
    assert result is False
    body = pr.create_issue_comment.return_value.edit.call_args.args[0]
    assert "Invalid CUDA suite `extra-a-test-4-gpu-h100`" in body
    assert path in body
    assert "test/run_suite.py" in body
    repo.get_workflow.assert_not_called()


@pytest.mark.parametrize("stage", ["extra-b", "nightly", "weekly"])
def test_valid_catalogs_remain_dispatchable(tmp_path, monkeypatch, stage):
    """Validation must accept extra, nightly and OTHER_SUITES registrations."""
    path = write_test(
        tmp_path,
        monkeypatch,
        f'register_cuda_ci(est_time=300, stage="{stage}", runner_config="4-gpu-h100")',
    )
    entries = handler._resolve_test_spec(path)
    assert len(entries) == 1
    assert entries[0]["error"] is None
    assert entries[0]["runs_on"] == "4-gpu-h100"


def test_cpu_registration_cannot_hide_invalid_suite(tmp_path, monkeypatch):
    """A valid CPU registration must not mask a second invalid registration."""
    path = write_test(
        tmp_path,
        monkeypatch,
        'register_cpu_ci(est_time=1, suite="base-a-test-cpu")\n'
        'register_cpu_ci(est_time=1, suite="base-a-test-1-gpu-small")\n',
    )
    entries = handler._resolve_test_spec(path)
    assert len(entries) == 1
    assert "Invalid CPU suite `base-a-test-1-gpu-small`" in entries[0]["error"]
