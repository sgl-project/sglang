"""push_run's dedup must not keep a failed run's tensors under a passed row."""

from pathlib import Path

import pytest

from sglang.test import precision_baseline_store as store
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MODEL = "org/model"
_SHA = "abcdef0123456789"


class _FakeApi:
    uploads: list[list[str]] = []

    def __init__(self, token=None):
        pass

    def upload_folder(self, *, folder_path, **_kwargs):
        _FakeApi.uploads.append(
            sorted(p.name for p in Path(folder_path).rglob("*") if p.is_file())
        )

    def upload_file(self, **_kwargs):
        pass


@pytest.fixture
def push(monkeypatch, tmp_path):
    _FakeApi.uploads = []
    monkeypatch.setattr(store, "HfApi", _FakeApi)
    _, date_path = store._today_path()
    run_path = f"{store._sanitize_model_name(_MODEL)}/{date_path}/run-{_SHA[:7]}"
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    (tensors / "x.pt").write_bytes(b"0")

    def run(*, pass_label, earlier_push_at_same_path):
        rows = [{"run_path": run_path}] if earlier_push_at_same_path else []
        monkeypatch.setattr(store, "_read_manifest", lambda _config: (rows, ""))
        store.push_run(
            config=store.HfStoreConfig(repo="r", revision="main"),
            model=_MODEL,
            sglang_commit=_SHA,
            today_tensors_dir=tensors,
            meta={"pass_label": pass_label},
        )
        return "x.pt" in _FakeApi.uploads[-1]

    return run


def test_passed_push_replaces_tensors_left_by_a_failed_push(push):
    assert push(pass_label="passed", earlier_push_at_same_path=True)


def test_failed_push_keeps_the_tensors_already_at_its_path(push):
    assert not push(pass_label="failed", earlier_push_at_same_path=True)


@pytest.mark.parametrize("pass_label", ["passed", "failed"])
def test_first_push_uploads_tensors(push, pass_label):
    assert push(pass_label=pass_label, earlier_push_at_same_path=False)
