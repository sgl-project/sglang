"""Guards for how this suite decides which GPUs it may use.

Every spawn in ``tests/e2e/`` is bound to a card by two pieces of arithmetic:
``_detect_gpu_count`` sizing the allocator from this process's
``CUDA_VISIBLE_DEVICES`` allotment, and ``model_pool.resolve_device_ids``
mapping the allocator's logical indices back onto that allotment. Get either
wrong and the suite runs on a card it does not own, or silently does not run at
all — both of which have happened, and neither of which turns CI red on its own.

Pure Python: no GPU, no subprocess, no model.
"""

from __future__ import annotations

import subprocess

import conftest
import pytest
from infra import model_pool


def _fixture_function(fixture):
    """The undecorated function behind a pytest fixture.

    pytest >= 8.4 wraps it in a ``FixtureFunctionDefinition``; older versions
    hand back the function itself.
    """
    for attr in ("_fixture_function", "__wrapped__"):
        raw = getattr(fixture, attr, None)
        if raw is not None:
            return raw
    return fixture


def _fake_nvidia_smi(
    monkeypatch, *, gpus: int = 8, raises: BaseException | None = None
):
    def check_output(*args, **kwargs):
        if raises is not None:
            raise raises
        return "\n".join(str(i) for i in range(gpus)).encode() + b"\n"

    monkeypatch.setattr(conftest.subprocess, "check_output", check_output)


# --- sizing the allocator --------------------------------------------------


def test_detect_gpu_count_counts_the_allotment_not_the_host(monkeypatch):
    _fake_nvidia_smi(monkeypatch, gpus=8)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert conftest._detect_gpu_count() == 2


def test_detect_gpu_count_reports_none_when_the_host_has_no_nvidia_stack(monkeypatch):
    """A stray CUDA_VISIBLE_DEVICES on a CPU-only box must still skip."""
    _fake_nvidia_smi(monkeypatch, raises=FileNotFoundError("nvidia-smi"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert conftest._detect_gpu_count() == 0


def test_detect_gpu_count_treats_hide_everything_as_no_gpus(monkeypatch):
    _fake_nvidia_smi(monkeypatch, gpus=8)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    assert conftest._detect_gpu_count() == 0


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0),
        subprocess.CalledProcessError(
            returncode=6, cmd="nvidia-smi", stderr=b"Failed to initialize NVML"
        ),
    ],
    ids=["hung", "errored"],
)
def test_detect_gpu_count_fails_when_nvidia_smi_cannot_answer(monkeypatch, failure):
    """A wedged driver must not read as "no GPUs" — that skips every test in
    this directory and reports a green job that ran nothing."""
    _fake_nvidia_smi(monkeypatch, raises=failure)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(pytest.fail.Exception, match="nvidia-smi"):
        conftest._detect_gpu_count()


def test_no_index_the_allocator_hands_out_escapes_the_allotment(monkeypatch):
    """The coupling `resolve_device_ids`'s bounds check exists to catch."""
    _fake_nvidia_smi(monkeypatch, gpus=8)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    allocator = conftest.GPUAllocator(conftest._detect_gpu_count())
    picked = allocator.acquire(allocator.total)

    assert model_pool.resolve_device_ids(picked) == ["2", "3"]


# --- handing the cards out -------------------------------------------------


def test_acquire_skips_when_the_allotment_is_too_small():
    allocator = conftest.GPUAllocator(1)

    with pytest.raises(pytest.skip.Exception, match="allotment is 1"):
        allocator.acquire(2)


def test_acquire_fails_when_another_reservation_holds_the_rest():
    """The session server holds one card of two. A 2-GPU test must go red, not
    skip: a skipped acceptance test on a green run is invisible."""
    allocator = conftest.GPUAllocator(2)
    allocator.acquire(1)

    with pytest.raises(RuntimeError, match="holding a reservation"):
        allocator.acquire(2)


def test_release_rejects_a_double_release():
    allocator = conftest.GPUAllocator(2)
    ids = allocator.acquire(1)
    allocator.release(ids)

    with pytest.raises(RuntimeError, match="double release"):
        allocator.release(ids)
    assert allocator.acquire(2) == [0, 1]


# --- the session server's reservation --------------------------------------


def test_session_server_releases_its_gpu_when_the_spawn_fails(monkeypatch):
    def explode(gpu_ids):
        raise FileNotFoundError("python3")
        yield  # pragma: no cover - generator marker

    monkeypatch.setattr(conftest, "_serve_session_sglang", explode)
    allocator = conftest.GPUAllocator(2)

    generator = _fixture_function(conftest.sglang_server)(allocator)
    with pytest.raises(FileNotFoundError):
        next(generator)

    assert allocator.acquire(2) == [0, 1]


def test_session_server_releases_its_gpu_on_teardown(monkeypatch):
    def serve(gpu_ids):
        yield "http://localhost:30000"

    monkeypatch.setattr(conftest, "_serve_session_sglang", serve)
    allocator = conftest.GPUAllocator(2)

    generator = _fixture_function(conftest.sglang_server)(allocator)
    assert next(generator) == "http://localhost:30000"
    with pytest.raises(RuntimeError, match="holding a reservation"):
        allocator.acquire(2)

    with pytest.raises(StopIteration):
        next(generator)
    assert allocator.acquire(2) == [0, 1]
