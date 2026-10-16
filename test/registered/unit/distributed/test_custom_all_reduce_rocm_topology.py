from types import SimpleNamespace

import pytest

from sglang.srt.distributed.device_communicators import custom_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _IpcSetupStarted(Exception):
    pass


def _build(monkeypatch, *, full_xgmi, deterministic):
    def allocate_meta_buffer(size):
        raise _IpcSetupStarted

    monkeypatch.setattr(custom_all_reduce, "_is_hip", True)
    monkeypatch.setattr(
        custom_all_reduce,
        "ops",
        SimpleNamespace(
            IS_CUSTOM_AR_AVAILABLE=True,
            meta_size=lambda: 0,
            allocate_meta_buffer=allocate_meta_buffer,
        ),
    )
    monkeypatch.setattr(
        custom_all_reduce, "_use_amd_deterministic_impl", lambda: deterministic
    )
    monkeypatch.setattr(custom_all_reduce.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(custom_all_reduce.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        custom_all_reduce,
        "can_use_custom_all_reduce_with_nvlink",
        lambda **kwargs: full_xgmi,
    )
    return custom_all_reduce.CustomAllreduce(group=object(), device="cpu")


def test_rocm_without_xgmi_skips_ipc_setup(monkeypatch):
    ca = _build(monkeypatch, full_xgmi=False, deterministic=False)
    assert ca.disabled


@pytest.mark.parametrize(
    ("full_xgmi", "deterministic"), [(True, False), (False, True), (True, True)]
)
def test_rocm_usable_custom_ar_still_sets_up_ipc(monkeypatch, full_xgmi, deterministic):
    with pytest.raises(_IpcSetupStarted):
        _build(monkeypatch, full_xgmi=full_xgmi, deterministic=deterministic)
