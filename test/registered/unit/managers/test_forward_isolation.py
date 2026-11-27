from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

import dataclasses
from types import SimpleNamespace

import pytest

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler


@dataclasses.dataclass
class DerivedBatch(ScheduleBatch):
    extra_field: object = None


class Sampling:
    def copy_for_forward(self):
        return Sampling()


@pytest.mark.parametrize("batch_type", [ScheduleBatch, DerivedBatch])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("spec", [False, True])
@pytest.mark.parametrize("error", [False, True])
def test_snapshot(batch_type, overlap, spec, error):
    scheduler = object.__new__(Scheduler)
    scheduler.batch_record_ct = 0
    scheduler.batch_record_buf = [None, None]
    batch = batch_type(
        reqs=[],
        spec_algorithm=SimpleNamespace(is_none=lambda: not spec),
        sampling_info=Sampling(),
    )
    original = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
    forwarded = None
    try:
        with scheduler._forward_isolation(batch, overlap=overlap):
            forwarded = batch.sampling_info
            assert forwarded is not original["sampling_info"]
            for name in original:
                setattr(batch, name, object())
            batch.dynamic_field = "preserved"
            if error:
                raise RuntimeError("test")
    except RuntimeError:
        assert error
    if spec:
        for name, value in original.items():
            assert getattr(batch, name) is value
    else:
        assert batch.sampling_info is original["sampling_info"]
        assert batch.reqs is not original["reqs"]
    assert batch.dynamic_field == "preserved"
    if overlap:
        pinned = scheduler.batch_record_buf[scheduler.batch_record_ct]
        assert pinned[0] is batch
        if spec:
            saved, copied = pinned[1]
            assert saved == original and copied is forwarded
        else:
            assert any(x is forwarded for x in pinned[1])
    else:
        assert scheduler.batch_record_buf == [None, None]
