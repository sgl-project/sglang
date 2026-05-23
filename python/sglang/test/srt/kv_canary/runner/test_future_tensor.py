from __future__ import annotations

from typing import cast

import pytest
import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensors


class FakeEvent:
    def __init__(self) -> None:
        self.synchronize_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1


def test_wait_clears_fields_and_rejects_second_wait() -> None:
    tensor = torch.tensor([1, 2, 3])
    event = FakeEvent()
    future = FutureTensors(_data=tensor, _event=cast(torch.cuda.Event, event))

    assert future.wait() is tensor

    assert event.synchronize_count == 1
    assert future._tensor is None
    assert future._event is None
    with pytest.raises(RuntimeError, match="called more than once"):
        future.wait()

    assert event.synchronize_count == 1
