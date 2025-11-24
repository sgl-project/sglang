import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner
        self._snapshot_tensors = None

    def handle(self, action: str):
        logger.info(f"[WeightChecker] handle action={action}")
        if action == "snapshot":
            self._snapshot()
        elif action == "reset_param":
            self._reset_param()
        elif action == "compare":
            self._compare()
        else:
            raise Exception(f"Unsupported {action=}")

    def _snapshot(self):
        named_tensors = [(name, param.data.detach().cpu()) for name, param in self._model_state()]
        self._snapshot_tensors = dict(named_tensors)
        assert len(self._snapshot_tensors) == len(named_tensors), f"should not have duplicated tensor name"

    def _reset_param(self):
        for name, param in self._model_state():
            param.copy_(_random_like(param))

    def _compare(self):
        assert self._snapshot_tensors is not None

        _check_tensors(
            expect_tensors=self._snapshot_tensors,
            actual_tensors=dict(self._model_state()),
        )

    def _model_state(self):
        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)
        yield from self._model_runner.model.named_parameters()


def _check_tensors(expect_tensors: Dict[str, torch.Tensor], actual_tensors: Dict[str, torch.Tensor]):
    assert len(expect_tensors) == len(actual_tensors)

    for name in expect_tensors:
        expect = expect_tensors[name]
        actual = actual_tensors[name]
        if not torch.all(curr_tensor == snapshot_tensor):
            TODO


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point:
        gen_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        tmp = torch.rand(shape, device=device, dtype=gen_dtype)
        return tmp.to(dtype)

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    # Integer types
    if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        info = torch.iinfo(dtype)
        low = int(info.min)
        high = int(info.max)
        return torch.randint(
            low=low, high=high, size=shape, device=device, dtype=torch.int64
        )

    raise TypeError(f"unsupported dtype: {dtype}")
