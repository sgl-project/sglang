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
        elif action == "reset_tensors":
            self._reset_tensors()
        elif action == "compare":
            self._compare()
        else:
            raise Exception(f"Unsupported {action=}")

    def _snapshot(self):
        named_tensors = [
            (name, param.data.detach().cpu()) for name, param in self._model_state()
        ]
        self._snapshot_tensors = dict(named_tensors)
        assert len(self._snapshot_tensors) == len(
            named_tensors
        ), f"should not have duplicated tensor name"

    def _reset_tensors(self):
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


def _check_tensors(
    expect_tensors: Dict[str, torch.Tensor], actual_tensors: Dict[str, torch.Tensor]
):
    from sglang.srt.debug_utils.dumper import get_tensor_info

    assert len(expect_tensors) == len(actual_tensors)

    good_names = []
    error_messages = []

    for name in expect_tensors:
        expect = expect_tensors[name].cuda()
        actual = actual_tensors[name].cuda()

        if torch.all(expect == actual):
            good_names.append(name)
        else:
            abs_diff = (actual.float() - expect.float()).abs()
            error_messages.append(
                f"name={name} "
                f"max_abs_err={abs_diff.max()} "
                f"mean_abs_err={abs_diff.mean()} "
                f"{get_tensor_info(expect)=} "
                f"{get_tensor_info(actual)=} "
            )

    logger.info(f"[check_tensors] passed: {good_names}")
    if len(error_messages) > 0:
        raise Exception(f"check tensor equality failed:\n" + "\n".join(error_messages))


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point:
        return torch.rand(shape, device=device, dtype=torch.float32).to(dtype)

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    info = torch.iinfo(dtype)
    return torch.randint(low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype)
