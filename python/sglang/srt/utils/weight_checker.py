import logging

import torch

logger = logging.getLogger(__name__)

class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner

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
        TODO

    def _reset_param(self):
        for name, param in self._model_state():
            param.copy_(_random_like(param))

    def _compare(self):
        TODO

    def _model_state(self):
        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)
        yield from self._model_runner.model.named_parameters()

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
        return torch.randint(low=low, high=high, size=shape, device=device, dtype=torch.int64)

    raise TypeError(f"unsupported dtype: {dtype}")

