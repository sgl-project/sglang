import torch

_is_low_precision_mode_stack = []


class LowPrecisionMode:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        global _is_low_precision_mode_stack
        _is_low_precision_mode_stack.append(self.enabled)

    def __exit__(self, exc_type, exc_value, traceback):
        global _is_low_precision_mode_stack
        _is_low_precision_mode_stack.pop()


def is_low_precision_mode() -> bool:
    global _is_low_precision_mode_stack
    if len(_is_low_precision_mode_stack) == 0:
        return False
    return _is_low_precision_mode_stack[-1]


def optional_cast_to_bf16_and_cast_back(tensor: torch.Tensor) -> torch.Tensor:
    assert (
        tensor.dtype == torch.float32
    ), "Input tensor must be of dtype torch.float32 for optional casting."
    if is_low_precision_mode():
        tensor_bf16 = tensor.to(torch.bfloat16)
        tensor_fp32 = tensor_bf16.to(torch.float32)
        return tensor_fp32
    else:
        return tensor
