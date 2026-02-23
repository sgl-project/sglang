import functools
from pathlib import Path
from typing import Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


def argmax_coord(x: torch.Tensor) -> Tuple[int, ...]:
    flat_idx = x.argmax()
    return tuple(idx.item() for idx in torch.unravel_index(flat_idx, x.shape))


def compute_smaller_dtype(
    dtype_a: torch.dtype, dtype_b: torch.dtype
) -> Optional[torch.dtype]:
    info_dict = {
        (torch.float32, torch.bfloat16): torch.bfloat16,
        # ... add more ...
    }
    return info_dict.get((dtype_a, dtype_b)) or info_dict.get((dtype_b, dtype_a))


def try_unify_shape(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    x_shape = x.shape
    num_dim_to_remove = len(x_shape) - len(target_shape)
    if (x_shape[num_dim_to_remove:] == target_shape) and all(
        val == 1 for val in x_shape[:num_dim_to_remove]
    ):
        return functools.reduce(lambda a, _: a.squeeze(0), range(num_dim_to_remove), x)

    return x


# Copied from DeepGEMM
def calc_rel_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def load_object(path: Path) -> Optional[torch.Tensor]:
    try:
        x = torch.load(path, weights_only=False)
    except Exception as e:
        print(f"Skip load {path} since error {e}")
        return None

    if isinstance(x, dict) and "value" in x:
        x = x["value"]

    if not isinstance(x, torch.Tensor):
        print(f"Skip load {path} since {type(x)=} is not a Tensor ({x=})")
        return None
    return x.cuda()
