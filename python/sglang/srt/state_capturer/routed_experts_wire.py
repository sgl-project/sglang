"""Host-side routed expert serialization, independent of the inference runtime."""

from typing import TYPE_CHECKING, Optional

import numpy as np
import pybase64

from sglang.srt.environ import envs

if TYPE_CHECKING:
    import torch

_WIRE_DTYPES = {
    "int32": np.dtype(np.int32),
    "uint16": np.dtype(np.uint16),
    "uint8": np.dtype(np.uint8),
}


def _wire_dtype(name: str) -> np.dtype:
    if not isinstance(name, str) or name.strip().lower() not in _WIRE_DTYPES:
        raise ValueError(
            f"Unsupported routed experts wire dtype {name!r}. "
            f"Supported values are: {', '.join(_WIRE_DTYPES)}."
        )
    return _WIRE_DTYPES[name.strip().lower()]


def encode_routed_experts_for_wire(routed_experts: "torch.Tensor") -> tuple[str, str]:
    """Return a flat row-major payload and its actual dtype.

    Preserve int32 when the requested unsigned dtype cannot represent every
    expert ID, including negative sentinels. The advertised dtype must describe
    the bytes actually sent, rather than the configured compression preference.
    """
    array = routed_experts.numpy()
    target = _wire_dtype(envs.SGLANG_ROUTED_EXPERTS_DTYPE.get())
    if target.kind == "u" and array.size:
        bounds = np.iinfo(target)
        if array.min() < bounds.min or array.max() > bounds.max:
            target = _WIRE_DTYPES["int32"]
    array = array.astype(target, copy=False)
    return pybase64.b64encode(array.tobytes()).decode("utf-8"), target.name


def extract_routed_experts_from_meta_info(
    data: dict,
    num_layers: Optional[int] = None,
    topk: Optional[int] = None,
) -> np.ndarray:
    """Decode using the sender's dtype, defaulting legacy payloads to int32.

    Optional dimensions reshape flat data into (positions, num_layers * topk).
    Decoder environment settings never override the sender's wire format.
    """
    meta_info = data["meta_info"]
    dtype_name = meta_info.get("routed_experts_dtype")
    dtype = _wire_dtype("int32" if dtype_name is None else dtype_name)
    decoded = np.frombuffer(
        pybase64.b64decode(meta_info["routed_experts"].encode("utf-8")), dtype=dtype
    )
    if num_layers is None and topk is None:
        return decoded
    if num_layers is None or topk is None:
        raise ValueError("num_layers and topk must be provided together.")
    if num_layers <= 0 or topk <= 0:
        raise ValueError("num_layers and topk must be positive.")
    return decoded.reshape(-1, num_layers * topk)
