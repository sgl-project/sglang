from __future__ import annotations

import struct
from array import array
from typing import Sequence, Union

import msgspec
import numpy as np
import torch

from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

# Stable wire IDs. Changing these requires updating the golden-wire test.
_MSGPACK_EXT_ARRAY = 1
_MSGPACK_EXT_TORCH_TENSOR = 2
_MSGPACK_EXT_NP_ARRAY = 3
_MSGPACK_EXT_SHM_POINTER_MM_DATA = 4
_MSGPACK_EXT_CUDA_IPC_TENSOR_PROXY = 5
_MSGPACK_BUFFER_METADATA_SIZE = struct.Struct(">I")


def _pack_ext(code: int, obj: object) -> msgspec.msgpack.Ext:
    return msgspec.msgpack.Ext(code, msgspec.msgpack.encode(obj, enc_hook=enc_hook))


def _unpack_ext(data: memoryview) -> object:
    return msgspec.msgpack.decode(data, ext_hook=ext_hook)


def _pack_buffer_ext(
    code: int, metadata: object, raw_data: memoryview
) -> msgspec.msgpack.Ext:
    metadata_bytes = msgspec.msgpack.encode(metadata)
    payload = bytearray(_MSGPACK_BUFFER_METADATA_SIZE.pack(len(metadata_bytes)))
    payload.extend(metadata_bytes)
    payload.extend(raw_data)
    return msgspec.msgpack.Ext(code, payload)


def _unpack_buffer_ext(data: memoryview) -> tuple[object, memoryview]:
    if len(data) < _MSGPACK_BUFFER_METADATA_SIZE.size:
        raise msgspec.DecodeError("MessagePack buffer extension is missing metadata")

    (metadata_size,) = _MSGPACK_BUFFER_METADATA_SIZE.unpack_from(data)
    raw_data_offset = _MSGPACK_BUFFER_METADATA_SIZE.size + metadata_size
    if raw_data_offset > len(data):
        raise msgspec.DecodeError("MessagePack buffer extension has invalid metadata")

    metadata = msgspec.msgpack.decode(
        data[_MSGPACK_BUFFER_METADATA_SIZE.size : raw_data_offset]
    )
    return metadata, data[raw_data_offset:]


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _torch_dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _restore_torch_tensor(
    shape: Sequence[int],
    dtype: str,
    data: Union[bytes, memoryview],
    device: str = "cpu",
) -> torch.Tensor:
    tensor_dtype = _torch_dtype_from_name(dtype)
    if len(data) == 0:
        tensor = torch.empty(shape, dtype=tensor_dtype, device="cpu")
    else:
        tensor = torch.frombuffer(bytearray(data), dtype=tensor_dtype).reshape(shape)
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


def _to_msgpack_state(obj: object) -> object:
    if isinstance(obj, torch.dtype):
        return {"__torch_dtype__": _torch_dtype_name(obj)}
    if isinstance(obj, torch.device):
        return {"__torch_device__": str(obj)}
    if isinstance(obj, np.dtype):
        return {"__np_dtype__": obj.str}
    if isinstance(obj, dict):
        return {key: _to_msgpack_state(value) for key, value in obj.items()}
    if isinstance(obj, torch.Size):
        return {"__torch_size__": list(obj)}
    if isinstance(obj, tuple):
        return {"__tuple__": [_to_msgpack_state(value) for value in obj]}
    if isinstance(obj, list):
        return [_to_msgpack_state(value) for value in obj]
    return obj


def _from_msgpack_state(obj: object) -> object:
    if isinstance(obj, dict):
        if "__torch_dtype__" in obj:
            return _torch_dtype_from_name(obj["__torch_dtype__"])
        if "__torch_device__" in obj:
            return torch.device(obj["__torch_device__"])
        if "__np_dtype__" in obj:
            return np.dtype(obj["__np_dtype__"])
        if "__torch_size__" in obj:
            return torch.Size(obj["__torch_size__"])
        if "__tuple__" in obj:
            return tuple(_from_msgpack_state(value) for value in obj["__tuple__"])
        return {key: _from_msgpack_state(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_from_msgpack_state(value) for value in obj]
    return obj


def _is_shm_pointer_mm_data(obj: object) -> bool:
    cls = type(obj)
    return cls.__name__ == "ShmPointerMMData" and cls.__module__.endswith(
        ".managers.mm_utils"
    )


def _encode_shm_pointer_mm_data(obj: object) -> object:
    return _to_msgpack_state(obj.__getstate__())


def _decode_shm_pointer_mm_data(state: dict[str, object]) -> object:
    from sglang.srt.managers.mm_utils import ShmPointerMMData

    obj = ShmPointerMMData.__new__(ShmPointerMMData)
    obj.__setstate__(_from_msgpack_state(state))
    return obj


def _encode_cuda_ipc_tensor_proxy(obj: CudaIpcTensorTransportProxy) -> object:
    return {
        "proxy_state": _to_msgpack_state(obj.proxy_state),
        "sync_data_meta": _to_msgpack_state(obj.sync_data_meta),
    }


def _decode_cuda_ipc_tensor_proxy(
    state: dict[str, object],
) -> CudaIpcTensorTransportProxy:
    obj = CudaIpcTensorTransportProxy.__new__(CudaIpcTensorTransportProxy)
    obj.proxy_state = _from_msgpack_state(state["proxy_state"])
    obj.reconstruct_tensor = None
    obj.sync_data_meta = _from_msgpack_state(state["sync_data_meta"])
    obj.sync_buffer = None
    obj._consumer_acknowledged = False
    return obj


def enc_hook(obj: object) -> object:
    if isinstance(obj, array):
        return _pack_buffer_ext(
            _MSGPACK_EXT_ARRAY, obj.typecode, memoryview(obj).cast("B")
        )
    if isinstance(obj, torch.Tensor):
        tensor_dtype = _torch_dtype_name(obj.dtype)
        tensor = obj.cpu().contiguous()
        raw_data = tensor.reshape(-1).view(torch.uint8).numpy().data
        return _pack_buffer_ext(
            _MSGPACK_EXT_TORCH_TENSOR,
            (tuple(obj.shape), tensor_dtype, str(obj.device)),
            raw_data,
        )
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        raw_data = arr.reshape(-1).view(np.uint8).data
        return _pack_buffer_ext(
            _MSGPACK_EXT_NP_ARRAY,
            (arr.shape, arr.dtype.str),
            raw_data,
        )
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, CudaIpcTensorTransportProxy):
        return _pack_ext(
            _MSGPACK_EXT_CUDA_IPC_TENSOR_PROXY,
            _encode_cuda_ipc_tensor_proxy(obj),
        )
    if _is_shm_pointer_mm_data(obj):
        return _pack_ext(
            _MSGPACK_EXT_SHM_POINTER_MM_DATA,
            _encode_shm_pointer_mm_data(obj),
        )
    raise TypeError(
        f"Cannot msgpack encode object of type {type(obj)} with enc_hook. "
        "Use an explicit PickleWrapper field via wrap_as_pickle(...) for "
        "arbitrary payloads, or add a dedicated enc_hook/dec_hook branch "
        "for this transport type."
    )


def dec_hook(tp: type, obj: object) -> object:
    if isinstance(obj, tp):
        return obj
    if tp is array:
        typecode, raw_data = obj
        res = array(typecode)
        res.frombytes(raw_data)
        return res
    if tp is torch.Tensor:
        shape, dtype, data, *device = obj
        return _restore_torch_tensor(shape, dtype, data, device[0] if device else "cpu")
    if tp is np.ndarray:
        shape, dtype, data = obj
        return np.frombuffer(data, dtype=np.dtype(dtype)).copy().reshape(shape)
    raise TypeError(
        f"Cannot msgpack decode object of type {type(obj)} as {tp} with "
        "dec_hook. Use an explicit PickleWrapper field via wrap_as_pickle(...) "
        "and unwrap_from_pickle(...) for arbitrary payloads, or add a "
        "dedicated enc_hook/dec_hook branch for this transport type."
    )


def ext_hook(code: int, data: memoryview) -> object:
    if code not in (
        _MSGPACK_EXT_ARRAY,
        _MSGPACK_EXT_TORCH_TENSOR,
        _MSGPACK_EXT_NP_ARRAY,
        _MSGPACK_EXT_SHM_POINTER_MM_DATA,
        _MSGPACK_EXT_CUDA_IPC_TENSOR_PROXY,
    ):
        return msgspec.msgpack.Ext(code, bytes(data))

    if code == _MSGPACK_EXT_ARRAY:
        typecode, raw_data = _unpack_buffer_ext(data)
        res = array(typecode)
        res.frombytes(raw_data)
        return res
    if code == _MSGPACK_EXT_TORCH_TENSOR:
        metadata, raw_data = _unpack_buffer_ext(data)
        shape, dtype, device = metadata
        return _restore_torch_tensor(shape, dtype, raw_data, device)
    if code == _MSGPACK_EXT_NP_ARRAY:
        metadata, raw_data = _unpack_buffer_ext(data)
        shape, dtype = metadata
        return np.frombuffer(raw_data, dtype=np.dtype(dtype)).copy().reshape(shape)
    if code == _MSGPACK_EXT_SHM_POINTER_MM_DATA:
        return _decode_shm_pointer_mm_data(_unpack_ext(data))
    if code == _MSGPACK_EXT_CUDA_IPC_TENSOR_PROXY:
        return _decode_cuda_ipc_tensor_proxy(_unpack_ext(data))
    raise AssertionError(f"Unhandled known MessagePack extension code: {code}")
