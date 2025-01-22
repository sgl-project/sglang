from typing import Dict, Tuple

import torch


def _get_cuda_stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


_cache_buf: Dict[Tuple[str, torch.device], torch.Tensor] = {}


def _get_cache_buf(name: str, bytes: int, device: torch.device) -> torch.Tensor:
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf
