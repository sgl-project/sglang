from typing import List

import torch

import sglang.srt.hardware_backend.npu.cmo
from sglang.srt.utils.custom_op import register_custom_op


@torch.library.custom_op("sglang::wait_cmo_stream", mutates_args=())
def wait_cmo_stream() -> None:
    if sglang.srt.hardware_backend.npu.cmo.get_cmo_stream():
        sglang.srt.hardware_backend.npu.cmo.wait_cmo_stream()


@wait_cmo_stream.register_fake
def wait_cmo_stream_fake() -> None:
    pass


def get_cmo_stream() -> bool:
    return True


def prepare_weight_cache_fake(handle: torch.Tensor, cache: List[torch.Tensor]) -> None:
    pass


@register_custom_op(fake_impl=prepare_weight_cache_fake)
def prepare_weight_cache(handle: torch.Tensor, cache: List[torch.Tensor]) -> None:
    sglang.srt.hardware_backend.npu.cmo.prepare_weight_cache(handle, cache)
