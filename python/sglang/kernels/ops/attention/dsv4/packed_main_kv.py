from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.mem_cache.dsv41_main_kv_layout import PackedMainKVView

from .utils import make_name


@cache_once
def _jit_packed_main_kv_module(page_slots: int):
    args = make_cpp_args(page_slots)
    kernel = f"PackDSV41MainKVKernel<{args}>"
    return load_jit(
        make_name("packed_main_kv_store"),
        *args,
        cuda_files=["deepseek_v4/packed_main_kv_store.cuh"],
        cuda_wrappers=[
            ("run", f"{kernel}::run"),
            ("run_debug", f"{kernel}::run_debug"),
        ],
    )


def pack_dsv41_main_kv_fp4(
    *,
    latent: torch.Tensor,
    freqs_cis: torch.Tensor,
    slots: torch.Tensor,
    view: PackedMainKVView,
    error_flag: torch.Tensor | None = None,
) -> None:
    """Pack post-RMSNorm, pre-RoPE BF16 latent rows into a typed Main-KV view."""
    if latent.ndim != 2 or latent.shape[1] != 512:
        raise ValueError(
            "packed Main KV latent must have shape [num_tokens, 512], "
            f"got {tuple(latent.shape)}"
        )
    if latent.dtype is not torch.bfloat16 or not latent.is_cuda:
        raise ValueError("packed Main KV writer requires CUDA BF16 latent")
    if torch.version.cuda is None:
        raise ValueError("packed Main KV writer requires an NVIDIA CUDA runtime")
    if torch.cuda.get_device_capability(latent.device)[0] < 9:
        raise ValueError("packed Main KV writer requires an SM90-or-newer GPU")
    if slots.ndim != 1 or slots.shape[0] != latent.shape[0]:
        raise ValueError(
            "packed Main KV slots must have one entry per latent row, "
            f"got {tuple(slots.shape)} for {latent.shape[0]} rows"
        )
    if slots.dtype not in (torch.int32, torch.int64) or slots.device != latent.device:
        raise ValueError(
            "packed Main KV slots must be CUDA int32/int64 on the input device"
        )
    if freqs_cis.is_complex():
        freqs_cis = torch.view_as_real(freqs_cis).flatten(-2)
    if (
        freqs_cis.ndim != 2
        or freqs_cis.shape != (latent.shape[0], 64)
        or freqs_cis.dtype is not torch.float32
        or freqs_cis.device != latent.device
        or not freqs_cis.is_contiguous()
    ):
        raise ValueError(
            "packed Main KV RoPE frequencies must be contiguous CUDA FP32 "
            f"[{latent.shape[0]}, 64]"
        )
    if view.storage.device != latent.device:
        raise ValueError("packed Main KV storage must be on the input device")
    if error_flag is not None and (
        error_flag.shape != (1,)
        or error_flag.dtype is not torch.int32
        or error_flag.device != latent.device
    ):
        raise ValueError("error_flag must be CUDA int32 with shape [1]")

    module = _jit_packed_main_kv_module(view.spec.page_slots)
    args = (latent, freqs_cis, slots, view.storage)
    if error_flag is None:
        module.run(*args)
    else:
        module.run_debug(*args, error_flag)


def flash_c1_decode_pack_main_kv_fp4(**kwargs) -> None:
    pack_dsv41_main_kv_fp4(**kwargs)


def flash_c2_decode_pack_main_kv_fp4(**kwargs) -> None:
    pack_dsv41_main_kv_fp4(**kwargs)


def flash_c1_prefill_pack_main_kv_fp4(**kwargs) -> None:
    pack_dsv41_main_kv_fp4(**kwargs)


def flash_c2_prefill_pack_main_kv_fp4(**kwargs) -> None:
    pack_dsv41_main_kv_fp4(**kwargs)
