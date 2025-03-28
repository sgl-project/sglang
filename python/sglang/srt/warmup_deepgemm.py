# TODO shall we put this file elsewhere?

import json
import logging
from typing import Any, List, Dict, Tuple

import torch
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
from sglang.srt.utils import get_bool_env_var

try:
    from deep_gemm import ceil_div, get_col_major_tma_aligned_tensor
except ImportError:
    pass

logger = logging.getLogger(__name__)


# --------------------------------------- warmup -------------------------------------

def warmup(model):
    from sglang.srt.layers.quantization.fp8_kernel import enable_jit_deepgemm
    if not enable_jit_deepgemm:
        return

    infos_source = _INFOS_SOURCE_OF_MODEL.get(type(model))
    if infos_source is None:
        return

    _warmup_by_infos(infos_source())


def _compute_infos_deepseek():
    return [
        TODO,
    ]


_INFOS_SOURCE_OF_MODEL = {
    DeepseekV2ForCausalLM: _compute_infos_deepseek,
    DeepseekV3ForCausalLM: _compute_infos_deepseek,
}


def _warmup_by_infos(infos: List[Dict[str, Any]]):
    for info in infos:
        TODO


# Copied from DeepGEMM's `test_core.py` :: `construct`
def _construct_gemm_inputs(m: int, k: int, n: int) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = _per_token_cast_to_fp8(x), _per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


# Copied from DeepGEMM's `test_core.py`
def _per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


# Copied from DeepGEMM's `test_core.py`
def _per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


# --------------------------------------- capture -------------------------------------

_ENABLE_CAPTURE = get_bool_env_var("SGLANG_ENABLE_WARMUP_DEEPGEMM_CAPTURE")


def capturer_on_kernel_executed(lhs, rhs, out):
    if not _ENABLE_CAPTURE:
        return
    _capturer.on_execution(lhs=lhs, rhs=rhs)


class _Capturer:
    def __init__(self):
        self._seen_infos = set()

    def on_execution(self, lhs, rhs):
        info = _compute_shape_from_args(lhs=lhs, rhs=rhs)

        if info in self._seen_infos:
            return

        self._seen_infos.add(info)
        logger.info(f"CAPTURER_NEW_INFO={json.dumps(info)}")


_capturer = _Capturer() if _ENABLE_CAPTURE else None


def _compute_shape_from_args(lhs, rhs):
    m, k = lhs[0].shape
    n, k_ = rhs[0].shape
    assert k == k_
    return dict(m=m, k=k, n=n)
