# TODO shall we put this file elsewhere?
import json
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Sequence, Callable

import torch
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.utils import get_bool_env_var

try:
    from deep_gemm import ceil_div, get_col_major_tma_aligned_tensor
    import deep_gemm
except ImportError:
    pass

logger = logging.getLogger(__name__)


# --------------------------------------- warmup -------------------------------------


def warmup(model):
    from sglang.srt.layers.quantization.fp8_kernel import enable_jit_deepgemm
    if not enable_jit_deepgemm:
        return

    sources = _compute_sources(model)
    if sources is None:
        return

    infos = _compute_infos_from_sources(sources)
    _warmup_by_infos(infos)


def _compute_sources(model) -> Optional[List[Dict[str, Any]]]:
    from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
    if isinstance(model, (DeepseekV2ForCausalLM, DeepseekV3ForCausalLM)):
        return _compute_sources_deepseek()
    return None


def _compute_sources_deepseek() -> List[Dict[str, Any]]:
    return [
        # 8xH200 + DeepSeek V3
        dict(n=4096, k=7168, m_min=1, m_max=8192),
        dict(n=7168, k=2048, m_min=1, m_max=8192),
        dict(n=7168, k=16384, m_min=1, m_max=8192),
        dict(n=24576, k=1536, m_min=1, m_max=8192),
        dict(n=4608, k=7168, m_min=1, m_max=8192),
        dict(n=7168, k=2304, m_min=1, m_max=8192),
        dict(n=32768, k=512, m_min=1, m_max=8192),
        dict(n=1536, k=7168, m_min=1, m_max=8192),
        dict(n=576, k=7168, m_min=1, m_max=8192),
    ]


def _compute_infos_from_sources(sources):
    return _deduplicate(_compute_infos_from_sources_raw(sources), key_fn=TODO)


def _compute_infos_from_sources_raw(sources):
    for source in sources:
        for m in range(source['m_min'], source['m_max'] + 1):
            yield dict(m=m, n=source['n'], k=source['k'])


def _deduplicate(items: Sequence[Any], key_fn: Callable):
    seen_keys = set()
    for item in items:
        item_key = key_fn(item)
        if item_key in seen_keys:
            continue
        seen_keys.add(item_key)
        yield item


def _warmup_by_infos(infos: Sequence[Dict[str, Any]]):
    for info in infos:
        print(f'hi warmup {info=}')
        x_fp8, y_fp8, out = _construct_gemm_inputs(m=info.m, k=info.k, n=info.n)
        # For simplicity, here we naively call the gemm
        # But alternatively, we may call `get_best_configs` or other configuration and deduplicate
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)


# Copied from DeepGEMM's `test_core.py` :: `construct`
def _construct_gemm_inputs(m: int, k: int, n: int):
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    x_fp8, y_fp8 = _per_token_cast_to_fp8(x), _per_block_cast_to_fp8(y)
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out


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

# TODO unify with fine_grained_benchmark, expert_distribution_recorder, etc
_dir_output = Path('/tmp/warmup_deepgemm_capture')
_ENABLE_CAPTURE = get_bool_env_var("SGLANG_ENABLE_WARMUP_DEEPGEMM_CAPTURE")


def capturer_on_kernel_executed(lhs, rhs):
    if not _ENABLE_CAPTURE:
        return
    _capturer.on_execution(lhs=lhs, rhs=rhs)


class _Capturer:
    def __init__(self):
        self._seen_infos = set()

    def on_execution(self, lhs, rhs):
        info = _compute_info_from_args(lhs=lhs, rhs=rhs)
        info_str = json.dumps(info)

        if info_str in self._seen_infos:
            return

        self._seen_infos.add(info_str)
        _write_output(info_str)


_capturer = _Capturer() if _ENABLE_CAPTURE else None


def _compute_info_from_args(lhs, rhs):
    m, k = lhs[0].shape
    n, k_ = rhs[0].shape
    assert k == k_
    return dict(m=m, k=k, n=n)


# TODO unify with fine_grained_benchmark, expert_distribution_recorder, etc
def _write_output(info):
    tp_rank = get_tensor_model_parallel_rank()
    path = Path(_dir_output) / f"TP{tp_rank}.jsonl"
    with path.open("a") as fp:
        fp.write(f"{info}\n")
    print(f"WARMUP_DEEPGEMM_CAPTURE={info}")


def clear_output():
    shutil.rmtree(_dir_output, ignore_errors=True)
    Path(_dir_output).mkdir(parents=True, exist_ok=True)


def read_output() -> List[Dict[str, Any]]:
    return [
        json.loads(row)
        for path in sorted(list(Path(_dir_output).glob("*.jsonl")))
        for row in path.read_text().split("\n")
        if row
    ]
