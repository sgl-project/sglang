# SPDX-License-Identifier: Apache-2.0
"""Low-latency BF16 skinny GEMM for DGX Spark (GB10, sm_121) decode shapes.

cuBLAS on sm_121 serves small-M BF16 GEMMs with SM80 WMMA kernels that reach
only 125-205 GB/s of the 273 GB/s available. This CUDA-core FMA kernel streams
the weight once and reaches 240-255 GB/s for 1 <= M <= 16.

Kernel and tuned plan table are ported from vLLM (Apache-2.0):
  vllm/model_executor/kernels/linear/cute_dsl/{skinny_gemm,_skinny_gemm}.py
  vllm/models/qwen4_exp/nvidia/low_latency_gemm.py  (QWEN4_EXP_SM121_GEMM_PLANS)

Only shapes with an explicitly measured plan run here; everything else keeps
cuBLAS. Output is BF16 with FP32 accumulation; the reduction order differs from
cuBLAS, so results are numerically equivalent but not bit-identical.

Env: SGLANG_SM121_SKINNY_GEMM=0 disables; SGLANG_SM121_SKINNY_GEMM_PDL=1 enables PDL.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_PREFETCH_TILES = 2


@dataclass(frozen=True)
class SkinnyGemmConfig:
    num_rows: int
    block_size: int
    outputs_per_block: int
    k_unroll: int = 1
    vector_width: int = 8
    static_k: Optional[int] = None


def row_stride_ok(a: torch.Tensor, config: SkinnyGemmConfig) -> bool:
    vector_bytes = config.vector_width * a.element_size()
    return (
        a.stride(1) == 1
        and a.stride(0) % config.vector_width == 0
        and a.data_ptr() % vector_bytes == 0
    )


# --------------------------------------------------------------------------- plans
# (N, K) -> {M: config}.  TP=2 local shapes of Qwen3.8-Flash-Next plus the TP=1
# rows vLLM measured on GB10; unused rows cost nothing (compiled only when the
# model actually has that weight shape).
SM121_GEMM_PLANS: Dict[Tuple[int, int], Dict[int, SkinnyGemmConfig]] = {
    # Shared-expert gate.
    (1, 2560): {
        1: SkinnyGemmConfig(1, 64, 1, k_unroll=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    # GDN fused B/A projection, TP=2.
    (48, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 64, 2, k_unroll=4, vector_width=4, static_k=2560),
    },
    # GDN fused B/A projection, TP=1.
    (96, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 4, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 1),
    },
    # Router.
    (512, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, k_unroll=2, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 2, static_k=2560),
        3: SkinnyGemmConfig(3, 64, 1, k_unroll=4, vector_width=4, static_k=2560),
    },
    # Shared-expert fused gate/up, TP=2.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 256, 1, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 64, 1, k_unroll=4, vector_width=4, static_k=2560),
    },
    # Shared-expert fused gate/up, TP=1.
    (1280, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 1, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, vector_width=2, static_k=2560),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, static_k=2560),
    },
    # Shared-expert down projection, TP=1.
    (2560, 640): {
        1: SkinnyGemmConfig(1, 64, 1, k_unroll=4, vector_width=2, static_k=640),
        2: SkinnyGemmConfig(2, 64, 1, vector_width=2, static_k=640),
        4: SkinnyGemmConfig(4, 64, 1, vector_width=2, static_k=640),
        8: SkinnyGemmConfig(8, 32, 1, k_unroll=4, vector_width=4, static_k=640),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, vector_width=4, static_k=640),
    },
    # GDN and QSA output projections, TP=2.
    (2560, 3072): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4, static_k=3072),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, static_k=3072),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2, static_k=3072),
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=2, vector_width=4, static_k=3072),
    },
    # GDN and QSA output projections, TP=1.
    (2560, 6144): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=6144),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, vector_width=4, static_k=6144),
        4: SkinnyGemmConfig(4, 64, 1, k_unroll=4, vector_width=4, static_k=6144),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=2, static_k=6144),
        16: SkinnyGemmConfig(16, 32, 1, static_k=6144),
    },
    # QSA fused QKV/gate projection, TP=2.
    (6656, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    # GDN fused QKVZ projection, TP=2.
    (8192, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    # HC up projection.
    (10240, 320): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=2, vector_width=2, static_k=320),
    },
    # QSA fused QKV/gate + replicated indexer Q/K, TP=1.
    (13952, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 1, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, vector_width=2, static_k=2560),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    # GDN fused QKVZ projection, TP=1.
    (16384, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=2, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, vector_width=2, static_k=2560),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, vector_width=2, static_k=2560),
    },
    # Per-layer BF16 projection seen at verify (SGLang TP=2 fusion), measured 2026-10-08.
    (10240, 2560): {
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    (2560, 2560): {
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        3: SkinnyGemmConfig(3, 128, 2, k_unroll=4, vector_width=4),
    },
    # GDN fused qkvz+ba input projection (SGLang packs both into one GEMM), TP=2.
    (8240, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    # HC inject projection.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=4, static_k=10240),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=2, vector_width=4, static_k=10240),
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=2, vector_width=4, static_k=10240),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=2, vector_width=4, static_k=10240),
    },
    # HC mix down projection.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 64, 1, k_unroll=4, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    # Hot-vocab NEXTN draft head (65,536-token map), TP=2: reuse the TP=2 LM head plan.
    (32768, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 32, 1, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 1, vector_width=4, static_k=2560),
    },
    # LM head, TP=2.
    (124160, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        3: SkinnyGemmConfig(3, 64, 1, k_unroll=4, vector_width=4, static_k=2560),
    },
    # LM head, TP=1.
    (248320, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=2, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
}


# --------------------------------------------------------------------------- kernel
def _load_kernel_class():
    """Import the CuTe kernel (module-level cutlass imports; works in and out of the package)."""
    try:
        from ._sm121_skinny_kernel import CuteSkinnyGemm  # type: ignore

        return CuteSkinnyGemm
    except ImportError:
        import importlib.util
        import sys

        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_sm121_skinny_kernel.py"
        )
        spec = importlib.util.spec_from_file_location("_sm121_skinny_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_sm121_skinny_kernel"] = mod
        spec.loader.exec_module(mod)
        return mod.CuteSkinnyGemm


# --------------------------------------------------------------------------- runtime
class ShapeDynamicSkinnyGemm:
    def __init__(self) -> None:
        self._compiled: Dict[Tuple[torch.dtype, SkinnyGemmConfig, bool], Any] = {}
        self._kernel_cls = None
        self._available: Optional[bool] = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import cutlass  # noqa: F401
                import cutlass.cute  # noqa: F401
                from quack.compile_utils import make_fake_tensor  # noqa: F401

                self._available = True
            except ImportError as e:
                self._available = False
                logger.info("sm121 skinny GEMM disabled: %s", e)
        return self._available

    @staticmethod
    def _cutlass_dtype(dtype: torch.dtype):
        from cutlass import BFloat16, Float16

        return BFloat16 if dtype == torch.bfloat16 else Float16

    @staticmethod
    def _stream():
        from cuda.bindings.driver import CUstream

        return CUstream(torch.cuda.current_stream().cuda_stream)

    def _compile(
        self, dtype: torch.dtype, config: SkinnyGemmConfig, has_residual: bool
    ):
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor

        if self._kernel_cls is None:
            self._kernel_cls = _load_kernel_class()
        element_type = self._cutlass_dtype(dtype)
        n = cute.sym_int(divisibility=config.outputs_per_block)
        k = (
            config.static_k
            if config.static_k is not None
            else cute.sym_int(divisibility=config.block_size * config.vector_width)
        )
        a = make_fake_tensor(
            element_type, (config.num_rows, k), divisibility=config.vector_width
        )
        b = make_fake_tensor(element_type, (n, k), divisibility=config.vector_width)
        c = make_fake_tensor(element_type, (config.num_rows, n), divisibility=1)
        residual = make_fake_tensor(element_type, (config.num_rows, n), divisibility=1)
        kernel = self._kernel_cls(
            element_type=element_type,
            num_rows=config.num_rows,
            block_size=config.block_size,
            outputs_per_block=config.outputs_per_block,
            vector_width=config.vector_width,
            k_unroll=config.k_unroll,
            has_residual=has_residual,
            use_pdl=envs.SGLANG_SM121_SKINNY_GEMM_PDL.get(),
            static_k=config.static_k,
        )
        compiled = cute.compile(
            kernel,
            a,
            b,
            residual,
            c,
            self._stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )
        self._compiled[(dtype, config, has_residual)] = compiled
        return compiled

    def precompile(
        self, dtype: torch.dtype, config: SkinnyGemmConfig, has_residual: bool = False
    ) -> None:
        key = (dtype, config, has_residual)
        with self._lock:
            if key in self._compiled:
                return
            fn = self._compile(dtype, config, has_residual)
        # One real launch so the module is resident before any CUDA graph capture.
        k = config.static_k or (config.block_size * config.vector_width * 2)
        n = config.outputs_per_block * 8
        dev = torch.cuda.current_device()
        a = torch.zeros(config.num_rows, k, dtype=dtype, device=dev)
        b = torch.zeros(n, k, dtype=dtype, device=dev)
        c = torch.empty(config.num_rows, n, dtype=dtype, device=dev)
        fn(a, b, c, c, self._stream())
        torch.cuda.synchronize()

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        config: SkinnyGemmConfig,
        residual: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        has_residual = residual is not None
        key = (a.dtype, config, has_residual)
        fn = self._compiled.get(key)
        if fn is None:
            with self._lock:
                fn = self._compiled.get(key) or self._compile(
                    a.dtype, config, has_residual
                )
        if (
            out is not None
            and out.shape == (a.shape[0], b.shape[0])
            and out.dtype == a.dtype
            and out.is_contiguous()
        ):
            output = out
        else:
            output = torch.empty(
                (a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device
            )
        fn(a, b, output if residual is None else residual, output, self._stream())
        return output


skinny_gemm = ShapeDynamicSkinnyGemm()

_is_sm121: Optional[bool] = None
_misses: Counter = Counter()
_miss_logged: Set[Tuple[int, int, int]] = set()


def sm121_skinny_enabled() -> bool:
    global _is_sm121
    if not envs.SGLANG_SM121_SKINNY_GEMM.get():
        return False
    if _is_sm121 is None:
        _is_sm121 = (
            torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)
        )
    return _is_sm121


def _runtime_ok(
    x: torch.Tensor, weight: torch.Tensor, config: SkinnyGemmConfig
) -> bool:
    return (
        x.dim() == 2
        and row_stride_ok(x, config)
        and weight.dim() == 2
        and weight.stride() == (weight.shape[1], 1)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.shape[1] == weight.shape[1]
        and (config.static_k is None or x.shape[1] == config.static_k)
        and weight.shape[0] % config.outputs_per_block == 0
        and x.shape[1] % (config.block_size * config.vector_width) == 0
    )


def _note_miss(n: int, k: int, m: int) -> None:
    key = (n, k, m)
    _misses[key] += 1
    if key not in _miss_logged and _misses[key] >= 4:
        _miss_logged.add(key)
        logger.info(
            "sm121 skinny GEMM: no plan for (N=%d, K=%d, M=%d); cuBLAS keeps it",
            n,
            k,
            m,
        )


def maybe_skinny_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Run the skinny kernel if (N, K, M) has a measured plan; else None (caller uses cuBLAS)."""
    if (
        not sm121_skinny_enabled()
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
    ):
        return None
    if torch.compiler.is_compiling():
        return None
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    if is_batch_invariant_mode_enabled():
        # Different M picks a different plan (and reduction order) than cuBLAS.
        return None
    x2 = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    m = x2.shape[0]
    if m < 1 or m > 16:
        return None
    n, k = weight.shape[0], weight.shape[1]
    plan = SM121_GEMM_PLANS.get((n, k))
    config = None if plan is None else plan.get(m)
    if config is None:
        _note_miss(n, k, m)
        return None
    if not _runtime_ok(x2, weight, config) or not skinny_gemm.is_available():
        return None
    if residual is not None:
        residual = residual.reshape(m, n)
        if not residual.is_contiguous() or residual.dtype != x.dtype:
            residual = residual.contiguous().to(x.dtype)
    res = skinny_gemm(x2, weight, config, residual, out if x.dim() == 2 else None)
    return res if x.dim() == 2 else res.reshape(*x.shape[:-1], n)


_warmed: Set[Tuple[int, int]] = set()


def warmup_for_model(model: torch.nn.Module) -> None:
    """Compile and touch every plan whose (N, K) matches a BF16 2-D weight in the model.

    Call before CUDA graph capture; the graph then only records plain launches.
    """
    if not sm121_skinny_enabled() or not skinny_gemm.is_available():
        return
    shapes: Set[Tuple[int, int]] = set()
    for mod in model.modules():
        # Plain weights plus the packed GDN in_proj buffer (qwen3_5.finalize_fused_in_proj),
        # which is a module attribute rather than a parameter.
        for w in (
            getattr(mod, "weight", None),
            getattr(mod, "_fused_in_proj_weight", None),
        ):
            if (
                isinstance(w, torch.Tensor)
                and w.dim() == 2
                and w.dtype == torch.bfloat16
                and w.is_cuda
            ):
                shapes.add((w.shape[0], w.shape[1]))
    todo = [s for s in shapes if s in SM121_GEMM_PLANS and s not in _warmed]
    if not todo:
        return
    import time

    t0 = time.time()
    n_cfg = 0
    for s in sorted(todo):
        for cfg in SM121_GEMM_PLANS[s].values():
            skinny_gemm.precompile(torch.bfloat16, cfg)
            n_cfg += 1
        _warmed.add(s)
    logger.info(
        "sm121 skinny GEMM: compiled %d configs for %d shapes %s in %.1fs",
        n_cfg,
        len(todo),
        sorted(todo),
        time.time() - t0,
    )
