"""How the exported MLX region fills ``cuda_graph_config`` on MPS.

Pure Python on purpose: ``ServerArgs`` resolution imports this, so it must
not pull in torch or MLX. The region is registered in ``graph_runners`` like
the CPU/NPU/XPU runners and reads the same ``cuda_graph_config`` they do
(``--cuda-graph-bs-{decode,prefill}`` / ``--cuda-graph-max-bs-{decode,prefill}``);
what differs on MPS is the *defaults*. A region export pays ~2 s per shape
(torch.export, MLX lowering, ``mx.compile``) where a CUDA-graph capture pays
milliseconds, so the default ladders here are far smaller than the CUDA
generators' -- same flags, MPS-sized defaults. Every ladder ends at its cap,
so a batch at exactly the cap is served rather than silently falling back to
eager.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
)

DEFAULT_MAX_DECODE_BATCH_SIZE = 16
DEFAULT_MAX_PREFILL_TOKENS = 2048
_SMALLEST_PREFILL_BUCKET = 128


def mlx_region_decode_batch_sizes(max_bs: int) -> list[int]:
    """Decode buckets up to and including ``max_bs``.

    The same shape as ``ServerArgs._generate_decode_cuda_graph_batch_sizes``
    in the non-speculative case (the region rejects speculative batches):
    dense at small sizes, coarsening as padding waste shrinks relative to
    the batch.
    """
    _require_positive("cuda_graph_config[decode].max_bs", max_bs)
    ladder = (
        [1, 2, 4, 8, 12]
        + list(range(16, 257, 8))
        + list(range(272, 512, 16))
        + list(range(512, max_bs + 1, 32))
    )
    buckets = [bs for bs in ladder if bs <= max_bs]
    if not buckets or buckets[-1] != max_bs:
        buckets.append(max_bs)
    return buckets


def mlx_region_prefill_token_buckets(max_tokens: int) -> list[int]:
    """Prefill token buckets up to and including ``max_tokens``.

    A x1.5 / x1.33 ladder from 128 (128, 192, 256, 384, ...): coarse enough
    that the export count stays small, fine enough that padding waste stays
    well below the region's kernel win.
    """
    _require_positive("cuda_graph_config[prefill].max_bs", max_tokens)
    buckets: list[int] = []
    size = _SMALLEST_PREFILL_BUCKET
    while size <= max_tokens:
        buckets.append(size)
        if size + size // 2 <= max_tokens:
            buckets.append(size + size // 2)
        size *= 2
    if not buckets or buckets[-1] != max_tokens:
        buckets.append(max_tokens)
    return buckets


def fill_mps_region_ladders(
    config: CudaGraphConfig, *, max_total_tokens: Optional[int] = None
) -> None:
    """Fill unset caps and ladders with MPS-sized defaults, in place.

    Mirrors what ``ServerArgs._handle_gpu_memory_settings`` does for CUDA: an
    explicit ``bs`` wins and pins ``max_bs`` to its maximum; an explicit
    ``max_bs`` gets the default ladder generated up to it; nothing set means
    the MPS defaults. Like CUDA's prefill graphs, ``prefill.bs`` carries token
    counts, and a default prefill cap never exceeds ``max_total_tokens``.
    """
    decode = config.decode
    if decode.bs is not None:
        decode.max_bs = max(decode.bs)
    else:
        if decode.max_bs is None:
            decode.max_bs = DEFAULT_MAX_DECODE_BATCH_SIZE
        decode.bs = mlx_region_decode_batch_sizes(decode.max_bs)

    prefill = config.prefill
    if prefill.bs is not None:
        prefill.max_bs = max(prefill.bs)
    else:
        if prefill.max_bs is None:
            prefill.max_bs = DEFAULT_MAX_PREFILL_TOKENS
            if max_total_tokens is not None:
                prefill.max_bs = min(prefill.max_bs, max_total_tokens)
        prefill.bs = mlx_region_prefill_token_buckets(prefill.max_bs)


def apply_mps_region_backends(config: CudaGraphConfig, locked: set) -> None:
    """Keep ``cuda_graph_config`` live on MPS instead of auto-disabling it.

    The region is one whole-forward executor per bucket for both phases,
    which is exactly ``Backend.FULL``; a phase the user did not pin
    (``locked`` holds the ``(phase, key)`` pairs that came from explicit
    flags) becomes FULL. A pinned ``disabled`` is respected per phase; any
    other pinned backend has no MPS implementation and is rejected.
    """
    for phase in Phase.ALL:
        phase_config = config[phase]
        if (phase, "backend") not in locked:
            phase_config.backend = Backend.FULL
        elif phase_config.backend not in (Backend.FULL, Backend.DISABLED):
            raise ValueError(
                f"cuda_graph_config[{phase}].backend={phase_config.backend!r} is "
                "not available on MPS; the MLX region serves 'full', or use "
                "'disabled'."
            )


def _require_positive(name: str, value: object) -> None:
    if value is None or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, found {value!r}")
