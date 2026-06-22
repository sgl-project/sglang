import hashlib
import logging
import time
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from sglang.srt.managers.mm_utils import tensor_hash

logger = logging.getLogger(__name__)


class _StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParallelismInfo(_StrictBaseModel):
    tp_rank: int
    tp_size: int
    dp_rank: int
    dp_size: int
    pp_rank: int
    pp_size: int
    rank: int
    size: int


class ChecksumInfo(_StrictBaseModel):
    checksums: Dict[str, str]
    per_gpu_checksum: str
    parallelism_info: ParallelismInfo


_NON_PERSISTENT_BUFFER_PATTERNS = (
    "cos_sin_cache",
    "inv_freq",
    "freqs_cis",
    "_weight_fp32",
)


def _is_non_persistent_buffer_name(name: str) -> bool:
    return any(pat in name for pat in _NON_PERSISTENT_BUFFER_PATTERNS)


class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner
        self._snapshot_tensors = None

    def handle(self, action: str, allow_quant_error: bool = False) -> Optional[Dict]:
        logger.info(
            f"[WeightChecker] handle action={action} allow_quant_error={allow_quant_error}"
        )
        if action == "snapshot":
            return self._snapshot()
        elif action == "reset_tensors":
            return self._reset_tensors()
        elif action == "compare":
            return self._compare(allow_quant_error=allow_quant_error)
        elif action == "checksum":
            return self._compute_checksum()
        else:
            raise Exception(f"Unsupported {action=}")

    def _snapshot(self):
        named_tensors = [
            (name, param.data.detach().cpu()) for name, param in self._model_state()
        ]
        self._snapshot_tensors = dict(named_tensors)
        assert len(self._snapshot_tensors) == len(
            named_tensors
        ), f"should not have duplicated tensor name"

    def _reset_tensors(self):
        for name, param in self._model_state():
            if _is_non_persistent_buffer_name(name):
                continue
            param.copy_(_random_like(param))

    def _compare(self, allow_quant_error: bool = False):
        assert self._snapshot_tensors is not None

        skip_compare_names = {
            name
            for name, param in self._model_state()
            if getattr(param, "_skip_weight_check", False)
        }
        _check_tensors(
            expect_tensors=_postprocess_tensors(
                self._snapshot_tensors, skip_compare_names
            ),
            actual_tensors=_postprocess_tensors(
                dict(self._model_state()), skip_compare_names
            ),
            allow_quant_error=allow_quant_error,
        )

    def _compute_checksum(self) -> Dict:
        torch.cuda.synchronize()
        start = time.perf_counter()

        skip_compare_names = {
            name
            for name, param in self._model_state()
            if getattr(param, "_skip_weight_check", False)
        }

        # Reuse the snapshot/compare postprocess pipeline so fp8 weights are
        # dequantized to bf16 before hashing — two (qweight, scale) pairs that
        # produce the same bf16 must produce the same checksum.
        checksums = {}
        for name, should_compare, entry in _postprocess_tensors(
            dict(self._model_state()), skip_compare_names
        ):
            if not should_compare:
                continue
            if entry[0] == "quant":
                _, w_q, w_s = entry
                w_s = _normalize_scale(w_q, w_s)
                tensor = block_quant_dequant(
                    w_q, w_s, block_size=_block_size_of(w_q, w_s), dtype=torch.bfloat16
                )
            else:
                tensor = entry[1]
            checksums[name] = _hash_tensor(tensor.data)

        h = hashlib.sha256()
        for name in sorted(checksums):
            h.update(name.encode())
            h.update(checksums[name].encode())
        overall = h.hexdigest()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        logger.info(
            f"[WeightChecker] checksum computed for {len(checksums)} tensors in {elapsed:.3f}s"
        )

        info = ChecksumInfo(
            checksums=checksums,
            per_gpu_checksum=overall,
            parallelism_info=self._parallelism_info(),
        )
        return info.model_dump()

    def _parallelism_info(self) -> ParallelismInfo:
        mr = self._model_runner
        return ParallelismInfo(
            tp_rank=mr.tp_rank,
            tp_size=mr.tp_size,
            dp_rank=mr.dp_rank if mr.dp_rank is not None else 0,
            dp_size=mr.dp_size,
            pp_rank=mr.pp_rank,
            pp_size=mr.pp_size,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            size=dist.get_world_size() if dist.is_initialized() else 1,
        )

    def _model_state(self):
        yield from self._model_runner.model.named_parameters()
        yield from self._model_runner.model.named_buffers()


def _hash_tensor(t: torch.Tensor) -> str:
    return f"{tensor_hash(t):016x}"


def _check_tensors(
    expect_tensors: Iterable[Tuple[str, bool, Tuple]],
    actual_tensors: Iterable[Tuple[str, bool, Tuple]],
    allow_quant_error: bool = False,
):
    good_names = []
    error_messages = []
    info_messages = []

    for (expect_name, expect_should_compare, expect_entry), (
        actual_name,
        actual_should_compare,
        actual_entry,
    ) in zip(expect_tensors, actual_tensors, strict=True):
        assert expect_name == actual_name, f"{expect_name=} {actual_name=}"
        assert (
            expect_should_compare == actual_should_compare
        ), f"{expect_should_compare=} {actual_should_compare=}"
        assert expect_entry[0] == actual_entry[0]
        name = expect_name
        should_compare = expect_should_compare

        if expect_entry[0] == "quant":
            _, expect_q, expect_s = expect_entry
            _, actual_q, actual_s = actual_entry
            try:
                equal, max_abs_err, mean_abs_err, num_exceed = _compare_quant_pair(
                    expect_q, expect_s, actual_q, actual_s
                )
            except Exception as e:
                e.add_note(
                    f"when handling {name=} "
                    f"expect: shape={tuple(expect_q.shape)} dtype={expect_q.dtype} "
                    f"actual: shape={tuple(actual_q.shape)} dtype={actual_q.dtype}"
                )
                raise
            if equal:
                good_names.append(name)
                continue
            msg = (
                f"name={name} "
                f"max_abs_err={max_abs_err} "
                f"mean_abs_err={mean_abs_err} "
                f"num_exceed_quant_ulp_tolerance={num_exceed} "
                f"expect_quant: shape={tuple(expect_q.shape)} dtype={expect_q.dtype} "
                f"actual_quant: shape={tuple(actual_q.shape)} dtype={actual_q.dtype} "
            )
            if not should_compare:
                info_messages.append(msg)
            elif allow_quant_error and num_exceed == 0:
                # Two faithful quantizations may differ by up to 1 ULP each.
                info_messages.append(msg + "(within quantization ULP tolerance)")
            else:
                error_messages.append(msg)
            continue

        expect = expect_entry[1]
        actual = actual_entry[1]
        equal, max_abs_err, mean_abs_err = _compare_raw_pair(
            expect, actual, compute_stats=should_compare
        )
        if equal:
            good_names.append(name)
        elif not should_compare:
            info_messages.append(
                f"name={name} differs (not compared) "
                f"shape={tuple(actual.shape)} dtype={actual.dtype} "
            )
        else:
            error_messages.append(
                f"name={name} "
                f"max_abs_err={max_abs_err} "
                f"mean_abs_err={mean_abs_err} "
                f"shape={tuple(actual.shape)} "
                f"expect_dtype={expect.dtype} actual_dtype={actual.dtype} "
                f"expect_head={expect.reshape(-1)[:5].float().tolist()} "
                f"actual_head={actual.reshape(-1)[:5].float().tolist()} "
            )

    logger.info(f"[check_tensors] equal tensors: {good_names}")
    if len(info_messages) > 0:
        logger.info(f"[check_tensors] info: {info_messages}")
    if len(error_messages) > 0:
        raise Exception(f"check tensor equality failed:\n" + "\n".join(error_messages))


# Bound GPU memory: a full-size fp32 intermediate won't fit beside the KV-cache pool.
_CHUNK_NUMEL = 64 * 1024 * 1024


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point:
        out = torch.empty(shape, device=device, dtype=dtype)
        for chunk in out.view(-1).split(_CHUNK_NUMEL):
            chunk.copy_(
                torch.rand(chunk.shape, device=device, dtype=torch.float32).to(dtype)
            )
        return out

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    info = torch.iinfo(dtype)
    return torch.randint(
        low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype
    )


def _quant_ulp(w_q: torch.Tensor) -> torch.Tensor:
    """Per-element ULP of w_q in its own dtype."""
    finfo = torch.finfo(w_q.dtype)
    x = w_q.to(torch.float32).abs()
    # frexp: x = m * 2^e, m in [0.5, 1), so 2^(e-1) is x's binade base.
    _, exponent = torch.frexp(x)
    binade = torch.exp2((exponent - 1).to(torch.float32))
    # Zeros and subnormals share the spacing of the smallest normal binade.
    binade = binade.masked_fill(x < finfo.smallest_normal, finfo.smallest_normal)
    return binade * finfo.eps


def _normalize_scale(w_q: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
    if w_s.dtype == torch.int32:
        # UE8M0 packed format (Blackwell DeepGEMM)
        w_s = inverse_transform_scale_ue8m0(w_s, mn=w_q.shape[-2])
    return w_s.to(torch.float32)


def _block_size_of(w_q: torch.Tensor, w_s: torch.Tensor) -> list:
    # Square blocks. The row dim may have a partial last block (so n//s_n is wrong),
    # but the K dim is block-aligned, so k//s_k recovers the true block size exactly.
    k, s_k = w_q.shape[-1], w_s.shape[-1]
    assert k % s_k == 0, f"cannot infer block size from {w_q.shape=} {w_s.shape=}"
    block = k // s_k
    return [block, block]


def _iter_quant_chunks(w_q: torch.Tensor, w_s: torch.Tensor, block_n: int):
    """Yields block-row-aligned (q_slice, s_slice) pairs of bounded size."""
    q3 = w_q.reshape(-1, *w_q.shape[-2:])
    s3 = w_s.reshape(-1, *w_s.shape[-2:])
    n, k = q3.shape[-2:]
    rows = max(block_n, _CHUNK_NUMEL // k // block_n * block_n)
    for b in range(q3.shape[0]):
        for r0 in range(0, n, rows):
            r1 = min(r0 + rows, n)
            yield q3[b, r0:r1], s3[b, r0 // block_n : -(-r1 // block_n)]


def _compare_quant_pair(
    expect_q: torch.Tensor,
    expect_s: torch.Tensor,
    actual_q: torch.Tensor,
    actual_s: torch.Tensor,
) -> Tuple[bool, float, float, int]:
    """Chunked compare of two block-quant tensors in dequant space. Returns
    (equal, max_abs_err, mean_abs_err, num_exceed); num_exceed counts elements
    off by >1 ULP per side (NaN counts as exceeding)."""
    assert expect_q.shape == actual_q.shape, f"{expect_q.shape=} {actual_q.shape=}"
    expect_s = _normalize_scale(expect_q, expect_s)
    actual_s = _normalize_scale(actual_q, actual_s)
    block_n = _block_size_of(actual_q, actual_s)[0]

    equal = True
    max_abs_err = torch.zeros((), dtype=torch.float32)
    sum_abs_err = 0.0
    num_exceed = 0
    for (e_q, e_s), (a_q, a_s) in zip(
        _iter_quant_chunks(expect_q, expect_s, block_n),
        _iter_quant_chunks(actual_q, actual_s, block_n),
        strict=True,
    ):
        e_q, e_s = e_q.cuda(), e_s.cuda()
        a_q, a_s = a_q.cuda(), a_s.cuda()
        if e_q.dtype == a_q.dtype and torch.equal(e_q, a_q) and torch.equal(e_s, a_s):
            continue
        block_size = _block_size_of(e_q, e_s)
        e_dequant = block_quant_dequant(e_q, e_s, block_size, dtype=torch.bfloat16)
        a_dequant = block_quant_dequant(a_q, a_s, block_size, dtype=torch.bfloat16)
        abs_diff = (a_dequant.float() - e_dequant.float()).abs()
        if torch.all(abs_diff == 0):
            continue
        equal = False
        tolerance = block_quant_dequant(
            _quant_ulp(e_q), e_s, block_size, dtype=torch.float32
        ) + block_quant_dequant(_quant_ulp(a_q), a_s, block_size, dtype=torch.float32)
        # torch.maximum propagates NaN, unlike builtin max().
        max_abs_err = torch.maximum(max_abs_err, abs_diff.max().cpu())
        sum_abs_err += abs_diff.sum().item()
        # `~(diff <= tol)` instead of `diff > tol` so NaN counts as exceeding.
        num_exceed += int((~(abs_diff <= tolerance)).sum())
    return equal, max_abs_err.item(), sum_abs_err / expect_q.numel(), num_exceed


def _compare_raw_pair(
    expect: torch.Tensor, actual: torch.Tensor, compute_stats: bool
) -> Tuple[bool, float, float]:
    """Chunked exact-compare of two raw tensors, optionally accumulating
    abs-diff stats. Returns (equal, max_abs_err, mean_abs_err)."""
    assert expect.shape == actual.shape, f"{expect.shape=} {actual.shape=}"
    expect_flat = expect.reshape(-1)
    actual_flat = actual.reshape(-1)

    equal = True
    max_abs_err = torch.zeros((), dtype=torch.float32)
    sum_abs_err = 0.0
    for start in range(0, expect_flat.numel(), _CHUNK_NUMEL):
        e = expect_flat[start : start + _CHUNK_NUMEL].cuda()
        a = actual_flat[start : start + _CHUNK_NUMEL].cuda()
        if torch.all(e == a):
            continue
        equal = False
        if not compute_stats:
            break
        abs_diff = (a.float() - e.float()).abs()
        # torch.maximum propagates NaN, unlike builtin max().
        max_abs_err = torch.maximum(max_abs_err, abs_diff.max().cpu())
        sum_abs_err += abs_diff.sum().item()
    return equal, max_abs_err.item(), sum_abs_err / max(expect_flat.numel(), 1)


def _postprocess_tensors(
    raw: Dict[str, torch.Tensor],
    skip_compare_names: Set[str],
) -> Iterable[Tuple[str, bool, Tuple]]:
    """Yields (name, should_compare, entry); entry is ("quant", w_q, w_s) for
    block-quant pairs (dequantized lazily) or ("raw", tensor)."""
    skip_compare_names = set(skip_compare_names)

    # Skip non-persistent buffers (registered with persistent=False; recomputed
    # after weight load and not part of the synced payload).
    for name in raw:
        if _is_non_persistent_buffer_name(name):
            skip_compare_names.add(name)
            logger.info(f"[check_tensors] Skipping non-persistent buffer: {name}")

    # dequant fp8
    quant_names = [
        name
        for name in raw
        # Match: `something.weight`, `something.experts.w2_weight`
        if name.endswith("weight") and name.replace("weight", "weight_scale_inv") in raw
    ]
    quant_scale_names = [
        name.replace("weight", "weight_scale_inv") for name in quant_names
    ]
    skip_compare_names.update(quant_names)
    skip_compare_names.update(quant_scale_names)
    for name in quant_names:
        yield name, True, (
            "quant",
            raw[name],
            raw[name.replace("weight", "weight_scale_inv")],
        )

    for name in raw:
        should_compare = name not in skip_compare_names
        yield name, should_compare, ("raw", raw[name])
