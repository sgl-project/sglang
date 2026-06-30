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

    def handle(self, action: str) -> Optional[Dict]:
        logger.info(f"[WeightChecker] handle action={action}")
        if action == "snapshot":
            return self._snapshot()
        elif action == "reset_tensors":
            return self._reset_tensors()
        elif action == "compare":
            return self._compare()
        elif action == "checksum":
            return self._compute_checksum()
        elif action == "lora_checksum":
            return self._compute_lora_checksum()
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

    def _compare(self):
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
        checksums = {
            name: _hash_tensor(tensor.data)
            for name, should_compare, tensor in _postprocess_tensors(
                dict(self._model_state()), skip_compare_names
            )
            if should_compare
        }

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

    def _compute_lora_checksum(self) -> Dict:
        """LoRA-only checksum: hash just the adapter weights (lora_cpu::/lora_gpu::),
        skipping the base model.

        Same pipeline as `checksum` (_postprocess_tensors + _hash_tensor, i.e. the GPU
        triton gpu_tensor_hash), but independent of the base-model hash. Use when the
        base hash is unavailable or IMAs (gpu_tensor_hash currently hits an
        illegal-memory-access on some GLM base tensors), or when only the LoRA
        train<->rollout sync needs verifying. Hashing the LoRA buffers themselves on
        GPU is fine (verified on GLM MoE/MLA), so this stays on the real serving path.
        """
        torch.cuda.synchronize()
        start = time.perf_counter()
        raw = self._lora_named_tensors()
        checksums = {
            name: _hash_resilient(tensor.data)
            for name, should_compare, tensor in _postprocess_tensors(raw, set())
            if should_compare
        }
        h = hashlib.sha256()
        for name in sorted(checksums):
            h.update(name.encode())
            h.update(checksums[name].encode())
        overall = h.hexdigest()
        torch.cuda.synchronize()
        logger.info(
            f"[WeightChecker] lora_checksum: {len(checksums)} LoRA tensors "
            f"in {time.perf_counter() - start:.3f}s"
        )
        return ChecksumInfo(
            checksums=checksums,
            per_gpu_checksum=overall,
            parallelism_info=self._parallelism_info(),
        ).model_dump()

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

    def _lora_named_tensors(self) -> Dict[str, torch.Tensor]:
        """LoRA adapter tensors held by THIS rollout engine, folded into `checksum`.

        LoRA weights are not in model.named_parameters(); they live in
        ``model_runner.lora_manager``. Returns two views, keyed by the STABLE
        ``lora_name`` (not the per-load uuid) so checksums line up across steps and
        ranks:
          ``lora_cpu::{name}::...`` — the delivered CPU adapter (lora_manager.loras)
          ``lora_gpu::{name}::...`` — the served GPU buffer slice the forward reads
                                      (memory_pool; only the resident adapter has one)

        Covers every module family — per-layer attn / MLP / MoE / MLA / DSA-indexer
        AND the separate embedding / lm_head / added-token stores. Fully defensive:
        returns ``{}`` if LoRA is disabled or internals are missing/renamed, and
        never raises (a serving engine must not crash on a checksum request).

        Note: GPU buffers are fused / zero-padded to max_lora_rank, so a
        ``lora_gpu::`` hash does not byte-equal the matching ``lora_cpu::`` one; the
        views are for cross-step (stale-buffer) and cross-rank (consistency) checks.
        """
        out: Dict[str, torch.Tensor] = {}
        lm = getattr(self._model_runner, "lora_manager", None)
        if lm is None:
            return out
        try:
            loras = getattr(lm, "loras", {}) or {}
            lora_refs = getattr(lm, "lora_refs", {}) or {}
            pool = getattr(lm, "memory_pool", None)
            for uid, adapter in loras.items():
                ref = lora_refs.get(uid)
                name = getattr(ref, "lora_name", None) or str(uid)[:8]

                # (1) delivered CPU adapter --------------------------------------
                for li, layer in enumerate(getattr(adapter, "layers", []) or []):
                    for wname, t in (getattr(layer, "weights", {}) or {}).items():
                        if isinstance(t, torch.Tensor):
                            out[f"lora_cpu::{name}::L{li}::{wname}"] = t
                for store, tag in (
                    (getattr(adapter, "embedding_layers", {}), "embed"),
                    (getattr(adapter, "added_tokens_embeddings", {}), "added_tokens"),
                ):
                    for wname, t in (store or {}).items():
                        if isinstance(t, torch.Tensor):
                            out[f"lora_cpu::{name}::{tag}::{wname}"] = t

                # (2) served GPU buffer slice (resident adapter only) ------------
                if pool is None:
                    continue
                bid = getattr(pool, "uid_to_buffer_id", {}).get(uid)
                if bid is None:
                    continue
                # per-layer A/B buffers: Dict[module -> List[per-layer tensor]]
                for ab, buf_dict in (
                    ("A", getattr(pool, "A_buffer", {})),
                    ("B", getattr(pool, "B_buffer", {})),
                ):
                    for module, per_layer in (buf_dict or {}).items():
                        for li, buf in enumerate(per_layer):
                            try:
                                out[f"lora_gpu::{name}::{module}::{ab}::L{li}"] = buf[bid]
                            except Exception:
                                pass
                # single-tensor embedding / lm_head / added-token buffers
                for buf_attr, tag in (
                    ("embedding_A_buffer", "embedA"),
                    ("embedding_B_buffer", "embedB"),
                    ("lm_head_A_buffer", "lmheadA"),
                    ("lm_head_B_buffer", "lmheadB"),
                    ("new_embeddings_buffer", "newemb"),
                ):
                    for module, buf in (getattr(pool, buf_attr, {}) or {}).items():
                        try:
                            out[f"lora_gpu::{name}::{module}::{tag}"] = buf[bid]
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"[WeightChecker] _lora_named_tensors skipped: {e}")
        return out


def _hash_tensor(t: torch.Tensor) -> str:
    return f"{tensor_hash(t):016x}"


def _hash_resilient(t: torch.Tensor) -> str:
    """Hash on the GPU triton path (tensor_hash -> gpu_tensor_hash for cuda tensors),
    i.e. the SAME path real serving uses; fall back to a CPU hashlib hash only if the
    GPU hash raises a RECOVERABLE exception.

    Caveat: a CUDA illegal-memory-access (IMA) is NOT recoverable — it corrupts the
    context, so the CPU fallback (and everything else) fails too; this wrapper cannot
    save an IMA, only clean exceptions. LoRA-buffer GPU hashing is verified not to IMA
    on GLM MoE/MLA, so for the LoRA checksum the GPU path always succeeds and the
    fallback never triggers — it is belt-and-suspenders for other models / edge cases.
    """
    try:
        return _hash_tensor(t)
    except Exception as e:
        logger.warning(
            "[WeightChecker] GPU hash failed (%s: %s); falling back to CPU hashlib",
            type(e).__name__, e,
        )
        return _hash_tensor(t.detach().to("cpu").contiguous())


def _check_tensors(
    expect_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
    actual_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
):
    from sglang.srt.debug_utils.dumper import get_tensor_info

    good_names = []
    error_messages = []
    info_messages = []

    for (expect_name, expect_should_compare, expect), (
        actual_name,
        actual_should_compare,
        actual,
    ) in zip(expect_tensors, actual_tensors, strict=True):
        assert expect_name == actual_name, f"{expect_name=} {actual_name=}"
        assert (
            expect_should_compare == actual_should_compare
        ), f"{expect_should_compare=} {actual_should_compare=}"
        name = expect_name
        should_compare = expect_should_compare

        expect = expect.cuda()
        actual = actual.cuda()

        if torch.all(expect == actual):
            good_names.append(name)
        else:
            abs_diff = (actual.float() - expect.float()).abs()
            msg = (
                f"name={name} "
                f"max_abs_err={abs_diff.max()} "
                f"mean_abs_err={abs_diff.mean()} "
                f"{get_tensor_info(expect)=} "
                f"{get_tensor_info(actual)=} "
            )
            (error_messages if should_compare else info_messages).append(msg)

    logger.info(f"[check_tensors] equal tensors: {good_names}")
    if len(info_messages) > 0:
        logger.info(f"[check_tensors] info: {info_messages}")
    if len(error_messages) > 0:
        raise Exception(f"check tensor equality failed:\n" + "\n".join(error_messages))


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point:
        return torch.rand(shape, device=device, dtype=torch.float32).to(dtype)

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    info = torch.iinfo(dtype)
    return torch.randint(
        low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype
    )


def _postprocess_tensors(
    raw: Dict[str, torch.Tensor],
    skip_compare_names: Set[str],
) -> Iterable[Tuple[str, bool, torch.Tensor]]:
    from sglang.srt.debug_utils.dumper import get_tensor_info

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
        w_q = raw[name]
        w_s = raw[name.replace("weight", "weight_scale_inv")]

        try:
            if w_s.dtype == torch.int32:
                # UE8M0 packed format (Blackwell DeepGEMM)
                w_s_for_dequant = inverse_transform_scale_ue8m0(w_s, mn=w_q.shape[-2])
            else:
                w_s_for_dequant = w_s

            w_dequant = block_quant_dequant(
                w_q,
                w_s_for_dequant,
                # TODO do not hardcode
                block_size=[128, 128],
                dtype=torch.bfloat16,
            )
            yield name, True, w_dequant
        except Exception as e:
            e.add_note(
                f"when handling {name=} {get_tensor_info(w_q)=} {get_tensor_info(w_s)=}"
            )
            raise

    for name in raw:
        should_compare = name not in skip_compare_names
        yield name, should_compare, raw[name]
