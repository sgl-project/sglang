"""One-shot diagnostic comparing carried and same-input DSA TopK.

This module is deliberately outside the normal serving path.  The probe is
enabled only by an exact request id, runs only in eager CUDA execution, restores
the current token's index-K bytes before attention consumes the carried TopK,
and performs its sole device-to-host read at request completion.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.dsa.index_buf_accessor import (
    restore_k_and_s_by_loc,
    snapshot_k_and_s_by_loc,
)
from sglang.srt.layers.attention.dsa.utils import should_use_dsa_fused_topk
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_memory, get_parallel
from sglang.srt.state_capturer.indexer_topk import suspend_indexer_topk_capture
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_METRIC_NAMES = (
    "exact_rows",
    "total_rows",
    "same_position",
    "carried_valid",
    "self_valid",
    "valid_set_intersection",
    "carried_checksum",
    "self_checksum",
)


@dataclass(frozen=True)
class _BucketKey:
    step: int
    provenance: str
    layer_id: int


class DSATopKShadowProbe:
    """Request-scoped, fail-closed owner for the shadow comparison."""

    def __init__(
        self,
        expected_rid: Optional[str],
        *,
        static_rejection: Optional[str] = None,
    ) -> None:
        self.expected_rid = expected_rid or None
        self._rejection = static_rejection
        self._sealed = False
        self._seen = False
        self._last_step: Optional[int] = None
        self._buckets: dict[_BucketKey, torch.Tensor] = {}

    @classmethod
    def from_runtime(
        cls, expected_rid: Optional[str], *, seed_enabled: bool
    ) -> DSATopKShadowProbe:
        rejection = None
        if expected_rid:
            parallel = get_parallel()
            memory = get_memory()
            if not is_cuda():
                rejection = "requires CUDA"
            elif not seed_enabled:
                rejection = "requires draft-extend DSA seed"
            elif should_use_dsa_fused_topk(seed_enabled):
                rejection = "requires request-relative unfused DSA TopK"
            elif parallel.attn_cp_size != 1:
                rejection = "attn_cp_size must be 1"
            elif parallel.dcp_enabled:
                rejection = "DCP is unsupported"
            elif parallel.enable_dsa_cache_layer_split:
                rejection = "DSA cache layer split is unsupported"
            elif memory.enable_hisparse:
                rejection = "HiSparse is unsupported"
        probe = cls(expected_rid, static_rejection=rejection)
        if expected_rid:
            if rejection is None:
                logger.warning(
                    "DSA TopK shadow diagnostic armed for exact rid=%s",
                    expected_rid,
                )
            else:
                logger.error(
                    "DSA TopK shadow diagnostic rejected for rid=%s: %s",
                    expected_rid,
                    rejection,
                )
        return probe

    @property
    def can_probe(self) -> bool:
        return (
            self.expected_rid is not None
            and self._rejection is None
            and not self._sealed
        )

    def matches_rids(self, rids: Optional[list[str]]) -> bool:
        return bool(self.can_probe and rids and self.expected_rid in rids)

    def matches_schedule_batch(self, batch) -> bool:
        return self.matches_rids([req.rid for req in batch.reqs])

    def _reject(self, reason: str) -> None:
        if self._rejection is None:
            self._rejection = reason
            logger.error(
                "DSA TopK shadow diagnostic failed closed for rid=%s: %s",
                self.expected_rid,
                reason,
            )

    @contextlib.contextmanager
    def forward_scope(
        self,
        forward_batch: ForwardBatch,
        *,
        step: int,
        using_cuda_graph: bool,
    ):
        """Attach the observer only around one real eager inner forward."""
        if not self.matches_rids(forward_batch.rids):
            yield
            return

        self._seen = True
        if not self.can_probe:
            yield
            return
        if forward_batch.rids != [self.expected_rid]:
            self._reject("target rid must be the sole request in the batch")
            yield
            return
        if using_cuda_graph or get_is_capture_mode():
            self._reject("CUDA graph execution is unsupported")
            yield
            return
        if step not in (0, 1):
            self._reject(f"expected exactly two inner forwards, got step={step}")
            yield
            return
        expected_step = 0 if self._last_step in (None, 1) else 1
        if step != expected_step:
            self._reject(
                f"inner-forward order mismatch: expected step={expected_step}, got {step}"
            )
            yield
            return
        carried = getattr(forward_batch.spec_info, "dsa_topk_indices", None)
        if carried is None:
            self._reject(f"missing carried TopK at step={step}")
            yield
            return

        previous_callback = forward_batch._dsa_topk_shadow_callback
        previous_step = forward_batch._dsa_topk_shadow_step
        callback_count = 0

        def callback(**kwargs):
            nonlocal callback_count
            callback_count += 1
            self.compare(step=step, carried=carried, **kwargs)

        forward_batch._dsa_topk_shadow_callback = callback
        forward_batch._dsa_topk_shadow_step = step
        try:
            yield
        finally:
            forward_batch._dsa_topk_shadow_callback = previous_callback
            forward_batch._dsa_topk_shadow_step = previous_step
            if callback_count != 1:
                self._reject(
                    f"expected one NextN DSA indexer callback at step={step}, "
                    f"got {callback_count}"
                )
            else:
                self._last_step = step

    def compare(
        self,
        *,
        step: int,
        carried: torch.Tensor,
        self_topk: Optional[torch.Tensor] = None,
        indexer=None,
        x: Optional[torch.Tensor] = None,
        q_lora: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        layer_id: int,
    ) -> None:
        """Compare a production self TopK, or compute one for legacy probes."""
        if not self.can_probe:
            return
        if get_is_capture_mode():
            self._reject("shadow callback entered CUDA graph capture/replay")
            return
        x_meta = x[0] if isinstance(x, tuple) else x
        if self_topk is not None:
            if carried.ndim != 2 or self_topk.shape != carried.shape:
                self._reject(
                    "self TopK shape mismatch: "
                    f"self={tuple(self_topk.shape)}, carried={tuple(carried.shape)}"
                )
                return
            self._accumulate(step, layer_id, carried, self_topk)
            return
        if (
            x_meta is None
            or q_lora is None
            or positions is None
            or forward_batch is None
            or indexer is None
        ):
            self._reject("legacy shadow comparison is missing indexer inputs")
            return
        if carried.ndim != 2 or carried.shape[0] != x_meta.shape[0]:
            self._reject(
                "carried TopK shape mismatch: "
                f"carried={tuple(carried.shape)}, rows={x_meta.shape[0]}"
            )
            return
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc is None or out_cache_loc.ndim != 1:
            self._reject("out_cache_loc must be a one-dimensional tensor")
            return
        if out_cache_loc.shape[0] != x_meta.shape[0]:
            self._reject(
                "out_cache_loc row mismatch: "
                f"loc={out_cache_loc.shape[0]}, rows={x_meta.shape[0]}"
            )
            return

        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        pool = get_token_to_kv_pool()
        if getattr(pool, "page_size", None) != 64:
            self._reject(f"expected CUDA DSA page_size=64, got {pool.page_size}")
            return
        if getattr(pool, "index_head_dim", None) != 128:
            self._reject(f"expected index_head_dim=128, got {pool.index_head_dim}")
            return
        if hasattr(pool, "_is_layer_owned"):
            self._reject("DSA cache layer split pool is unsupported")
            return

        buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
        snapshot = snapshot_k_and_s_by_loc(
            buf=buf,
            loc=out_cache_loc,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )
        try:
            with suspend_indexer_topk_capture():
                self_topk = indexer(
                    x=x,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                )
        finally:
            restore_k_and_s_by_loc(
                buf=buf,
                loc=out_cache_loc,
                snapshot=snapshot,
                page_size=pool.page_size,
                index_head_dim=pool.index_head_dim,
            )

        if self_topk is None or self_topk.shape != carried.shape:
            shape = None if self_topk is None else tuple(self_topk.shape)
            self._reject(
                f"self TopK shape mismatch: self={shape}, carried={tuple(carried.shape)}"
            )
            return
        self._accumulate(step, layer_id, carried, self_topk)

    def _accumulate(
        self,
        step: int,
        layer_id: int,
        carried: torch.Tensor,
        self_topk: torch.Tensor,
    ) -> None:
        carried_i64 = carried.to(torch.int64)
        self_i64 = self_topk.to(torch.int64)
        carried_valid = carried_i64 >= 0
        self_valid = self_i64 >= 0
        same = carried_i64 == self_i64

        sorted_self = torch.sort(
            torch.where(self_valid, self_i64, torch.iinfo(torch.int64).max), dim=1
        ).values
        search = torch.searchsorted(sorted_self, carried_i64.clamp_min(0))
        search.clamp_(max=sorted_self.shape[1] - 1)
        intersection = (
            torch.gather(sorted_self, 1, search) == carried_i64
        ) & carried_valid
        weights = torch.arange(
            1, carried_i64.shape[1] + 1, device=carried.device, dtype=torch.int64
        )
        carried_checksum = (
            torch.where(carried_valid, carried_i64 + 1, 0) * weights
        ).sum()
        self_checksum = (torch.where(self_valid, self_i64 + 1, 0) * weights).sum()
        metrics = torch.stack(
            (
                same.all(dim=1).sum(),
                torch.full(
                    (), carried.shape[0], dtype=torch.int64, device=carried.device
                ),
                (same & carried_valid & self_valid).sum(),
                carried_valid.sum(),
                self_valid.sum(),
                intersection.sum(),
                carried_checksum,
                self_checksum,
            )
        )
        provenance = "draft_extend_seed" if step == 0 else "step_0_publish"
        key = _BucketKey(step=step, provenance=provenance, layer_id=layer_id)
        if key in self._buckets:
            self._buckets[key].add_(metrics)
        else:
            self._buckets[key] = metrics

    def finish(self, *, rid: str, natural_stop: bool) -> None:
        if rid != self.expected_rid or self._sealed:
            return
        self._sealed = True
        if self._rejection is None and not self._buckets:
            self._rejection = "target request produced no complete shadow samples"
        keys = sorted(
            self._buckets, key=lambda key: (key.step, key.layer_id, key.provenance)
        )
        rows = []
        if keys:
            # This is the diagnostic's only device-to-host transfer and sync.
            values = torch.stack([self._buckets[key] for key in keys]).cpu().tolist()
            for key, metrics in zip(keys, values):
                rows.append(
                    {
                        "step": key.step,
                        "provenance": key.provenance,
                        "layer_id": key.layer_id,
                        **dict(zip(_METRIC_NAMES, metrics)),
                    }
                )
        payload = {
            "rid": rid,
            "natural_stop": natural_stop,
            "status": "rejected" if self._rejection else "complete",
            "rejection": self._rejection,
            "seen": self._seen,
            "buckets": rows,
        }
        try:
            parallel = get_parallel()
            payload["rank"] = {
                "world": parallel.world_rank,
                "pp": parallel.pp_rank,
                "attn_dp": parallel.attn_dp_rank,
            }
        except (AssertionError, AttributeError, RuntimeError):
            # Unit harnesses and startup-failure paths may not have published
            # process groups.  The result remains self-identifying by RID.
            pass
        logger.warning(
            "DSA_TOPK_SHADOW_RESULT %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        self._buckets.clear()
