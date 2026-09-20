# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Startup tuning and immutable capacity profiles for FlashInfer MegaMoE."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class _AutotuneContext:
    cache_path: Path
    decode_num_tokens: int
    prefill_num_tokens: int
    reuse_cache: bool


_active_context: _AutotuneContext | None = None
# Native workspace pooling spans startup contexts. Keep their
# selected tactics consistent too; a cache-path change must not retune storage
# already captured by another layer. These payloads own no weights or tensors.
_session_winners: dict[tuple[Any, str], dict] = {}


@contextmanager
def megamoe_autotune_context(
    cache_path: Path,
    decode_num_tokens: int,
    prefill_num_tokens: int,
    reuse_cache: bool = True,
):
    """Tune only inside the engine's collective, pre-capture startup forward."""
    global _active_context
    previous = _active_context
    _active_context = _AutotuneContext(
        Path(cache_path), decode_num_tokens, prefill_num_tokens, reuse_cache
    )
    try:
        yield
    finally:
        _active_context = previous


@contextmanager
def _native_cache(path: Path):
    name = "FLASHINFER_MOE_EP_KNOB_CACHE"
    previous = os.environ.get(name)
    os.environ[name] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _capacities(limit: int) -> list[int]:
    limit = max(1, limit)
    return sorted({1 << i for i in range(limit.bit_length())} | {limit})


def _profile_inputs(tensors, capacity: int, num_experts: int):
    count = tensors.num_tokens
    if count:

        def resize(tensor):
            if capacity <= count:
                return tensor[:capacity]
            return tensor.repeat(
                ((capacity + count - 1) // count,) + (1,) * (tensor.ndim - 1)
            )[:capacity]

        hidden = resize(tensors.hidden_states)
        ids = resize(tensors.topk_ids)
        scores = resize(tensors.topk_weights)
        scales = resize(tensors.scales) if tensors.scales is not None else None
    else:
        # Idle ranks still enter every collective tuning iteration.
        hidden = tensors.hidden_states.new_zeros(
            (capacity, tensors.hidden_states.shape[1])
        )
        top_k = tensors.topk_ids.shape[1]
        ids = (
            torch.arange(
                capacity * top_k, dtype=tensors.topk_ids.dtype, device=hidden.device
            ).reshape(capacity, top_k)
            % num_experts
        )
        scores = tensors.topk_weights.new_full((capacity, top_k), 1.0 / top_k)
        scales = (
            tensors.scales.new_ones((capacity, *tensors.scales.shape[1:]))
            if tensors.scales is not None
            else None
        )
    return replace(
        tensors, hidden_states=hidden, topk_ids=ids, topk_weights=scores, scales=scales
    )


def _valid_native_cache(payload) -> bool:
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return False
    entries = payload.get("entries")
    return (
        isinstance(entries, list)
        and len(entries) == 1
        and isinstance(entries[0], dict)
        and isinstance(entries[0].get("knobs"), dict)
        and bool(entries[0]["knobs"])
    )


class MegaMoeTunedForward:
    """Own public workspace handles; FlashInfer pools their storage across layers.

    The caller installs this for backends with prepared weights and default
    ``knobs=None``. Preparation must precede ordinary forwarding and capture.
    """

    def __init__(self, bootstrap, fleet_params, backend):
        if backend.megakernel.knobs is not None:
            raise ValueError("MegaMoE startup tuning requires knobs=None")
        if backend.preprocess_weights or backend.transformed_weights is None:
            raise ValueError("MegaMoE startup tuning requires prepared weights")
        self.bootstrap = bootstrap
        self.fleet_params = fleet_params
        self.backend = backend
        self.workspaces: dict[int, Any] = {}
        self._fallback_used = False

    def _common_tokens(self, tensors) -> int:
        if self.bootstrap.world_size == 1:
            return tensors.num_tokens
        from sglang.srt.layers.dp_attention import get_dp_global_num_tokens

        counts = get_dp_global_num_tokens()
        if counts is None:
            raise RuntimeError(
                "MegaMoE workspace selection requires synchronized DP token counts"
            )
        return max(counts, default=0)

    def _metadata(self, capacity: int) -> dict:
        from flashinfer.autotuner import _collect_metadata

        if self.bootstrap.world_size > 1:
            import torch.distributed as dist

            members = dist.get_process_group_ranks(self.bootstrap.process_group)
        else:
            members = [self.bootstrap.rank]

        config = {}
        for item in fields(self.backend.megakernel):
            # Per-layer scale values do not change the pooled kernel geometry.
            if item.name in (
                "knobs",
                "input_norm_const",
                "fc1_norm_const",
                "fc1_alpha",
                "fc2_alpha",
            ):
                continue
            value = getattr(self.backend.megakernel, item.name)
            if not isinstance(value, torch.Tensor):
                config[item.name] = value
        return {
            "version": 1,
            "runtime": _collect_metadata(),
            "torch": torch.__version__,
            "mega_use_ncu": os.environ.get("MEGA_USE_NCU", "0"),
            "config": config,
            "ep_members": members,
            "world_size": self.bootstrap.world_size,
            "hidden": self.fleet_params.token_hidden_size,
            "num_experts": self.fleet_params.num_experts,
            "capacity": capacity,
        }

    def _require_all_ranks(self, success: bool, message: str) -> None:
        if self.bootstrap.world_size > 1:
            import torch.distributed as dist

            ready = torch.tensor(int(success), dtype=torch.int32, device="cuda")
            dist.all_reduce(
                ready, op=dist.ReduceOp.MIN, group=self.bootstrap.process_group
            )
            success = bool(ready.item())
        if not success:
            raise RuntimeError(message)

    def _broadcast(self, value):
        if self.bootstrap.world_size == 1:
            return value
        import torch.distributed as dist

        group = self.bootstrap.process_group
        values = [value]
        dist.broadcast_object_list(values, group=group, group_src=0)
        return values[0]

    def _prepare_profile(self, context, mega, tensors, capacity: int) -> None:
        from flashinfer.moe_ep import MoEEpMegaLayer

        metadata = self._metadata(capacity)
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()
        key = (self.bootstrap.process_group, digest)
        directory = context.cache_path.with_suffix(".megamoe") / digest
        directory.mkdir(parents=True, exist_ok=True)
        # SGLang's parent path is rank-local; keep native writes isolated too.
        native_path = directory / f"rank{self.bootstrap.rank}.{os.getpid()}.native.json"
        cache_path = directory / f"rank{self.bootstrap.rank}.json"
        profile_inputs = _profile_inputs(
            tensors, capacity, self.fleet_params.num_experts
        )
        payload = _session_winners.get(key)
        if payload is None:
            if self.bootstrap.rank == 0 and context.reuse_cache:
                try:
                    saved = json.loads(cache_path.read_text())
                    if saved.get("metadata") == metadata and _valid_native_cache(
                        saved.get("native_cache")
                    ):
                        payload = saved["native_cache"]
                except (OSError, ValueError, KeyError, AttributeError):
                    pass
            payload = self._broadcast(payload)
            if payload is None:
                with _native_cache(native_path):
                    native_path.unlink(missing_ok=True)
                    temporary = MoEEpMegaLayer(
                        bootstrap=self.bootstrap,
                        fleet_params=replace(
                            self.fleet_params, max_tokens_per_rank=capacity
                        ),
                        weights=None,
                        backend=replace(
                            self.backend,
                            megakernel=replace(self.backend.megakernel, knobs="auto"),
                        ),
                    )
                    try:
                        temporary.warmup(profile_inputs)
                        if self.bootstrap.rank == 0:
                            try:
                                recorded = json.loads(native_path.read_text())
                                if _valid_native_cache(recorded):
                                    payload = recorded
                            except (OSError, ValueError):
                                pass
                    finally:
                        temporary.destroy()
                payload = self._broadcast(payload)
                if payload is None:
                    raise RuntimeError(
                        "MegaMoE autotune did not record its selected tactic"
                    )
                source = "tuned"
            else:
                source = "cache hit"
            _session_winners[key] = payload
            if self.bootstrap.rank == 0:
                logger.info(
                    "MegaMoE %s: capacity=%s, knobs=%s",
                    source,
                    capacity,
                    payload["entries"][0]["knobs"],
                )
            try:
                pending = cache_path.with_suffix(f".{os.getpid()}.tmp")
                pending.write_text(
                    json.dumps({"metadata": metadata, "native_cache": payload})
                )
                pending.replace(cache_path)
            except OSError as exc:
                logger.warning("Cannot persist MegaMoE tuning result: %s", exc)

        with _native_cache(native_path):
            # The backend owns dtype/layout keys and loads its original payload.
            try:
                native_path.write_text(json.dumps(payload))
                written = True
            except OSError:
                written = False
            self._require_all_ranks(
                written,
                "Cannot stage MegaMoE's selected tactic for workspace creation",
            )
            workspace = mega.create_workspace(capacity)
        mega.warmup(profile_inputs, workspace=workspace)
        self.workspaces[capacity] = workspace

    @staticmethod
    def _forward(mega, tensors, workspace=None):
        kwargs = {} if workspace is None else {"workspace": workspace}
        if getattr(mega, "supports_output_view", False):
            kwargs["return_workspace_view"] = True
        return mega.forward(tensors, **kwargs)

    def __call__(self, mega, tensors):
        context = _active_context
        if context is None and not self.workspaces:
            self._fallback_used = True
            return self._forward(mega, tensors)
        if context is not None and not self.workspaces:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoE autotune must complete before CUDA graph capture"
                )
            if self._fallback_used:
                raise RuntimeError(
                    "MegaMoE autotune must precede its first ordinary forward"
                )
            import torch.distributed as dist

            if (
                dist.is_initialized()
                and self.bootstrap.world_size != dist.get_world_size()
            ):
                raise RuntimeError(
                    "FlashInfer MegaMoE startup tuning currently requires the full WORLD process group"
                )
            # The startup dummy can be much larger than captured decode batches.
            # Keep small-N profiles bounded by decode, plus one prefill profile.
            capacities = set(_capacities(context.decode_num_tokens))
            capacities.add(context.prefill_num_tokens)
            for capacity in sorted(capacities):
                self._prepare_profile(context, mega, tensors, capacity)

        count = self._common_tokens(tensors)
        capacity = next((n for n in sorted(self.workspaces) if n >= count), None)
        if capacity is not None:
            return self._forward(mega, tensors, self.workspaces[capacity])
        raise ValueError(
            f"MegaMoE received {count} tokens per rank, exceeding its prepared "
            f"decode/prefill capacity {max(self.workspaces)}"
        )
