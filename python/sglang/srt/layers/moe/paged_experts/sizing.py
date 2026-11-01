"""K, the experts per MoE layer resident on the GPU, with ``--paged-experts-num-resident auto``
(see "Choosing K" in the paged experts docs). Layers are built with K = top_k; the KV cache's aim
is applied as ``--max-total-tokens``, and after the KV cache is allocated the expert slots take
what the static budget has left, in one allocation for all layers.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.runtime_context import (
    get_context,
    get_mm,
    get_model,
    get_schedule,
    get_server_args,
)

logger = logging.getLogger(__name__)

_GIB = 1 << 30


@dataclass
class KVFloor:
    tokens: int  # what the KV cache aims to hold
    minimum: int  # what it must hold: one full-length request, or --max-total-tokens
    requests: Optional[
        int
    ]  # full-length requests aimed for; None with --max-total-tokens
    source: str  # the flags they come from, for the log and errors
    hard: bool  # set by the user: fail if the KV cache cannot hold ``minimum``


def _requests_set() -> Optional[int]:
    """--max-running-requests as the user set it (the record): a value SGLang resolves itself
    (e.g. a model's default) caps the request pool, but is no target for the KV cache."""
    return get_server_args().max_running_requests


def kv_floor(model_context_len: int) -> KVFloor:
    schedule = get_schedule()
    if schedule.max_total_tokens is not None:
        mtt = schedule.max_total_tokens
        return KVFloor(mtt, mtt, None, "--max-total-tokens", True)
    requests = _requests_set() or 1
    hard = get_model().context_length is not None
    context = "--context-length" if hard else "the model's context"
    # In whole pages: the KV cache rounds down to them, and one request must still fit.
    page = schedule.page_size
    request = -(-model_context_len // page) * page
    return KVFloor(
        request * requests,
        request,
        requests,
        f"{context} {model_context_len} x {requests} request(s)",
        hard,
    )


def static_budget_bytes(kv_configurator, pre_model_load_memory: float) -> int:
    """What is left of the static budget (``--mem-fraction-static``): free memory less the
    non-static slack and the multimodal reserve, as the KV configurator measures it (its other
    terms are for multi-GPU and post-capture KV sizing, which paged experts do not run)."""
    from sglang.srt.mem_cache.kv_cache_configurator import mm_runtime_reservation_gb
    from sglang.srt.utils.common import get_available_gpu_memory

    gc.collect()
    free_gb = get_available_gpu_memory(kv_configurator.device, kv_configurator.gpu_id)
    slack_gb = pre_model_load_memory * (1 - get_schedule().mem_fraction_static)
    mm_gb = mm_runtime_reservation_gb(
        is_multimodal=kv_configurator.model_config.is_multimodal,
        mm_feature_transport=get_mm().mm_feature_transport,
    )
    return int((free_gb - slack_gb - mm_gb) * _GIB)


def _mamba_charge(slots: int, per_slot: int) -> int:
    """Bytes a Mamba pool of ``slots`` takes: one padding slot beyond its size, as SGLang
    charges it (``KVCacheConfigurator._handle_max_mamba_cache``)."""
    return (slots + 1) * per_slot


def _mamba_share(budget_bytes: int, ratio: float, per_slot: int) -> int:
    """The slots SGLang's own share of ``budget_bytes`` holds (``--mamba-full-memory-ratio``),
    less the padding slot, as its ratio branch solves it."""
    return int((budget_bytes * ratio / (1 + ratio) - per_slot) // per_slot)


def _kv_tokens(kv_configurator, budget_bytes: int) -> int:
    """The tokens the KV cache holds in ``budget_bytes``, as SGLang's pool configurator sizes
    it."""
    from sglang.srt.model_executor.pool_configurator import (
        create_memory_pool_configurator,
    )

    configurator = create_memory_pool_configurator(kv_configurator)
    return configurator.calculate_pool_sizes(
        max(budget_bytes, 0), get_schedule().page_size
    ).max_total_num_tokens


def _bytes_per_slot(method) -> int:
    """One more GPU slot of a layer: a row of every paged tensor."""
    from sglang.srt.layers.moe.paged_experts.method import _row_bytes

    return sum(_row_bytes(h) for h in method.store.host.values())


class ExpertSlotSizing:
    """Sizes the slot tables of the paged layers of one model built for ``auto``."""

    def __init__(
        self,
        layers: List,
        kv_configurator,
        pre_model_load_memory: float,
        budget_before_kv: int,
    ):
        self.layers = layers  # FusedMoE layers whose paged method waits to be sized
        self.kv_configurator = kv_configurator
        self.pre_model_load_memory = pre_model_load_memory
        self.floor = kv_floor(kv_configurator.model_config.context_len)
        self.budget_before_kv = budget_before_kv
        self.mamba_slots = self._size_mamba_cache()
        if get_schedule().max_total_tokens is None:
            get_context().override(
                "paged_experts.sizing", max_total_tokens=self.floor.tokens
            )

    @classmethod
    def for_model(
        cls, model, kv_configurator, pre_model_load_memory: float
    ) -> Optional[ExpertSlotSizing]:
        from sglang.srt.layers.moe.paged_experts.method import PagedExpertsMoEMethod

        layers = [
            m
            for m in model.modules()
            if isinstance(getattr(m, "quant_method", None), PagedExpertsMoEMethod)
            and m.quant_method.auto_size
        ]
        if not layers:
            return None
        if kv_configurator.post_capture_kv_active:
            raise RuntimeError(
                "Paged experts: --paged-experts-num-resident auto sizes the experts against "
                "the KV cache before CUDA graph capture, which post-capture KV sizing "
                "(SGLANG_ENABLE_POST_CAPTURE_KV_SIZING) would resize; set an explicit "
                "--paged-experts-num-resident"
            )
        return cls(
            layers,
            kv_configurator,
            pre_model_load_memory,
            static_budget_bytes(kv_configurator, pre_model_load_memory),
        )

    def _size_mamba_cache(self) -> Optional[int]:
        """Mamba state for the requests the KV cache aims for, within SGLang's own share of the
        budget (which is the slots' under ``auto``) and what the KV cache's minimum leaves: one
        full-length request comes first, more requests queue."""
        kvc = self.kv_configurator
        if (
            kvc.mambaish_config is None
            or get_schedule().max_mamba_cache_size is not None
        ):
            return None
        ratio = get_schedule().mamba_full_memory_ratio
        per_slot = (
            kvc.mambaish_config.mamba2_cache_params.mamba_cache_per_req
            + kvc.extra_mamba_cache_bytes_per_req
        )
        share = _mamba_share(self.budget_before_kv, ratio, per_slot)
        per_request = kvc._calculate_mamba_ratio()  # state slots one request needs
        requests = _requests_set()
        self.mamba_source = (
            f"--max-running-requests {requests}" if requests else "one request"
        )
        slots = max(per_request, min((requests or 1) * per_request, share))
        # One request fewer at a time until the KV cache's minimum fits next to them, by
        # size()'s rule (the KV cache rounds down to whole pages).
        page = get_schedule().page_size
        while slots > per_request and (
            _kv_tokens(kvc, self.budget_before_kv - _mamba_charge(slots, per_slot))
            + page
            <= self.floor.minimum
        ):
            slots = max(per_request, slots - per_request)
        get_context().override("paged_experts.sizing", max_mamba_cache_size=slots)
        return slots

    def size(self, kv_tokens: int) -> List[int]:
        """After the KV cache is allocated: free the K = top-k tables, measure what the
        static budget has left and give it to the slots, in one allocation."""
        from sglang.srt.layers.moe.paged_experts.method import (
            _cap_decode_capture_batch_sizes,
            _router_top_k,
        )

        floor = self.floor
        page = get_schedule().page_size
        short = kv_tokens + page <= floor.minimum
        if floor.hard and short:
            need = (
                f"the {floor.minimum} from {floor.source}"
                if floor.requests is None
                else f"one request of {floor.minimum} ({floor.source})"
            )
            raise RuntimeError(
                f"Paged experts: the KV cache holds {kv_tokens} tokens next to the smallest "
                f"expert slots (K = top-k), fewer than {need}; lower them or raise "
                "--mem-fraction-static"
            )
        methods = [layer.quant_method for layer in self.layers]
        base = [m.num_resident for m in methods]
        num_experts = [m.num_experts for m in methods]
        row = [_bytes_per_slot(m) for m in methods]
        for layer, m in zip(self.layers, methods):
            m.release_slots(layer)
        spare = static_budget_bytes(self.kv_configurator, self.pre_model_load_memory)
        spare -= sum(k * r for k, r in zip(base, row))
        ks = self._grow(base, num_experts, row, spare)
        slab = self._allocate(ks, methods)
        offset = 0
        for layer, m, k in zip(self.layers, methods, ks):
            offset = m.materialize(layer, k, slab, offset)
        _cap_decode_capture_batch_sizes(
            min(k // _router_top_k(layer) for layer, k in zip(self.layers, ks))
        )
        self._log(ks, num_experts, row, kv_tokens, short)
        return ks

    @staticmethod
    def _grow(base, num_experts, row, spare) -> List[int]:
        """K per layer: whole rounds of one more slot in every layer below its E, while a
        round fits ``spare`` bytes."""
        ks = list(base)
        while True:
            open_ = [i for i, k in enumerate(ks) if k < num_experts[i]]
            cost = sum(row[i] for i in open_)
            if not open_ or spare < cost:
                return ks
            rounds = min(spare // cost, min(num_experts[i] - ks[i] for i in open_))
            for i in open_:
                ks[i] += rounds
            spare -= rounds * cost

    def _allocate(self, ks, methods) -> torch.Tensor:
        from sglang.srt.layers.moe.paged_experts.method import _slot_table_bytes

        nbytes = sum(_slot_table_bytes(m, k) for m, k in zip(methods, ks))
        device = getattr(self.layers[0], next(iter(methods[0].store.host))).device
        try:
            return torch.empty(nbytes, dtype=torch.uint8, device=device)
        except torch.OutOfMemoryError as e:
            raise RuntimeError(
                "Paged experts: the expert slots do not fit the memory the KV cache left; "
                "lower --max-total-tokens, --context-length or --max-running-requests, or "
                "set --paged-experts-num-resident"
            ) from e

    def _log(self, ks, num_experts, row, kv_tokens, short) -> None:
        floor = self.floor
        k_text = str(min(ks)) if min(ks) == max(ks) else f"{min(ks)}-{max(ks)}"
        at_once = (
            ""
            if floor.requests is None
            else f", {kv_tokens // floor.minimum} request(s) of {floor.minimum} at once"
        )
        mamba = (
            f"; Mamba state for {self.mamba_slots} slot(s) ({self.mamba_source})"
            if self.mamba_slots
            else ""
        )
        logger.info(
            "Paged experts: K=%s of %d experts per MoE layer resident (%.2f GiB); KV cache "
            "%d tokens%s (aim %d: %s)%s. To pin: --paged-experts-num-resident %d",
            k_text,
            max(num_experts),
            sum(k * r for k, r in zip(ks, row)) / _GIB,
            kv_tokens,
            at_once,
            floor.tokens,
            floor.source,
            mamba,
            min(ks),
        )
        if short:
            logger.warning(
                "Paged experts: one request of the model's context (%d tokens) does not fit "
                "next to the smallest expert slots; the KV cache holds %d tokens, and requests "
                "are capped at that. Set --context-length to choose between context and "
                "resident experts",
                floor.minimum,
                kv_tokens,
            )
