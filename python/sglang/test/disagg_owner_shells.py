"""PD owner shells for scheduler tests: a DisaggPrefill or DisaggDecode with
empty lists and no transfer engine, and the result processor the PD prefill
commit calls; a test sets what it needs on top."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.decode import DisaggDecode
from sglang.srt.disaggregation.prefill import DisaggPrefill
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler_components import (
    request_retirer as request_retirer_mod,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.engine_state import EngineFlags


def prefill_owner_shell(**attrs) -> DisaggPrefill:
    owner = DisaggPrefill.__new__(DisaggPrefill)
    owner.bootstrap_queue = SimpleNamespace(queue=[], kv_senders={})
    owner.kv_senders = owner.bootstrap_queue.kv_senders
    owner.prefilling = []
    owner.transfer = []
    owner.pending_chunk_rids = set()
    owner.core_lists = CoreLists()
    owner.__dict__.update(attrs)
    return owner


def decode_owner_shell(**attrs) -> DisaggDecode:
    owner = DisaggDecode.__new__(DisaggDecode)
    owner.prealloc_queue = SimpleNamespace(
        queue=[],
        retracted_queue=[],
        held_rebootstrap_reqs=[],
        kv_manager=SimpleNamespace(
            should_verify_kv_checksum=lambda bootstrap_addr: True
        ),
        pre_allocated_token_counts=lambda: (0, 0),
    )
    owner.transfer_queue = SimpleNamespace(queue=[])
    owner.attn_tp_cpu_group = None
    # Empty pools: the prealloc usage gauges read 0.
    owner.pool_stats_observer = SimpleNamespace(
        get_pool_stats=lambda: SimpleNamespace(
            full_capacity=0, swa_capacity=0, is_hybrid_swa=False
        )
    )
    owner.ready = []
    owner.releasing = []
    owner.decode_offload_manager = None
    owner.enable_decode_hicache = False
    owner.enable_host_receive = False
    owner.flags = EngineFlags()
    owner.__dict__.update(attrs)
    return owner


def tree_release(owner):
    """The retirer's retire_finished without HiSparse, for a PD decode owner or
    queue shell: the tree release, looked up in the retirer module so a test
    can patch it there."""

    def retire_finished(req, *, checkpoint):
        request_retirer_mod.release_kv_cache(
            req, owner.tree_cache, checkpoint=checkpoint
        )

    return retire_finished


def result_processor_shell(**attrs) -> SimpleNamespace:
    """A result processor whose extend loop, host_prefill_result and row
    helpers are the real ones, over mocked logprob, auxiliary-output and
    free-group steps."""
    processor = SimpleNamespace(
        kv=SimpleNamespace(token_to_kv_pool_allocator=Mock()),
        snapshot_auxiliary_output_starts=Mock(return_value=[]),
        move_logprobs_to_cpu=Mock(),
        materialize_sampling_mask_output=(
            SchedulerBatchResultProcessor.materialize_sampling_mask_output
        ),
        _validate_pp_skip_output_comm=(
            SchedulerBatchResultProcessor._validate_pp_skip_output_comm
        ),
        consume_auxiliary_output=Mock(),
    )
    processor.__dict__.update(attrs)
    for name in (
        "commit_extend",
        "host_prefill_result",
        "sampling_mask_finish_reason",
        "apply_prefill_logprobs",
        "apply_chunked_prefill_logprobs",
        "apply_prefill_grammar",
    ):
        if name not in attrs:
            setattr(
                processor,
                name,
                MethodType(getattr(SchedulerBatchResultProcessor, name), processor),
            )
    return processor
