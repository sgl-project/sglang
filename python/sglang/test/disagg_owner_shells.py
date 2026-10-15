"""PD owner shells for scheduler tests: a DisaggPrefill or DisaggDecode with
empty lists and no transfer engine, and the result processor the PD prefill
commit calls; a test sets what it needs on top."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.decode import DisaggDecode
from sglang.srt.disaggregation.prefill import DisaggPrefill
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
    owner.__dict__.update(attrs)
    return owner


def decode_owner_shell(**attrs) -> DisaggDecode:
    owner = DisaggDecode.__new__(DisaggDecode)
    owner.prealloc_queue = SimpleNamespace(
        queue=[], retracted_queue=[], held_rebootstrap_reqs=[]
    )
    owner.transfer_queue = SimpleNamespace(queue=[])
    owner.ready = []
    owner.releasing = []
    owner.decode_offload_manager = None
    owner.flags = EngineFlags()
    owner.__dict__.update(attrs)
    return owner


def tree_release(owner):
    """The retirer's retire_finished without HiSparse, for a PD decode owner or
    queue shell: the tree release, looked up in the decode module so a test
    can patch it there."""

    def retire_finished(req, *, checkpoint):
        decode_mod.release_kv_cache(req, owner.tree_cache, checkpoint=checkpoint)

    return retire_finished


def result_processor_shell(**attrs) -> SimpleNamespace:
    """A result processor whose host_prefill_result is the real one, over mocked
    logprob and auxiliary-output steps."""
    processor = SimpleNamespace(
        snapshot_auxiliary_output_starts=Mock(return_value=[]),
        move_logprobs_to_cpu=Mock(),
        materialize_sampling_mask_output=(
            SchedulerBatchResultProcessor.materialize_sampling_mask_output
        ),
        consume_auxiliary_output=Mock(),
    )
    processor.__dict__.update(attrs)
    processor.host_prefill_result = MethodType(
        SchedulerBatchResultProcessor.host_prefill_result, processor
    )
    return processor
