"""Synchronous eager DSA verifier transactions for native GLM MTP diagnostics.

Opt in with SGLANG_HISPARSE_SPECULATIVE_EAGER=1. The supported checkpoint is
aggregate GLM DSA EAGLE/EAGLE3 steps3/topk1/draft4, page64 bf16, without overlap
or graph capture. Each request/layer stages its complete selected union; an
insufficient hot buffer fails closed instead of evicting another verifier row.
The expected layer set comes from the actual target pool (78 for full GLM-5.3),
not from an assumed model-wide count or the draft head's pool.

CPU tests exercise ownership and production call sites, not GPU numerical
correctness or performance. Separate PDD destination/state transfer, graphs,
asynchronous retirement, and broader layouts remain unimplemented. Scheduler
logical reservations survive transaction retirement until request teardown.
"""

from copy import copy
from functools import wraps

import torch

from sglang.srt.mem_cache.hisparse_spec_state import VerifyRow


def validate_eager_layout(cfg, hf_config):
    """Reject unsupported routes before workers allocate pools or capture graphs."""
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.cuda_graph_config import Backend

    if not (cfg.enable_hisparse and cfg.speculative_algorithm):
        return
    if not envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.get():
        raise ValueError(
            "HiSparse speculation requires SGLANG_HISPARSE_SPECULATIVE_EAGER=1"
        )
    supported = (
        hf_config.architectures == ["GlmMoeDsaForCausalLM"]
        and cfg.speculative_algorithm in ("EAGLE", "EAGLE3")
        and cfg.speculative_eagle_topk == 1
        and cfg.speculative_num_steps == 3
        and cfg.speculative_num_draft_tokens == 4
        and not cfg.speculative_adaptive
        and not cfg.enable_multi_layer_eagle
        and cfg.disable_overlap_schedule
        and cfg.cuda_graph_config.decode.backend == Backend.DISABLED
        and cfg.cuda_graph_config.prefill.backend == Backend.DISABLED
        and cfg.disaggregation_mode == "null"
        and cfg.pp_size == 1
        and cfg.dp_size == 1
        and cfg.attn_cp_size == 1
        and not cfg.enable_dp_attention
        and cfg.page_size == 64
        and cfg.kv_cache_dtype == "bfloat16"
        and getattr(hf_config, "index_kpool", 1) == 1
    )
    if not supported:
        raise ValueError(
            "eager HiSparse speculation supports only GLM DSA, EAGLE/EAGLE3 steps3/topk1/draft4, "
            "page64 bf16, single PP/DP/CP, aggregate, without graphs or overlap"
        )
    if envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get():
        raise ValueError("eager HiSparse speculation requires real KV copies")


def save_staging_draft(req, draft, index):
    """Keep a request-owned copy while the prefill batch is dismantled."""
    if draft is None or draft.future_indices is not None:
        raise ValueError("HiSparse staging requires synchronous native draft state")
    saved = copy(draft)
    selected = torch.tensor(
        [index], dtype=torch.int64, device=draft.bonus_tokens.device
    )
    saved.filter_batch(selected)
    req.hisparse_staging_draft = saved


def restore_staging_drafts(reqs):
    """Reassemble in ready-request order, which can differ from prefill order."""
    drafts = [getattr(req, "hisparse_staging_draft", None) for req in reqs]
    if not drafts or any(draft is None for draft in drafts):
        raise ValueError("missing native draft state after HiSparse staging")
    merged = copy(drafts[0])
    for draft in drafts[1:]:
        merged.merge_batch(draft)
    for req in reqs:
        del req.hisparse_staging_draft
    return merged


class EagerHiSparseBatch:
    """Keep every target/draft reader alive until a synchronous worker boundary."""

    def __init__(self, coordinator):
        self.adapter = coordinator.speculative_verifier()
        self.dm = self.adapter.dm
        self.entries = []
        self.target_readers = []
        self.draft_readers = []
        self.target_recorded = False
        self.committed = False
        self.width = None
        self.expected_layers = set(range(coordinator.mem_pool_device.layer_num))
        self.staged_layers = set()

    def prepare(self, batch, forward_batch, width, reserved_rows):
        if self.entries:
            raise ValueError("verifier transaction already prepared")
        ids = batch.out_cache_loc.reshape(len(batch.reqs), width).tolist()
        lengths = batch.seq_lens.tolist()
        self.width = width
        for req, old_len, writes in zip(batch.reqs, lengths, ids, strict=True):
            key = self.adapter.prepare(req, old_len, writes, reserved_rows)
            self.entries.append((req, key, old_len))
            self.target_readers.append(self.adapter.add_reader(req, key))
            # Register before any submission; draft extension may share verify
            # tensors and must finish before any provisional mapping is released.
            self.draft_readers.append(self.adapter.add_reader(req, key))
        forward_batch.hisparse_spec_transaction = self

    def stage_layer(self, layer, selections, output_rows):
        if self.target_recorded or self.committed:
            raise ValueError("cannot stage after verifier completion")
        real_rows = len(self.entries) * self.width
        if (
            selections is None
            or selections.ndim != 2
            or selections.shape[0] < real_rows
        ):
            raise ValueError("missing raw verifier selections")
        if output_rows < real_rows:
            raise ValueError("verifier output omits real rows")
        selected = selections[:real_rows].tolist()
        tables = []
        for index, (req, key, old_len) in enumerate(self.entries):
            rows = tuple(
                VerifyRow(
                    key, old_len + offset, tuple(selected[index * self.width + offset])
                )
                for offset in range(self.width)
            )
            tables.append(self.adapter.stage_layer(req, key, layer, rows))
        self.staged_layers.add(layer)
        table = torch.cat(tables)
        if output_rows > real_rows:
            table = torch.cat(
                (table, table.new_full((output_rows - real_rows, table.shape[1]), -1))
            )
        return table

    def target_done(self):
        for fence in self.target_readers:
            fence.record(self.dm.current_stream())
        self.target_recorded = True

    def commit(self, accept_lens, accept_index):
        if not self.target_recorded:
            raise ValueError("cannot commit before target completion is fenced")
        if self.staged_layers != self.expected_layers:
            raise ValueError("not every verifier attention layer staged a union")
        lengths = accept_lens.tolist()
        paths = accept_index.tolist()
        # The linear accepted path includes the root; bonus output has no KV.
        for index, length in enumerate(lengths):
            if not 1 <= length <= self.width:
                raise ValueError("invalid accepted verifier length")
            if paths[index][:length] != list(
                range(index * self.width, index * self.width + length)
            ):
                raise ValueError("nonlinear accepted path is unsupported")
        for (req, key, _), length in zip(self.entries, lengths, strict=True):
            self.adapter.verified(req, key)
            self.adapter.commit(req, key, length)
        self.committed = True

    def close(self, *, failed):
        if not self.target_recorded:
            self.target_done()
        for fence in self.draft_readers:
            fence.record(self.dm.current_stream())
        # Diagnostic eager boundary: no scheduler frees or next-iteration slot
        # reuse can run before target, backup, union-copy and draft completion.
        self.dm.synchronize()
        for req, key, _ in self.entries:
            if not self.adapter.retire(req, key, cancel=failed or not self.committed):
                raise RuntimeError("synchronized verifier still owns pending readers")


def eager_hisparse_worker_boundary(function):
    """Defer publication and all scheduler frees until retirement completes."""

    @wraps(function)
    def wrapped(self, batch, on_publish=None, **kwargs):
        coordinator = getattr(batch, "hisparse_coordinator", None)
        if (
            coordinator is None
            or batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
        ):
            return function(self, batch, on_publish=on_publish, **kwargs)
        from sglang.srt.environ import envs

        if not envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.get():
            raise ValueError("HiSparse speculative worker requires eager opt-in")
        transaction = EagerHiSparseBatch(coordinator)
        batch.hisparse_spec_transaction = transaction
        failed = True
        try:
            result = function(self, batch, on_publish=None, **kwargs)
            failed = False
        finally:
            transaction.close(failed=failed)
            batch.hisparse_spec_transaction = None
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        return result

    return wrapped
