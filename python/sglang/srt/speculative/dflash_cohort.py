"""Synchronous DFlash cohorts; target scheduling and verification stay local.

This eager execution adapter accepts greedy DFlash2 with explicit target owners.
Every owner, including idle owners, participates in PLAN/DRAFT/VERIFY/COMMIT.
The original path remains the default. Draft model forward may use an opt-in
graph; PLAN/COMMIT and selection remain eager. PD uses the typed lane.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from sglang.srt.configs.draft_parallel import DraftParallelLayout
from sglang.srt.distributed.parallel_state import init_model_parallel_group
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftHeadLayout:
    logical_vocab_size: int
    valid_global_ranges: tuple
    replicated: bool = True
    group: object = None


class DraftHeadView(nn.Module):
    def __init__(self, weight, vocab_size):
        super().__init__()
        self.register_buffer("weight", weight)
        self.org_vocab_size = vocab_size
        self.quant_method = None
        self.draft_head_layout = DraftHeadLayout(vocab_size, ((0, vocab_size),))


class DFlashCohort:
    def __init__(self, worker, size):
        parallel = get_parallel()
        if parallel.attn_cp_size != 1 or parallel.pp_group.world_size != 1:
            raise ValueError("Eager draft cohorts currently require target CP1/PP1")
        self.attn_tp_size = parallel.attn_tp_size
        self.owner_count = size
        memberships = parallel.tp_group.all_gather_object(
            tuple(parallel.attn_tp_group.ranks)
        )
        owners = list(dict.fromkeys(memberships))
        width = size * self.attn_tp_size
        if (
            size <= 1
            or len(owners) % size
            or any(len(group) != self.attn_tp_size for group in owners)
        ):
            raise ValueError("Draft cohort size must divide target DP owners")
        groups = [
            list(rank for owner in owners[i : i + size] for rank in owner)
            for i in range(0, len(owners), size)
        ]
        self.worker = worker
        self.group = init_model_parallel_group(
            groups,
            local_rank=parallel.tp_group.local_rank,
            backend=torch.distributed.get_backend(parallel.tp_group.device_group),
            use_pynccl=False,
            use_custom_allreduce=False,
            use_quick_allreduce=False,
            use_mscclpp=False,
            use_torch_symm_mem_allreduce=False,
            group_name="dflash_cohort",
        )
        self.layout = DraftParallelLayout(tp_group=self.group, attn_tp_size=width)
        self.owner_rank = self.group.rank_in_group // self.attn_tp_size
        self.owner_slots = None
        self.sequence = 0
        self.pending = None
        self.committed = {}
        self.head = None

    def prepare_head(self):
        # Only the first owner's complete attention-TP group gathers the head.
        # All its members participate; the cohort leader alone retains it.
        if self.owner_rank != 0:
            return
        head = self.worker.target_worker.model_runner.model.lm_head
        vocab = int(self.worker.model_runner.model_config.vocab_size)
        if head.weight.ndim != 2 or head.weight.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            raise ValueError(
                "Eager draft cohorts require a dense floating-point target head"
            )
        if self.attn_tp_size == 1:
            if head.weight.shape[0] < vocab:
                raise ValueError("Incomplete target head on an attention-TP1 owner")
            self.head = DraftHeadView(head.weight[:vocab], vocab)
            return
        group = get_parallel().attn_tp_group
        gathered = group.gather(head.weight, dst=0, dim=0)
        if group.rank_in_group == 0:
            shards = gathered.view(self.attn_tp_size, *head.weight.shape)
            full = head.weight.new_empty((vocab, head.weight.shape[1]))
            for rank, shard in enumerate(shards):
                indices = head._get_indices(
                    head.num_embeddings_padded,
                    head.org_vocab_size_padded,
                    head.num_embeddings,
                    head.org_vocab_size,
                    rank,
                    head.tp_size,
                )
                if indices.num_added_elements:
                    raise ValueError(
                        "Draft head view does not yet support an added-vocabulary partition"
                    )
                full[indices.org_vocab_start_index : indices.org_vocab_end_index].copy_(
                    shard[: indices.num_org_elements]
                )
            self.head = DraftHeadView(full, vocab)

    def _exchange(self, phase, value):
        self.sequence += 1
        records = self.group.all_gather_object(
            dict(phase=phase, sequence=self.sequence, value=value)
        )
        if any(x["phase"] != phase or x["sequence"] != self.sequence for x in records):
            raise RuntimeError("DFlash cohort phase mismatch")
        values = []
        for start in range(0, len(records), self.attn_tp_size):
            peers = records[start : start + self.attn_tp_size]
            if any(x["value"] != peers[0]["value"] for x in peers):
                raise RuntimeError(
                    "Target attention-TP peers disagree on their owner descriptor"
                )
            values.append(peers[0]["value"])
        return values

    def _descriptor(self, batch, prefixes=None):
        if batch.forward_mode.is_idle():
            return dict(slots=[], generations=[], ids=[], prefixes=[], greedy=True)
        slots = batch.req_pool_indices.cpu().tolist()
        target_requests = self.worker.model_runner.req_to_token_pool
        return dict(
            slots=slots,
            generations=[int(target_requests.req_generation[s]) for s in slots],
            ids=[r.rid for r in batch.reqs],
            prefixes=prefixes,
            greedy=batch.sampling_info is None or batch.sampling_info.is_all_greedy,
        )

    def _slots(self, owner, descriptor):
        return [s + owner * self.owner_slots for s in descriptor["slots"]]

    def _validate_prefixes(self, owner, descriptor):
        for slot, generation, rid, prefix in zip(
            self._slots(owner, descriptor),
            descriptor["generations"],
            descriptor["ids"],
            descriptor["prefixes"],
        ):
            if not 0 <= prefix <= self.worker._independent_cache.layout.context_length:
                raise RuntimeError("Draft cohort context capacity exceeded")
            previous = self.committed.get(slot)
            identity = (generation, rid)
            if prefix and (
                previous is None or previous[:2] != identity or previous[2] < prefix
            ):
                raise RuntimeError(
                    "Draft cohort has no complete context for this owner/generation"
                )
            if prefix == 0:
                self.committed[slot] = (*identity, 0)

    def _payload(self, owner, tensor, shape, dtype):
        source = owner * self.attn_tp_size
        if self.group.rank_in_group == source:
            if tensor is None or tensor.numel() != torch.Size(shape).numel():
                raise RuntimeError("Draft cohort payload does not match its descriptor")
            tensor = tensor.reshape(shape).contiguous()
            if tensor.dtype != dtype:
                raise RuntimeError("Draft cohort payload dtype mismatch")
        else:
            tensor = torch.empty(shape, dtype=dtype, device=self.worker.device)
        self.group.broadcast(tensor, src=source)
        return tensor

    def prefill(self, batch, hidden):
        idle = batch.forward_mode.is_idle()
        descriptor = self._descriptor(batch, [] if idle else list(batch.prefix_lens))
        descriptor["extends"] = [] if idle else list(batch.extend_lens)
        records = self._exchange("PREFILL", descriptor)
        width = self.worker.draft_model.num_context_features * int(
            self.worker.draft_model.config.hidden_size
        )
        for owner, rec in enumerate(records):
            self._validate_prefixes(owner, rec)
            count = sum(rec["extends"])
            if not count:
                continue
            rows = hidden[:count] if owner == self.owner_rank else None
            rows = self._payload(owner, rows, (count, width), torch.bfloat16)
            positions = torch.cat(
                [
                    torch.arange(p, p + n, device=self.worker.device, dtype=torch.int64)
                    for p, n in zip(rec["prefixes"], rec["extends"])
                ]
            )
            slots = torch.tensor(
                self._slots(owner, rec), device=self.worker.device, dtype=torch.int64
            )
            lengths = torch.tensor(
                rec["extends"], device=self.worker.device, dtype=torch.int32
            )
            full, window, keep = self.worker._independent_cache.locations(
                slots, positions, lengths
            )
            self.worker._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=rows,
                cache_loc=full,
                swa_cache_loc=window,
                positions=positions,
                extend_lens=lengths,
                swa_keep_mask=keep,
            )
            for slot, generation, rid, p, n in zip(
                self._slots(owner, rec),
                rec["generations"],
                rec["ids"],
                rec["prefixes"],
                rec["extends"],
            ):
                self.committed[slot] = (generation, rid, p + n)

    def propose(self, batch, prefix_lens=None, *, target_only=False):
        worker = self.worker
        idle = batch.forward_mode.is_idle()
        rec = self._descriptor(batch, [] if idle else prefix_lens.cpu().tolist())
        rec["bonuses"] = [] if idle else batch.spec_info.bonus_tokens.cpu().tolist()
        records = self._exchange("PLAN", rec)
        if any(not x["greedy"] for x in records):
            raise ValueError(
                "Experimental DFlash cohorts currently accept greedy requests only"
            )
        for owner, item in enumerate(records):
            self._validate_prefixes(owner, item)
        counts = [len(x["slots"]) for x in records]
        count = sum(counts)
        block = 1 if target_only else worker.block_size
        self.pending = dict(records=records, counts=counts, block=block)
        if not count:
            return torch.empty((0, block), dtype=torch.int64, device=worker.device)
        device = worker.device
        prefixes = torch.tensor(
            [p for x in records for p in x["prefixes"]],
            dtype=torch.int64,
            device=device,
        )
        slots = torch.tensor(
            [s for o, x in enumerate(records) for s in self._slots(o, x)],
            dtype=torch.int64,
            device=device,
        )
        anchors = torch.tensor(
            [t for x in records for t in x["bonuses"]], dtype=torch.int64, device=device
        )
        positions = (
            prefixes[:, None] + torch.arange(block, device=device)[None, :]
        ).reshape(-1)
        full, window, _ = worker._independent_cache.locations(
            slots,
            positions,
            torch.full((count,), block, device=device, dtype=torch.int32),
        )
        if target_only:
            if any(
                p + 1 > worker._independent_cache.layout.context_length
                for x in records
                for p in x["prefixes"]
            ):
                raise RuntimeError(
                    "Target-only step exceeds the model context boundary"
                )
            self.pending.update(
                slots=slots, positions=positions, full=full, window=window
            )
            start = sum(counts[: self.owner_rank])
            return anchors[start : start + counts[self.owner_rank], None].contiguous()
        token_ids = torch.full(
            (count, block), worker._mask_token_id, dtype=torch.int64, device=device
        )
        token_ids[:, 0] = anchors
        embedding = worker._full_embed_gpu
        if (
            embedding is None
            or embedding.shape[0] < worker.model_runner.model_config.vocab_size
        ):
            raise RuntimeError("Draft cohort requires a complete target embedding view")
        embeds = torch.nn.functional.embedding(token_ids, embedding).reshape(
            count * block, -1
        )
        if worker._noise_embed_scale != 1.0:
            # Match _draft_block_embeds, including the anchor embedding.
            embeds = embeds * worker._noise_embed_scale
        host_lengths = torch.tensor(
            [p + block for x in records for p in x["prefixes"]], dtype=torch.int64
        )
        plan = worker.draft_model_runner.runner_ids.plan(
            req_pool_indices=slots,
            seq_lens=prefixes,
            seq_lens_cpu=host_lengths,
            write_virtual=full,
            read_extent=block,
        )
        forward = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=count,
            input_ids=token_ids.reshape(-1),
            req_pool_indices=slots,
            seq_lens=prefixes,
            out_cache_loc=full,
            seq_lens_sum=int(host_lengths.sum()),
            seq_lens_cpu=host_lengths,
            positions=positions,
            input_embeds=embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=worker._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=(
                torch.full((), count * block, dtype=torch.int32, device=device)
                if enable_num_token_non_padded()
                else None
            ),
            global_num_token_non_padded_cpu=count * block,
            can_run_decode_cuda_graph=False,
        )
        plan.bind(forward, worker.draft_model_runner.runner_ids)
        with torch.inference_mode(), draft_tp_context(worker.draft_parallel_layout):
            result = worker.draft_model_runner.forward(forward)
        next_ids = torch.empty((count, block - 1), dtype=torch.int64, device=device)
        worker._selector_sample = None
        if self.group.rank_in_group == 0:
            # This call intentionally uses the original target scope/head group.
            next_ids.copy_(
                worker._propose_selector_block(
                    draft_logits_output=result.logits_output,
                    bs=count,
                    lm_head=self.head,
                    anchor_token_ids=anchors,
                    sampling_info=None,
                )
            )
        self.group.broadcast(next_ids, src=0)
        proposals = torch.cat((anchors[:, None], next_ids), dim=1)
        self.pending.update(slots=slots, positions=positions, full=full, window=window)
        start = sum(counts[: self.owner_rank])
        return proposals[start : start + counts[self.owner_rank]].contiguous()

    def commit(self, hidden, commit_lens):
        if self.pending is None:
            raise RuntimeError("DFlash cohort COMMIT without PLAN")
        worker = self.worker
        counts = self.pending["counts"]
        local = counts[self.owner_rank]
        lengths = [] if local == 0 else commit_lens.cpu().tolist()
        records = self._exchange("COMMIT", lengths)
        if any(len(lengths) != count for lengths, count in zip(records, counts)):
            raise RuntimeError("Draft cohort COMMIT owner count mismatch")
        block = self.pending["block"]
        width = worker.draft_model.num_context_features * int(
            worker.draft_model.config.hidden_size
        )
        offset = 0
        for owner, (count, accepted) in enumerate(zip(counts, records)):
            if not count:
                continue
            if any(n < 0 or n > block for n in accepted):
                raise RuntimeError("Draft cohort invalid commit length")
            rows = hidden if owner == self.owner_rank else None
            rows = self._payload(owner, rows, (count * block, width), torch.bfloat16)
            sl = slice(offset * block, (offset + count) * block)
            full = self.pending["full"][sl]
            window = self.pending["window"]
            worker._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=rows,
                cache_loc=full,
                cache_loc_2d=full.view(count, block),
                positions=self.pending["positions"][sl],
                commit_lens=torch.tensor(
                    accepted, dtype=torch.int32, device=worker.device
                ),
                swa_cache_loc=window[sl] if window is not None else None,
            )
            desc = self.pending["records"][owner]
            for slot, generation, rid, prefix, n in zip(
                self._slots(owner, desc),
                desc["generations"],
                desc["ids"],
                desc["prefixes"],
                accepted,
            ):
                self.committed[slot] = (generation, rid, prefix + n)
            offset += count
        self.pending = None
