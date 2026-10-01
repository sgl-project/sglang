"""Incremental, committed-prefix projection from target KV into draft KV."""

from __future__ import annotations

from pathlib import Path

import msgspec
import torch
from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    validate_target_kv_draft_contract,
)
from sglang.srt.training_capture.identity import (
    bind_rank_target_contract,
    bind_target_contract,
    local_safetensors_digest,
)
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.startup import (
    coordinate_policy_startup,
    coordinate_target_startup,
)


class TargetKVSourceRange(msgspec.Struct, frozen=True, kw_only=True):
    start: int
    cache_locs: torch.Tensor
    positions: torch.Tensor


class ProjectedContextState(msgspec.Struct, frozen=True):
    weight_version: str
    end: int


class TargetKVInjector:
    def __init__(self, *, draft_model, draft_model_runner, model_runner):
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.model_runner = model_runner
        self.sources = None
        self.weights_digest = None
        self.epoch = 0
        self.projected_token_ct = 0
        self.invalidation_ct = 0
        self.tp_group = None
        self.head_replicas = {}

    @property
    def weight_version(self):
        if self.weights_digest is None:
            raise ContractError(
                "target KV injector has not been bound to loaded weights"
            )
        return f"{self.weights_digest}:{self.epoch}"

    def bind(self, *, tokenizer_path, prediction_count, mask_token_id):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        contract = self.draft_model.target_kv_contract
        group = self.model_runner.tp_group
        self.tp_group = group if group.world_size > 1 else None
        target_args = {
            "model_id": contract.teacher.model_id,
            "selected_layer_ids": contract.kv.selected_layer_ids,
            "storage_chunk_tokens": contract.kv.storage_chunk_tokens,
            "model": self.model_runner.model,
            "model_config": self.model_runner.model_config,
            "tokenizer_path": tokenizer_path,
            "pool": self.model_runner.token_to_kv_pool,
            "expected_weights_revision": contract.teacher.weights_revision,
            "expected_tokenizer_revision": contract.teacher.tokenizer_revision,
        }
        if self.tp_group is None:
            teacher, kv = bind_target_contract(**target_args)
        else:
            teacher, kv, _ = coordinate_target_startup(
                group=group.cpu_group,
                build_local=lambda: bind_rank_target_contract(
                    **target_args,
                    tp_rank=group.rank_in_group,
                    tp_size=group.world_size,
                    pp_rank=0,
                    pp_size=1,
                ),
                tp_size=group.world_size,
                pp_size=1,
            )

        def validate_draft():
            pool = self.draft_model_runner.token_to_kv_pool
            if (
                not isinstance(pool, MHATokenToKVPool)
                or pool.is_quantized_kv_cache
                or pool.use_hnd
            ):
                raise ContractError(
                    "target-KV DSpark requires an unquantized NHD draft pool"
                )
            validate_target_kv_draft_contract(
                target=teacher,
                draft=contract,
                pool_codec=kv,
                target_hidden_size=self.model_runner.model_config.hf_text_config.hidden_size,
                prediction_count=prediction_count,
                mask_token_id=mask_token_id,
            )
            self.weights_digest = local_safetensors_digest(
                Path(self.draft_model_runner.model_config.model_path)
            )
            return contract, self.weights_digest

        if self.tp_group is None:
            validate_draft()
        else:
            coordinate_policy_startup(
                group=group.cpu_group, build_policy=validate_draft
            )
        pool = self.model_runner.token_to_kv_pool
        self.sources = {}
        for layer in kv.layers:
            for component, buffer in (
                ("k", pool.get_key_buffer(layer.layer_id)),
                ("v", pool.get_value_buffer(layer.layer_id)),
            ):
                name = f"target_{component}.{layer.layer_id}"
                self.sources[name] = buffer
                self.head_replicas[name] = max(
                    1, group.world_size // layer.num_kv_heads
                )

    def _select_target_kv(self, locations):
        selected = {}
        for name, buffer in self.sources.items():
            values = buffer.index_select(0, locations)
            if self.tp_group is not None:
                values = self.tp_group.all_gather(values, dim=1)
                # TP can replicate a logical KV head on adjacent ranks. Keep
                # the canonical copy before the encoder flattens global heads.
                values = values[:, :: self.head_replicas[name], :].contiguous()
            selected[name] = values
        return selected

    def invalidate_projected_context(self, req, *, reason, new_weight_version):
        if not reason or new_weight_version != self.weight_version:
            raise ContractError("projected KV invalidation requires reason and version")
        req.dspark_projected_context = None
        self.invalidation_ct += 1

    def invalidate_all(self):
        self.epoch += 1

    @torch.no_grad()
    def inject_target_kv(
        self, req, *, committed_prefix_end, source_ranges, draft_weight_version
    ):
        if self.sources is None:
            raise ContractError("target KV source buffers have not been bound")
        if draft_weight_version != self.weight_version:
            raise ContractError("projected KV write uses a stale draft weight version")
        state = req.dspark_projected_context
        start = (
            state.end
            if state is not None and state.weight_version == self.weight_version
            else 0
        )
        cursor = start
        slots, positions = [], []
        for part in source_ranges:
            if (
                part.start != cursor
                or part.cache_locs.ndim != 1
                or part.positions.shape != part.cache_locs.shape
                or part.cache_locs.numel() == 0
                or part.cache_locs.dtype not in (torch.int32, torch.int64)
                or part.positions.dtype not in (torch.int32, torch.int64)
                or part.positions.device != part.cache_locs.device
            ):
                raise ContractError("target KV source ranges contain a gap or overlap")
            cursor += part.cache_locs.numel()
            slots.append(part.cache_locs)
            positions.append(part.positions)
        if cursor != committed_prefix_end or cursor < start:
            raise ContractError("target KV source ranges differ from committed prefix")
        if not slots:
            return None
        locations = torch.cat(slots).long()
        actual_positions = torch.cat(positions).long()
        # Bound temporary feature storage even when the entire target prefix
        # came from radix/HiCache and needs its first draft projection.
        for offset in range(0, locations.numel(), 1024):
            chunk_locs = locations[offset : offset + 1024]
            selected = self._select_target_kv(chunk_locs)
            self.draft_model.write_target_kv(
                target_kv=selected,
                pool=self.draft_model_runner.token_to_kv_pool,
                positions=actual_positions[offset : offset + 1024],
                cache_loc=chunk_locs,
            )
        event = None
        if locations.is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(locations.device))
        req.dspark_projected_context = ProjectedContextState(
            self.weight_version, cursor
        )
        self.projected_token_ct += cursor - start
        return event

    def ensure_context(self, batch):
        ends = batch.seq_lens_cpu.tolist()
        mapping = self.model_runner.req_to_token_pool.req_to_token
        for req, end in zip(batch.reqs, ends, strict=True):
            state = req.dspark_projected_context
            if state is not None and (
                state.weight_version != self.weight_version or state.end > end
            ):
                self.invalidate_projected_context(
                    req,
                    reason="version_changed_or_retracted",
                    new_weight_version=self.weight_version,
                )
                state = None
            start = state.end if state is not None else 0
            if start == end:
                continue
            slots = mapping[req.req_pool_idx, start:end]
            self.inject_target_kv(
                req,
                committed_prefix_end=end,
                source_ranges=(
                    TargetKVSourceRange(
                        start=start,
                        cache_locs=slots,
                        positions=torch.arange(start, end, device=slots.device),
                    ),
                ),
                draft_weight_version=self.weight_version,
            )

    def inject_verify(self, *, batch, verify_window, commit_lens):
        prefix_lens = batch.seq_lens_cpu.tolist()
        num_commit = commit_lens.cpu().tolist()
        for row, (req, start, count) in enumerate(
            zip(batch.reqs, prefix_lens, num_commit, strict=True)
        ):
            if not 0 < count <= verify_window.verify_cache_loc_2d.shape[1]:
                raise ContractError("invalid committed DSpark verify length")
            # The next bonus token has no target KV yet. commit_lens covers the
            # forwarded anchor and correct drafts, ending before that token.
            self.inject_target_kv(
                req,
                committed_prefix_end=start + count,
                source_ranges=(
                    TargetKVSourceRange(
                        start=start,
                        cache_locs=verify_window.verify_cache_loc_2d[row, :count],
                        positions=verify_window.positions_2d[row, :count],
                    ),
                ),
                draft_weight_version=self.weight_version,
            )
