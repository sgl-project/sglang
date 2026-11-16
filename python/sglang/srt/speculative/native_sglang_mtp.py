"""Native MTP with token-aligned private KV and request-private pending anchors."""

import json
import os
from copy import copy, deepcopy
from dataclasses import replace
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dmtp_runtime import mtp_config, scoped_config
from sglang.srt.speculative.mtp_append_pack import scatter_commit_mapping
from sglang.srt.speculative.mtp_hidden_only import HiddenOnlyLogitsProcessor
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class NativeSGLangMTP:
    def __init__(self, owner):
        self.graph_requested = os.getenv("SGLANG_MTP_PREFIX_MTP_GRAPH", "0") == "1"
        self.graph_implementation = os.getenv(
            "SGLANG_MTP_PREFIX_MTP_GRAPH_IMPL", "native_bridge"
        )
        if self.graph_requested and self.graph_implementation != "native_bridge":
            raise ValueError("The upstream port supports only native_bridge MTP graphs")
        self.bridge = None
        self.update_graphs = None
        self.last_graph_used = False
        self.graph_audit_count = 0
        self.owner = owner
        args = deepcopy(owner.server_args)
        declare_resolution(args, "dmtp_private_mtp", **mtp_config(owner))
        with scoped_config(**mtp_config(owner)):
            self.worker = TpModelWorker(
                server_args=args,
                gpu_id=owner.gpu_id,
                nccl_port=owner.nccl_port,
                is_draft_worker=True,
                context_length=owner.model_runner.model_config.context_len,
                draft_attention_backend="fa3",
                random_seed=owner.target_worker.random_seed,
            )
        target = owner.target_worker.model_runner.model
        embed, head = target.get_embed_and_head()
        self.worker.model_runner.model.set_embed_and_head(embed, head)
        model = self.worker.model_runner.model
        self.hidden_only = HiddenOnlyLogitsProcessor(
            model.logits_processor,
            enabled=os.getenv("SGLANG_MTP_PREFIX_SKIP_UNUSED_LOGITS", "1") == "1",
        )
        model.logits_processor = self.hidden_only
        self.audit_path = os.getenv("SGLANG_MTP_PREFIX_AUDIT_LOGITS", "")
        self.audit_count = 0

    def alloc_memory_pool(self, config, mapping, allocator):
        self.token_aligned_kv = True
        self.target_mapping = mapping
        self.pending_positions = {}
        self.pending_base = max(
            config.max_total_num_tokens, config.full_max_total_num_tokens or 0
        ) + max(int(self.owner.server_args.page_size), 1)
        capacity = self.pending_base + mapping.req_to_token.shape[0] * (
            8 if getattr(self, "graph_requested", False) else 1
        )
        native_config = replace(
            config,
            max_total_num_tokens=capacity,
            full_max_total_num_tokens=(
                capacity if config.full_max_total_num_tokens is not None else None
            ),
        )
        # Preserve hybrid-pool scheduler metadata; only the native attention map
        # differs. Native MTP itself contains no recurrent GDN layer.
        native_mapping = copy(mapping)
        native_mapping.req_to_token = torch.zeros_like(mapping.req_to_token)
        with scoped_config(**mtp_config(self.owner)):
            self.worker.alloc_memory_pool(
                memory_pool_config=native_config,
                req_to_token_pool=native_mapping,
                token_to_kv_pool_allocator=allocator,
            )
        if (
            self.worker.model_runner.token_to_kv_pool
            is self.owner.model_runner.token_to_kv_pool
        ):
            raise RuntimeError(
                "MTP must have private KV tensors, not target KV aliases"
            )

    def init_attention_backends(self):
        with scoped_config(**mtp_config(self.owner)):
            self.worker.init_attention_backends()

    def init_cuda_graphs(self):
        with scoped_config(**mtp_config(self.owner)):
            self.worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        if self.graph_requested:
            if not self.hidden_only.enabled:
                raise ValueError("MTP update graph requires hidden-only native output")
            from sglang.srt.speculative.native_mtp_bridge import NativeMTPBridge

            with scoped_config(**mtp_config(self.owner)):
                self.bridge = NativeMTPBridge(self)
            self.update_graphs = self.bridge

    def forward(self, batch):
        runner = self.worker.model_runner
        plan = runner.runner_ids.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            write_virtual=None,
            write_slots=batch.out_cache_loc,
        )
        plan.bind(batch, runner.runner_ids)
        with scoped_config(**mtp_config(self.owner)):
            return runner.forward(batch).logits_output

    @torch.no_grad()
    def append(
        self,
        tokens,
        preceding_hidden,
        req_indices,
        cache_locations,
        prefix_length,
        lengths=None,
        *,
        continuation=False,
        prepared=None,
        request_ids=None
    ):
        count = tokens.numel()
        if (
            count < 1
            or preceding_hidden.shape[0] != count
            or cache_locations.numel() != count
        ):
            raise ValueError("Native MTP append rows are misaligned")
        prefixes = (
            [prefix_length]
            if isinstance(prefix_length, int)
            else [int(x) for x in prefix_length]
        )
        lengths = [count] if lengths is None else [int(x) for x in lengths]
        bs = req_indices.numel()
        if (
            len(prefixes) != bs
            or len(lengths) != bs
            or sum(lengths) != count
            or any(x < 0 for x in prefixes)
            or any(x < 1 for x in lengths)
        ):
            raise ValueError("Native MTP ragged batch metadata is misaligned")
        device = tokens.device
        totals = [p + n for p, n in zip(prefixes, lengths)]
        if prepared is not None:
            if not continuation or not getattr(self, "token_aligned_kv", False):
                raise ValueError(
                    "Prepared MTP append requires a token-aligned continuation"
                )
            reqs = req_indices.tolist() if request_ids is None else request_ids
            if len(reqs) != bs:
                raise ValueError("MTP continuation request metadata is misaligned")
            own_map = self.worker.model_runner.req_to_token_pool.req_to_token
            if any(p + n > own_map.shape[1] for p, n in zip(prefixes, lengths)):
                raise ValueError(
                    "MTP continuation exceeds its request mapping capacity"
                )
            for req, p in zip(reqs, prefixes):
                if self.pending_positions.get(req) != p:
                    raise ValueError(
                        "MTP continuation does not match its pending anchor"
                    )
            pool = self.worker.model_runner.token_to_kv_pool
            keys, values = pool.get_kv_buffer(pool.start_layer)
            keys.index_copy_(
                0, prepared.anchor_locations, keys[prepared.pending_locations]
            )
            values.index_copy_(
                0, prepared.anchor_locations, values[prepared.pending_locations]
            )
            scatter_commit_mapping(req_indices, prepared, own_map)
            self.pending_positions.update(zip(reqs, totals))
            positions = prepared.positions
            prefix_gpu, lens, starts = prepared.metadata.unbind(0)
            cache_locations = prepared.native_locations
        else:
            positions = torch.cat(
                [
                    torch.arange(p + 1, p + n + 1, dtype=torch.int64, device=device)
                    for p, n in zip(prefixes, lengths)
                ]
            )
            lens = torch.tensor(lengths, dtype=torch.int32, device=device)
            prefix_gpu = torch.tensor(prefixes, dtype=torch.int32, device=device)
            starts = torch.cumsum(lens, 0, dtype=torch.int32) - lens
            if getattr(self, "token_aligned_kv", False):
                cache_locations = self._token_aligned_locations(
                    req_indices, cache_locations, prefixes, lengths, continuation
                )
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=tokens,
            req_pool_indices=req_indices,
            seq_lens=torch.tensor(totals, device=device),
            out_cache_loc=cache_locations,
            seq_lens_sum=sum(totals),
            seq_lens_cpu=torch.tensor(totals, dtype=torch.int32),
            positions=positions,
            extend_num_tokens=count,
            extend_seq_lens=lens,
            extend_prefix_lens=prefix_gpu,
            extend_start_loc=starts,
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens_cpu=prefixes,
            extend_logprob_start_lens_cpu=[x - 1 for x in lengths],
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            spec_info=SimpleNamespace(
                hidden_states=preceding_hidden, custom_mask=None, kv_indptr=None
            ),
            capture_hidden_mode=CaptureHiddenMode.FULL,
            mrope_positions=positions[None].expand(3, -1).contiguous(),
        )
        audit = getattr(self, "audit_path", "") and self.audit_count < 4
        desired_mode = self.hidden_only.enabled if audit else None
        if audit:
            self.hidden_only.enabled = False
        graphs = getattr(self, "update_graphs", None)
        graph_hidden = (
            graphs.run(batch, prefixes, lengths) if continuation and graphs else None
        )
        self.last_graph_used = graph_hidden is not None
        result = (
            SimpleNamespace(hidden_states=graph_hidden)
            if self.last_graph_used
            else self.forward(batch)
        )
        if self.last_graph_used and self._graph_check_required(graphs, lengths):
            pool = self.worker.model_runner.token_to_kv_pool
            keys, values = pool.get_kv_buffer(pool.start_layer)
            graph_keys, graph_values = (
                keys[cache_locations.long()].clone(),
                values[cache_locations.long()].clone(),
            )
            # Compare graph execution against eager execution of the SAME fixed
            # layout. Packed-layout numerical differences are audited separately.
            reference = graphs.eager_same_layout(batch)
            match = (
                torch.eq(result.hidden_states, reference).all()
                & torch.eq(keys[cache_locations.long()], graph_keys).all()
                & torch.eq(values[cache_locations.long()], graph_values).all()
            )
            failed = get_tp_group().all_reduce((~match).to(torch.int32).reshape(1))
            if failed.item():
                raise RuntimeError(
                    "Fixed8 MTP graph hidden/KV differs from same-layout eager"
                )
            graphs.verified_batches.add(bs)
            self.graph_audit_count += 1
        if result.hidden_states is None or result.hidden_states.shape[0] != count:
            raise RuntimeError("Native MTP must return its full normalized hidden rows")
        if audit:
            original = result.hidden_states.clone()
            pool = self.worker.model_runner.token_to_kv_pool
            keys, values = pool.get_kv_buffer(pool.start_layer)
            before_k, before_v = (
                keys[cache_locations.long()].clone(),
                values[cache_locations.long()].clone(),
            )
            try:
                self.hidden_only.enabled = True
                checked = self.forward(batch).hidden_states
                torch.testing.assert_close(checked, original, rtol=0, atol=0)
                torch.testing.assert_close(
                    keys[cache_locations.long()], before_k, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    values[cache_locations.long()], before_v, rtol=0, atol=0
                )
                with open(self.audit_path, "a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "rows": count,
                                "requests": bs,
                                "prefixes": prefixes,
                                "hidden_exact": True,
                                "kv_exact": True,
                            }
                        )
                        + "\n"
                    )
                self.audit_count += 1
            finally:
                self.hidden_only.enabled = desired_mode
        return result.hidden_states

    def _graph_check_required(self, graphs, lengths):
        if os.getenv("SGLANG_MTP_PREFIX_MTP_GRAPH_CHECK", "0") == "1":
            return True
        return len(lengths) not in graphs.verified_batches

    def _token_aligned_locations(
        self, req_indices, old_locations, prefixes, lengths, continuation
    ):
        own = self.worker.model_runner.req_to_token_pool.req_to_token
        target = self.target_mapping.req_to_token
        reqs = req_indices.tolist()
        starts, start = [], 0
        for n in lengths:
            starts.append(start)
            start += n
        if continuation:
            for req, p in zip(reqs, prefixes):
                if self.pending_positions.get(req) != p:
                    raise ValueError(
                        "MTP continuation does not match its pending anchor"
                    )
            # The target has just consumed the previous pending anchor. Its native
            # KV moves bitwise to the committed token slot before the next append.
            src = req_indices.long() + self.pending_base
            dst = old_locations[
                torch.tensor(starts, device=old_locations.device)
            ].long()
            pool = self.worker.model_runner.token_to_kv_pool
            keys, values = pool.get_kv_buffer(pool.start_layer)
            keys.index_copy_(0, dst, keys[src])
            values.index_copy_(0, dst, values[src])
            own[req_indices.long(), torch.tensor(prefixes, device=own.device) - 1] = (
                dst.to(own.dtype)
            )
        else:
            for req, p in zip(reqs, prefixes):
                own[req, :p] = target[req, 1 : p + 1]
        locations = []
        for req, p, n, start in zip(reqs, prefixes, lengths, starts):
            pending = torch.tensor(
                [self.pending_base + req],
                device=old_locations.device,
                dtype=old_locations.dtype,
            )
            slots = torch.cat((old_locations[start + 1 : start + n], pending))
            own[req, p : p + n] = slots.to(own.dtype)
            locations.append(slots)
            self.pending_positions[req] = p + n
        return torch.cat(locations)
