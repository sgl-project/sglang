"""TP4 synchronous batched MTP-first controller with per-request state."""

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from sglang.srt.speculative.mtp_prefix_mask import build_full_prefix_mask
from sglang.srt.speculative.native_sglang_mtp import NativeSGLangMTP


@dataclass
class RequestState:
    hidden: torch.Tensor
    anchor: int
    token: int
    serial: int
    round: int = 0
    features: torch.Tensor = None


class MTPFirstController:
    @staticmethod
    def configure_attention(backend):
        if getattr(backend, "fa_impl_ver", None) == 3:
            from sglang.srt.speculative.mtp_anchor_fa3 import AnchorFA3Forward

            backend.mtp_anchor_fa3 = AnchorFA3Forward(backend)
            backend.forward_extend = backend.mtp_anchor_fa3
            return
        if getattr(backend, "enable_deterministic", False):
            raise ValueError(
                "MTP-first requires the explicit-mask two-stage draft backend"
            )
        # Ordinary speculative attention assumes all cached prefix rows are visible.
        backend.extend_attention_fwd = partial(
            backend.extend_attention_fwd, skip_prefix_custom_mask=False
        )
        if os.getenv("SGLANG_MTP_PREFIX_ANCHOR_WINDOW_GRAPH", "0") == "1":
            from sglang.srt.speculative.mtp_anchor_window import AnchorWindowForward

            backend.extend_attention_fwd = AnchorWindowForward(
                backend.extend_attention_fwd
            )

    def __init__(self, worker):
        args = worker.server_args
        if (
            args.tp_size != 4
            or not 1 <= args.max_running_requests <= 32
            or args.quantization is not None
            or worker.use_compact_draft_cache
        ):
            raise ValueError(
                "MTP-first requires TP4, 1..32 requests, "
                "no quantization; compact draft KV parity is not yet validated"
            )
        if args.speculative_draft_attention_backend != "fa3":
            raise ValueError("FA3 alignment runtime requires FA3 draft attention")
        if float(os.getenv("SGLANG_SIMULATE_ACC_LEN", "0")) > 0:
            raise ValueError("MTP-first must not use mock acceptance")
        self.draft_slots = 8
        self.worker = worker
        self.opt_sync = os.getenv("SGLANG_MTP_PREFIX_OPT_SYNC", "1") == "1"
        self.fused_mask = os.getenv("SGLANG_MTP_PREFIX_FUSED_MASK", "1") == "1"
        self.fused_append = os.getenv("SGLANG_MTP_PREFIX_FUSED_APPEND", "1") == "1"
        self.target_only = False
        self.native_selector = True
        target = worker.target_worker.model_runner.model
        config = worker.target_worker.model_runner.model_config.hf_config
        from sglang.srt.speculative.dmtp_runtime import validate_mtp_target

        validate_mtp_target(config)
        self.vocab_size = getattr(config, "text_config", config).vocab_size
        self.head = target.lm_head
        self.generator = torch.Generator(device=self.head.weight.device).manual_seed(
            args.random_seed + 101
        )
        self.native = NativeSGLangMTP(worker)
        self.final_hidden = None
        # This allocation must not move after graph capture. The copy in the
        # norm hook becomes a graph node; Python hooks are not called on replay.
        self.final_buffer = torch.empty(
            (
                max(
                    worker.model_runner.model_config.context_len,
                    getattr(args, "max_prefill_tokens", 0) or 0,
                    args.max_running_requests * 8,
                    32768,
                ),
                (
                    config.text_config.hidden_size
                    if hasattr(config, "text_config")
                    else config.hidden_size
                ),
            ),
            dtype=self.head.weight.dtype,
            device=self.head.weight.device,
        )
        self.target_hidden_cache = None
        worker.draft_model.mtp_prefix_graph_preconditioned = True
        target.model.norm.register_forward_hook(self._capture_final)
        self.states = {}
        self.request = 0
        self.trace_dir = os.getenv("SGLANG_MTP_PREFIX_TRACE_DIR")
        self.batch_trace = os.getenv("SGLANG_MTP_PREFIX_BATCH_TRACE")
        if self.trace_dir:
            Path(self.trace_dir).mkdir(parents=True, exist_ok=True)

    def init_graphs(self):
        if self.native.bridge is None:
            raise ValueError("T1 residual requires the native MTP bridge")
        from sglang.srt.speculative.dmtp_selector_graph import DMTPSelectorGraphs

        enabled = self.worker.draft_model_runner.decode_cuda_graph_runner is not None
        self.t1_selector = DMTPSelectorGraphs(self, t1=True, enable_graph=enabled)
        self.tail_selector = DMTPSelectorGraphs(self, t1=False, enable_graph=enabled)
        self.conditioner_graphs = None
        if os.getenv("SGLANG_MTP_PREFIX_CONDITIONER_GRAPH", "0") == "1":
            from sglang.srt.speculative.mtp_conditioner_graph import ConditionerGraphs

            module = self.worker.draft_model.mtp_prefix_conditioner
            self.conditioner_graphs = ConditionerGraphs(
                module, self.worker.server_args.max_running_requests, owner=self
            )
            module.forward = self.conditioner_graphs.run
        self.t1_combined = None
        if os.getenv("SGLANG_MTP_PREFIX_HEAD_CONDITIONER_GRAPH", "0") == "1":
            if self.conditioner_graphs is None:
                raise ValueError(
                    "Sampled T1 combination requires the original conditioner graph"
                )
            from sglang.srt.speculative.dmtp_head_conditioner_graph import (
                SampledHeadConditionerGraphs,
            )

            self.t1_combined = SampledHeadConditionerGraphs(self, enable_graph=enabled)
            if os.getenv("SGLANG_DMTP_COMBINED_VALIDATE_DIR"):
                from sglang.srt.speculative.dmtp_combined_validation import (
                    validate_combined,
                )

                validate_combined(self, os.environ["SGLANG_DMTP_COMBINED_VALIDATE_DIR"])

    def prepare_t1(self, block_ids):
        self.prepared_embeds = None
        combined = getattr(self, "t1_combined", None)
        if combined is not None:
            result = combined.run(self.hidden, block_ids, self.selector_sampling)
            if result is not None:
                tokens, ids, q, self.prepared_embeds = result
                block_ids[:, 1].copy_(tokens[:, 0])
                self.t1_sparse = (ids, q)
                self.graph_used.update(
                    t1_head=True, conditioner=True, t1_combined=combined.last_used
                )
                return
        tokens, ids, q = self.t1_selector.run(
            self.hidden[:, None], block_ids[:, 0], self.selector_sampling
        )
        block_ids[:, 1] = tokens[:, 0]
        self.t1_sparse = (ids, q)
        self.graph_used["t1_head"] = self.t1_selector.last_used

    def _capture_final(self, module, inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        if value.shape[0] > self.final_buffer.shape[0]:
            raise ValueError("Target norm exceeds graph-stable hidden capacity")
        self.final_buffer[: value.shape[0]].copy_(value)

    def target_hidden(self, count):
        buffer = getattr(self, "final_buffer", None)
        return (buffer if buffer is not None else self.final_hidden)[:count]

    def alloc_hidden_cache(self):
        pool = self.native.worker.model_runner.token_to_kv_pool
        rows = pool.get_key_buffer(pool.start_layer).shape[0]
        self.target_hidden_cache = torch.empty(
            (rows, self.final_buffer.shape[1]),
            device=self.final_buffer.device,
            dtype=self.final_buffer.dtype,
        )

    def validate_sampling(self, batch):
        info = batch.sampling_info
        if info is None:
            raise ValueError("MTP-first needs explicit sampling info")
        penalizer = info.penalizer_orchestrator
        if (
            info.need_min_p_sampling
            or info.has_custom_logit_processor
            or info.logit_bias is not None
            or info.grammars
            or info.grammar_mask is not None
            or (penalizer is not None and penalizer.is_required)
        ):
            raise ValueError(
                "MTP-first supports temperature/top-k/top-p without penalties/grammar"
            )
        if getattr(self, "native_selector", False):
            self.selector_sampling = info
            return None
        if getattr(self, "opt_sync", False):
            if getattr(self, "target_only", False) and not getattr(
                self, "trace_dir", None
            ):
                # Target-only sampling reads the original SamplingBatchInfo in
                # the base verifier. No draft distribution needs these values.
                return None
            temperatures = (
                [0.0] * len(info.top_ks)
                if info.is_all_greedy
                else info.temperatures.flatten().tolist()
            )
            return list(zip(temperatures, info.top_ks.tolist(), info.top_ps.tolist()))
        return [
            (
                0.0 if info.is_all_greedy else float(info.temperatures.flatten()[i]),
                int(info.top_ks[i]),
                float(info.top_ps[i]),
            )
            for i in range(batch.req_pool_indices.numel())
        ]

    @torch.no_grad()
    def prefill(self, batch, anchor, features):
        lengths = [int(x) for x in batch.extend_lens]
        prefixes = [int(x) for x in batch.prefix_lens]
        final = self.target_hidden(sum(lengths))
        if sum(lengths) != batch.input_ids.numel() or final.shape[0] != sum(lengths):
            raise ValueError("Target/MTP prefill row alignment mismatch")
        cache = getattr(self, "target_hidden_cache", None)
        if cache is not None:
            cache.index_copy_(0, batch.out_cache_loc.long(), final)
        anchors = anchor.flatten()
        tokens, hidden_rows, slots, native_prefixes, native_lengths, start = (
            [],
            [],
            [],
            [],
            [],
            0,
        )
        for i, (n, prefix) in enumerate(zip(lengths, prefixes)):
            if prefix:
                if cache is None:
                    raise ValueError(
                        "Prefix reuse requires target normalized-hidden cache"
                    )
                req = int(batch.req_pool_indices[i])
                mapping = getattr(
                    self.native,
                    "target_mapping",
                    self.native.worker.model_runner.req_to_token_pool,
                ).req_to_token
                boundary_slot = mapping[req, prefix - 1 : prefix].long()
                tokens.append(
                    torch.cat((batch.input_ids[start : start + n], anchors[i : i + 1]))
                )
                hidden_rows.append(
                    torch.cat((cache[boundary_slot], final[start : start + n]))
                )
                slots.append(
                    torch.cat((boundary_slot, batch.out_cache_loc[start : start + n]))
                )
                native_prefixes.append(prefix - 1)
                native_lengths.append(n + 1)
            else:
                tokens.append(
                    torch.cat(
                        (batch.input_ids[start + 1 : start + n], anchors[i : i + 1])
                    )
                )
                hidden_rows.append(final[start : start + n])
                slots.append(batch.out_cache_loc[start : start + n])
                native_prefixes.append(0)
                native_lengths.append(n)
            start += n
        hidden = self.native.append(
            torch.cat(tokens),
            torch.cat(hidden_rows),
            batch.req_pool_indices,
            torch.cat(slots),
            native_prefixes,
            native_lengths,
        )
        start, native_start = 0, 0
        for i, (req, n) in enumerate(zip(batch.req_pool_indices.tolist(), lengths)):
            self.request += 1
            # Pool-slot reuse always starts a new state; stale proposal q is not retained.
            self.states[req] = RequestState(
                hidden[native_start + native_lengths[i] - 1].clone(),
                prefixes[i] + n,
                int(anchors[i]),
                self.request,
                features=(
                    features[start : start + n][-4096:].clone()
                    if self.trace_dir
                    else None
                ),
            )
            start += n
            native_start += native_lengths[i]
        if getattr(self.native, "bridge", None) is not None:
            tails = torch.tensor(
                [sum(native_lengths[: i + 1]) - 1 for i in range(len(lengths))],
                device=hidden.device,
                dtype=torch.int64,
            )
            anchor_rows = torch.tensor(
                [p + n for p, n in zip(prefixes, lengths)],
                device=hidden.device,
                dtype=torch.int64,
            )
            self.native.bridge.seed(
                batch.req_pool_indices,
                hidden.index_select(0, tails),
                anchor_rows,
                anchors,
            )
        self.final_hidden = None

    @torch.no_grad()
    def t1(self, batch, block_ids):
        self.prepared_embeds = None
        return self.native.bridge.t1(batch, block_ids)

    @torch.no_grad()
    def proposal(self, draft_hidden, block_ids, positions):
        rest, ids, q = self.tail_selector.run(
            draft_hidden[:, 2:], block_ids[:, 1], self.selector_sampling
        )
        tokens = torch.cat((block_ids[:, 1:2], rest), dim=1)
        if not self.selector_sampling.is_all_greedy:
            t1_ids, t1_q = self.t1_sparse
            self.worker._selector_sample = (
                torch.cat((t1_ids, ids), dim=1),
                torch.cat((t1_q, q), dim=1),
            )
        self.graph_used["tail_head"] = self.tail_selector.last_used
        self.t1_sparse = self.q1 = self.mixed = None
        return tokens

    @torch.no_grad()
    def commit(
        self,
        candidates,
        accept_len,
        bonus,
        target_features,
        next_positions,
        cache_locations,
    ):
        return self.native.bridge.commit(
            candidates,
            accept_len,
            bonus,
            target_features,
            next_positions,
            cache_locations,
        )

    def mask(self, seq_lens):
        if os.getenv("SGLANG_DMTP_FA3_SKIP_DENSE_MASK", "0") == "1":
            from sglang.srt.speculative.mtp_anchor_fa3 import AnchorFA3Forward

            backend = self.worker.draft_model_runner.attn_backend
            adapter = getattr(backend, "mtp_anchor_fa3", None)
            # Only this installed producer expresses the complete visibility
            # contract through last4096+block pages; generic masks still apply.
            if (
                isinstance(adapter, AnchorFA3Forward)
                and backend.forward_extend is adapter
                and adapter.backend is backend
                and adapter.slots == self.draft_slots
                and backend.fa_impl_ver == 3
                and backend.topk <= 1
                and backend.page_size == 1
            ):
                return None
        if getattr(self, "fused_mask", False) and seq_lens.is_cuda:
            return build_full_prefix_mask(seq_lens, slots=self.draft_slots)
        rows = []
        for length in seq_lens.tolist():
            keys = torch.arange(length + self.draft_slots, device=seq_lens.device)
            visible = ((keys >= max(0, length - 4096)) & (keys < length)) | (
                keys >= length
            )
            rows.append(visible[None].expand(self.draft_slots, -1).reshape(-1))
        return torch.cat(rows).contiguous()
