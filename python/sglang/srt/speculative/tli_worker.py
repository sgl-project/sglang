# SPDX-License-Identifier: Apache-2.0
"""
TLI (Token-Level Intersection) speculative decoding worker.

Implements lossless speculative decoding for target and draft models with
heterogeneous (overlapping but different) vocabularies.

Based on the ICML 2025 oral paper:
  "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for
   Heterogeneous Vocabularies" — Timor et al., https://arxiv.org/abs/2502.05202

Algorithm overview
------------------
1. At startup, build a normalized token intersection between target and draft
   vocabularies (see :class:`~sglang.srt.speculative.vocab_mapping.VocabMapping`).
2. Prompt / prefill phase: token IDs (in target vocab) are mapped to draft vocab
   before being fed into the draft model's KV-cache prefill.
3. Draft decode phase: logits from the draft model are constrained to the
   intersection (non-intersection logits → −∞).  The top-k tokens remain in
   draft vocab space and are fed as inputs for the next draft step, but stored
   in target vocab space for the verification tree.
4. Rejection sampling runs on the target model unchanged — the algorithm is
   provably lossless for tokens in the intersection.
"""

import contextlib
import logging
from typing import List, Optional

import torch
import torch.nn as nn

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
)
from sglang.srt.speculative.eagle_utils import (
    _eagle_prefill_tail_tokens,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
from sglang.srt.speculative.spec_utils import (
    fast_topk,
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.speculative.standalone_worker_v2 import StandaloneWorkerV2
from sglang.srt.speculative.vocab_mapping import VocabMapping
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

logger = logging.getLogger(__name__)


class _PrunedReindexLMHead(nn.Module):
    """Pruned draft LM head restricted to vocabulary-intersection tokens.

    Replaces the full ``[vocab_size, hidden_dim]`` weight matrix with a compact
    ``[intersection_size, hidden_dim]`` one.  The forward pass:
      1. Computes ``compact = hidden @ pruned_weight.T``  — ``O(B × K × H)`` instead
         of ``O(B × V × H)``.
      2. Scatters the K results back into a ``[B, V]`` tensor pre-filled with
         ``-inf``, so the output shape is identical to the original lm_head and
         no change is needed downstream.

    The ``-inf`` fill means that ``constrain_draft_logits`` in
    :class:`TLIWorker` becomes a no-op (already a no-op by construction).

    ``set_lora`` / ``apply_lora`` stubs are provided so that
    :func:`~sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head`
    routes through ``lm_head(hidden_states)`` rather than performing the matmul
    itself (which would bypass the scatter step).
    """

    def __init__(
        self,
        pruned_weight: torch.Tensor,
        pruned_bias: Optional[torch.Tensor],
        intersection_ids: torch.Tensor,
        full_vocab_size: int,
        use_fp32: bool = False,
    ):
        super().__init__()
        # registered as buffers (not parameters) to avoid unintended training
        self.register_buffer("weight", pruned_weight)
        if pruned_bias is not None:
            self.register_buffer("bias", pruned_bias)
        else:
            self.bias = None
        self.register_buffer("intersection_ids", intersection_ids)
        self.full_vocab_size = full_vocab_size
        self.use_fp32 = use_fp32

    # ── Stubs so _compute_lm_head routes through self(hidden_states) ──────────
    def set_lora(self, *args, **kwargs):
        pass

    def apply_lora(self, *args, **kwargs):
        pass

    # ── Core forward ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp32:
            compact = torch.matmul(x.to(torch.float32), self.weight.to(torch.float32).T)
        else:
            compact = torch.matmul(x.to(self.weight.dtype), self.weight.T)
        if self.bias is not None:
            compact = compact + self.bias
        # scatter into full-vocab output; non-intersection positions stay at -inf
        out = torch.full(
            (x.shape[0], self.full_vocab_size),
            float("-inf"),
            dtype=compact.dtype,
            device=x.device,
        )
        out[:, self.intersection_ids] = compact
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.weight.shape[1]}, "
            f"out_features={self.full_vocab_size}, "
            f"pruned_size={self.weight.shape[0]}"
        )


class TLIWorker(StandaloneWorkerV2):
    """Speculative decoding worker for heterogeneous-vocabulary draft models.

    Inherits the "separate draft model that does not share embeddings / lm_head
    with the target" boot-up logic from :class:`StandaloneWorkerV2`, then adds
    vocabulary mapping on top via :class:`VocabMapping`.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # StandaloneWorkerV2.__init__ creates the draft model without sharing
        # embed / lm_head with the target model — exactly what we need.
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

        # For TLI we never use the hot-token-map shortcut; vocab mapping
        # supersedes it entirely.
        self._draft_worker.hot_token_id = None

        # Bind TLI-specific overrides to the draft worker so the V2
        # delegation chain (EAGLEWorkerV2 -> _draft_worker.method()) picks
        # them up.
        import types

        self._draft_worker._draft_extend_for_prefill = types.MethodType(
            TLIWorker._tli_draft_extend_for_prefill, self
        )
        self._draft_worker._draft_extend_for_decode = types.MethodType(
            TLIWorker._tli_draft_extend_for_decode, self
        )
        self._draft_worker.draft_forward = types.MethodType(
            TLIWorker._tli_draft_forward, self
        )

        # ── Load tokenizers ──────────────────────────────────────────────────
        target_tokenizer_path = server_args.tokenizer_path or server_args.model_path
        draft_tokenizer_path = server_args.speculative_draft_model_path

        target_tokenizer = get_tokenizer(
            target_tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            tokenizer_revision=server_args.revision,
        )
        draft_tokenizer = get_tokenizer(
            draft_tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            tokenizer_revision=server_args.speculative_draft_model_revision,
        )

        # ── Get true vocab sizes from the model configs ───────────────────────
        target_vocab_size = target_worker.model_runner.model_config.vocab_size
        draft_vocab_size = self._draft_worker.draft_runner.model_config.vocab_size

        # ── Build vocabulary mapping ─────────────────────────────────────────
        self.vocab_mapping = VocabMapping(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            device=self.device,
        )

        # ── Prune the draft LM head to the intersection ──────────────────────
        # This happens before init_backends() is called externally, so CUDA
        # graphs will be captured with the pruned (smaller) LM head.
        self._try_prune_draft_lm_head(server_args)

    # ─────────────────────────────────────────────────────────────────────────
    # LM head pruning
    # ─────────────────────────────────────────────────────────────────────────

    def _try_prune_draft_lm_head(self, server_args) -> None:
        """Replace the draft LM head with a pruned version restricted to intersection tokens.

        This mirrors HuggingFace Transformers' ``_PruneReindexingLMHead``:
        the effective matmul shrinks from ``hidden_dim × draft_vocab_size`` to
        ``hidden_dim × intersection_size``, which is the dominant cost in draft
        decoding when the intersection is much smaller than the full vocabulary.

        The pruned head output has shape ``[batch, draft_vocab_size]`` with
        non-intersection positions pre-filled with ``-inf``, so all downstream
        code (including ``constrain_draft_logits`` and the logits buffer copy)
        remains unchanged.

        CUDA graph capture is deliberately deferred until *after* this method
        returns (see ``_defer_cuda_graphs`` in ``__init__``).  The captured
        graph therefore encodes the smaller ``[intersection_size, hidden_dim]``
        matmul rather than the full ``[draft_vocab_size, hidden_dim]`` one,
        including the scatter-back step that fills non-intersection positions
        with ``-inf``.
        """
        try:
            lm_head = self._draft_worker.draft_runner.model.lm_head
        except AttributeError:
            logger.warning(
                "Draft model has no '.lm_head' attribute; skipping LM head pruning. "
                "Set self._draft_worker.draft_runner.model.lm_head before calling TLIWorker.__init__ "
                "if you want pruning for this architecture."
            )
            return

        if not hasattr(lm_head, "weight"):
            logger.warning(
                "Draft LM head has no '.weight' attribute (quantized or unusual arch); "
                "skipping LM head pruning."
            )
            return

        intersection_ids = self.vocab_mapping.intersection_draft_ids  # [K]
        K = intersection_ids.shape[0]
        full_vocab_size = self.vocab_mapping.draft_vocab_size
        use_fp32 = getattr(server_args, "enable_fp32_lm_head", False)

        pruned_weight = lm_head.weight.data[intersection_ids].clone()
        pruned_bias = (
            lm_head.bias.data[intersection_ids].clone()
            if getattr(lm_head, "bias", None) is not None
            else None
        )

        pruned_head = _PrunedReindexLMHead(
            pruned_weight=pruned_weight,
            pruned_bias=pruned_bias,
            intersection_ids=intersection_ids,
            full_vocab_size=full_vocab_size,
            use_fp32=use_fp32,
        )
        self._draft_worker.draft_runner.model.lm_head = pruned_head
        logger.info(
            "Draft LM head pruned to intersection: %d → %d tokens (%.1f%% of draft vocab). "
            "Draft decode matmul reduced by %.1f×.",
            full_vocab_size,
            K,
            100.0 * K / max(full_vocab_size, 1),
            full_vocab_size / max(K, 1),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Draft extend for prefill
    # ─────────────────────────────────────────────────────────────────────────

    def _tli_draft_extend_for_prefill(
        self,
        batch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds=None,
    ):
        """Run draft-model KV-cache prefill with target→draft token ID mapping.

        Mirrors EagleDraftWorker._draft_extend_for_prefill but maps token IDs
        through the vocabulary intersection and constrains logits before topk.
        """
        from sglang.srt.kv_canary.runner.canary_manager import context_tuple

        dw = self._draft_worker

        # Map input_ids from target vocab → draft vocab
        batch.input_ids = self.vocab_mapping.map_target_to_draft_ids(batch.input_ids)
        draft_next_token_ids = self.vocab_mapping.map_target_to_draft_ids(
            next_token_ids
        )

        # Construct input_ids (shift by 1 and append tail token)
        if not batch.forward_mode.is_idle():
            tail_tokens = _eagle_prefill_tail_tokens(batch, draft_next_token_ids)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], tail_tokens[i].reshape(1))
                )
                pt += extend_len

        batch.spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        # STANDALONE/TLI skips hidden states
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        forward_batch = ForwardBatch.init_new(batch, dw.draft_runner)
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := dw.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            logits_output = dw.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(
            logits_output.next_token_logits, "tli_draft_extend_for_prefill"
        )
        maybe_detect_inf(
            logits_output.next_token_logits, "tli_draft_extend_for_prefill"
        )

        # Constrain logits to intersection before topk
        constrained_logits = self.vocab_mapping.constrain_draft_logits(
            logits_output.next_token_logits
        )
        probs = torch.softmax(constrained_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, dw.topk, dim=-1)
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=logits_output.hidden_states,
            bonus_tokens=next_token_ids,  # keep in target vocab
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Draft extend for decode
    # ─────────────────────────────────────────────────────────────────────────

    def _tli_draft_extend_for_decode(self, batch, batch_result):
        """Run draft-model KV-cache update after token acceptance.

        Mirrors EagleDraftWorker._draft_extend_for_decode but maps the
        next_token_ids (verified tokens) from target→draft vocab before
        feeding them into the draft model, and constrains logits before topk.
        """
        from sglang.srt.kv_canary.runner.canary_manager import context_tuple

        dw = self._draft_worker

        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_correct_drafts=batch_result.accept_lens - 1,
            num_accept_tokens=batch_result.accept_lens,
            num_tokens_per_req=dw.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=dw.speculative_num_draft_tokens,
        )
        select_index = (
            torch.arange(len(batch.seq_lens), device=dw.device)
            * dw.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )

        # Map next_token_ids from target vocab → draft vocab for draft model
        draft_next_token_ids = self.vocab_mapping.map_target_to_draft_ids(
            batch_result.next_token_ids
        )

        with dw.plan_stream_ctx:
            forward_batch = dw.prepare_for_draft_extend(
                draft_extend_input,
                batch,
                draft_next_token_ids,
                dw.speculative_num_draft_tokens,
                dw.draft_runner,
                dw.cuda_graph_runner_for_draft_extend,
            )

        if dw.plan_stream:
            torch.get_device_module(dw.device).current_stream().wait_stream(
                dw.plan_stream
            )

        can_cuda_graph = (
            dw.cuda_graph_runner_for_draft_extend
            and dw.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := dw.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            if can_cuda_graph:
                draft_logits_output = dw.cuda_graph_runner_for_draft_extend.replay(
                    forward_batch
                )
            else:
                draft_logits_output = dw.draft_runner.forward(
                    forward_batch
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            "tli_draft_extend_for_decode",
        )
        maybe_detect_inf(
            draft_logits_output.next_token_logits,
            "tli_draft_extend_for_decode",
        )

        # Select the relevant rows
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]

        # Constrain logits to intersection before topk
        constrained_logits = self.vocab_mapping.constrain_draft_logits(
            draft_logits_output.next_token_logits
        )
        probs = torch.softmax(constrained_logits, dim=-1)
        ret_topk_p, ret_topk_index = fast_topk(probs, dw.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Draft decode (multi-step draft forward)
    # ─────────────────────────────────────────────────────────────────────────

    def _tli_draft_forward(self, forward_batch: ForwardBatch):
        """Multi-step draft forward with vocabulary mapping.

        Mirrors EagleDraftWorker.draft_forward but constrains logits to the
        vocabulary intersection and maps draft tokens to target vocab for the
        verification tree.
        """
        dw = self._draft_worker
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        out_cache_loc = per_step_draft_out_cache_loc(
            out_cache_loc,
            forward_batch.batch_size,
            dw.topk,
            dw.speculative_num_steps,
        )

        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        scores = None
        for i in range(dw.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, dw.topk
            )

            # Map draft tokens to target vocab for the verification tree
            target_tree_tokens = self.vocab_mapping.map_draft_to_target_ids(
                tree_info[1]
            )
            score_list.append(tree_info[0])
            token_list.append(target_tree_tokens)
            parents_list.append(tree_info[2])

            if i == dw.speculative_num_steps - 1:
                break

            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            spec_info.hidden_states = hidden_states
            forward_batch.positions.add_(1)

            canary_index_ctx = (
                c.with_active_single_forward_manager(i)
                if (c := dw.draft_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with (
                forward_context(
                    ForwardContext(attn_backend=dw.draft_attn_backend.attn_backends[i])
                ),
                canary_index_ctx,
            ):
                logits_output = dw.draft_runner.forward(forward_batch).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            maybe_detect_inf(logits_output.next_token_logits, f"draft_forward step {i}")

            # Constrain logits to intersection before topk
            constrained_logits = self.vocab_mapping.constrain_draft_logits(
                logits_output.next_token_logits
            )
            probs = torch.softmax(constrained_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, dw.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                constrained_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB",
            )
            hidden_states = logits_output.hidden_states

        return organize_draft_results(
            score_list, token_list, parents_list, dw.speculative_num_draft_tokens
        )
