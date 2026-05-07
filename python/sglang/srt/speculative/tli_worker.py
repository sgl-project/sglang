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

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.spec_utils import (
    fast_topk,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.speculative.standalone_worker import StandaloneWorker
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


class TLIWorker(StandaloneWorker):
    """Speculative decoding worker for heterogeneous-vocabulary draft models.

    Inherits the "separate draft model that does not share embeddings / lm_head
    with the target" boot-up logic from :class:`StandaloneWorker`, then adds
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
        # Defer CUDA graph capture so we can prune the draft LM head first.
        # init_cuda_graphs() (overridden below) checks this flag and returns
        # immediately; we call super().init_cuda_graphs() explicitly below after
        # VocabMapping is built and the pruned lm_head is in place.
        self._defer_cuda_graphs = True

        # StandaloneWorker.__init__ loads the draft model without sharing
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
        self.hot_token_id = None

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
        draft_vocab_size = self.draft_model_runner.model_config.vocab_size

        # ── Build vocabulary mapping ─────────────────────────────────────────
        self.vocab_mapping = VocabMapping(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            device=self.device,
        )

        # ── Prune the draft LM head to the intersection (gap 3 vs HF) ────────
        self._try_prune_draft_lm_head(server_args)

        # ── Now capture CUDA graphs with the pruned LM head ──────────────────
        self._defer_cuda_graphs = False
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            super().init_cuda_graphs()

    # ── Property alias ────────────────────────────────────────────────────────
    # StandaloneWorker inherits from EAGLEWorker which uses `draft_model_runner`
    # as a property pointing to `self.model_runner`.

    def init_cuda_graphs(self):
        """Defer CUDA graph capture until after LM head pruning.

        Called by ``StandaloneWorker.__init__`` (via the context-managed block).
        We skip it there and re-invoke explicitly once VocabMapping is built and
        the draft LM head has been pruned.
        """
        if getattr(self, "_defer_cuda_graphs", False):
            return  # will be called manually after LM head pruning
        super().init_cuda_graphs()

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
            lm_head = self.draft_model_runner.model.lm_head
        except AttributeError:
            logger.warning(
                "Draft model has no '.lm_head' attribute; skipping LM head pruning. "
                "Set self.draft_model_runner.model.lm_head before calling TLIWorker.__init__ "
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
        self.draft_model_runner.model.lm_head = pruned_head
        logger.info(
            "Draft LM head pruned to intersection: %d → %d tokens (%.1f%% of draft vocab). "
            "Draft decode matmul reduced by %.1f×.",
            full_vocab_size,
            K,
            100.0 * K / max(full_vocab_size, 1),
            full_vocab_size / max(K, 1),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Draft extend (prefill phase)
    # ─────────────────────────────────────────────────────────────────────────

    def forward_draft_extend(
        self,
        batch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu,
        mm_input_embeds=None,
    ):
        """Run draft-model KV-cache prefill with target→draft token ID mapping.

        ``batch.input_ids`` contains prompt token IDs in TARGET vocab space
        (they were used by the target model).  Before running the draft model
        we map them to DRAFT vocab.  The last token of each sequence (i.e.
        the newly generated target token ``next_token_ids``) is also mapped.
        """
        # Map all prompt token IDs from target → draft vocab in-place.
        # The target model is already done with batch.input_ids at this point.
        batch.input_ids = self.vocab_mapping.map_target_to_draft_ids(batch.input_ids)
        # Map the newly generated target tokens as well.
        draft_next_token_ids = self.vocab_mapping.map_target_to_draft_ids(
            next_token_ids
        )
        super().forward_draft_extend(
            batch, hidden_states, draft_next_token_ids, seq_lens_cpu, mm_input_embeds
        )

    def forward_draft_extend_after_decode(self, batch):
        """Run draft-model KV-cache update after token acceptance.

        ``batch.spec_info.verified_id`` contains all accepted tokens in TARGET
        vocab.  We temporarily remap them to DRAFT vocab so the draft model
        can process them correctly, then restore them to TARGET vocab for use
        as the tree root in subsequent decode iterations.
        """
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        # Remap accepted token IDs from target → draft vocab.
        # This affects both the `batch.input_ids` constructed by
        # `prepare_extend_after_decode` and the new `spec_info.verified_id`
        # (last accepted token per request) written by the same function.
        original_verified_id = spec_info.verified_id
        spec_info.verified_id = self.vocab_mapping.map_target_to_draft_ids(
            original_verified_id
        )

        # Run the parent's draft extend (uses draft vocab throughout).
        super().forward_draft_extend_after_decode(batch)

        # After `prepare_extend_after_decode`, `spec_info.verified_id` holds
        # the LAST accepted token per request in DRAFT vocab.  Restore it to
        # TARGET vocab so that `build_tree_kernel_efficient` (called in
        # `draft()` of the NEXT decode iteration) uses the correct token as
        # the tree root.
        batch.spec_info.verified_id = self.vocab_mapping.map_draft_to_target_ids(
            batch.spec_info.verified_id
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Draft decode (multi-step draft forward)
    # ─────────────────────────────────────────────────────────────────────────

    def capture_for_decode(self, logits_output, draft_input: EagleDraftInput):
        """Save top-k draft tokens after constraining logits to the intersection.

        Logits are constrained before softmax so the draft model only ever
        proposes tokens that exist in both vocabularies.  ``topk_index`` is
        stored in DRAFT vocab; it will be used as ``input_ids`` for the next
        draft step and mapped to TARGET vocab inside ``draft_forward`` for the
        verification tree.
        """
        constrained_logits = self.vocab_mapping.constrain_draft_logits(
            logits_output.next_token_logits
        )
        probs = torch.softmax(constrained_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        # topk_index is in DRAFT vocab space.
        draft_input.hidden_states = logits_output.hidden_states

    def draft_forward(self, forward_batch: ForwardBatch):
        """Multi-step draft forward with vocabulary mapping.

        ``spec_info.topk_index`` arriving here is in DRAFT vocab (see
        ``capture_for_decode``).  Token IDs fed to the draft model stay in
        DRAFT vocab.  Token IDs stored in ``token_list`` (which later becomes
        ``draft_tokens`` for the verification tree) are mapped to TARGET vocab.
        """
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,  # DRAFT vocab
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        # NOTE: hot_token_id is deliberately None for TLI; vocab mapping
        #       supersedes it.

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []  # populated with TARGET vocab IDs
        parents_list: List[torch.Tensor] = []

        scores = None
        for i in range(self.speculative_num_steps):
            # select_top_k_tokens uses topk_index (DRAFT) to:
            #   - produce input_ids (DRAFT vocab) — correct for the draft model
            #   - produce tree_info[1] = topk_index selection — we need TARGET
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )

            # Map draft token IDs in tree_info to target vocab for verification.
            target_tree_tokens = self.vocab_mapping.map_draft_to_target_ids(
                tree_info[1]
            )
            score_list.append(tree_info[0])
            token_list.append(target_tree_tokens)  # TARGET vocab ✓
            parents_list.append(tree_info[2])

            # Last step: no need to run another forward pass.
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs for the next draft step.
            forward_batch.input_ids = input_ids  # DRAFT vocab ✓
            # GptOss rope kernel needs cache_loc to be contiguous.
            if self.model_config.hf_config.architectures[0] == "GptOssForCausalLM":
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run the draft model forward.
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")

            # Constrain logits to intersection, then sample top-k in DRAFT space.
            constrained_logits = self.vocab_mapping.constrain_draft_logits(
                logits_output.next_token_logits
            )
            probs = torch.softmax(constrained_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            # topk_index stays in DRAFT vocab for the next iteration's input_ids.
            maybe_detect_oob(
                topk_index,
                0,
                constrained_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={constrained_logits.shape[-1]}",
            )
            hidden_states = logits_output.hidden_states

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )
        # draft_tokens is now in TARGET vocab ✓

        return parent_list, top_scores_index, draft_tokens
