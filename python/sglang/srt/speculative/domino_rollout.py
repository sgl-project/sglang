import logging
from typing import Optional

import torch

from sglang.srt.speculative.domino_helper import DFlashDominoHelper
from sglang.srt.speculative.domino_kernels import (
    fused_gru_cell_from_table,
    fused_silu_fc2_argmax,
)

logger = logging.getLogger(__name__)

_DOMINO_BLOCK_V = 512
_DOMINO_BLOCK_M = 32
_DOMINO_SCORE_NUM_WARPS = 4
_DOMINO_SCORE_NUM_STAGES = 3


class DFlashDominoRollout:
    """Domino-specific draft-token rollout for DFLASH speculative decoding.

    This first port supports CUDA + TP=1 only. The draft-token selection at each
    step feeds the next GRU state, so TP>1 needs per-step cross-rank
    synchronization; that combination is rejected early during worker init
    (`DFlashWorkerV2.__init__`). The rollout keeps a defensive CUDA-only guard
    because it relies on `torch.cuda.CUDAGraph()` and Triton kernels.

    Scoring is full-vocab: each step computes the true argmax of
    `base_logits + domino_bias` over the whole vocabulary, so the result matches
    a dense reference implementation exactly (no candidate-pool approximation).
    """

    def __init__(
        self,
        *,
        domino_helper: DFlashDominoHelper,
        block_size: int,
    ) -> None:
        self.domino_helper = domino_helper
        self.block_size = int(block_size)

    def _slice_domino_fc2(
        self,
        *,
        state: dict,
        start: int,
        length: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        end = int(start) + int(length)
        fc2_w = state["fc2_weight"][int(start) : end].contiguous()
        fc2_b = (
            state["fc2_bias"][int(start) : end].contiguous()
            if state["fc2_bias"] is not None
            else None
        )
        return fc2_w, fc2_b

    def _init_domino_prefix_gru_hidden(
        self,
        *,
        prefix_ids: torch.Tensor,
        z_dtype: torch.dtype,
        embed_module,
    ) -> torch.Tensor:
        """Initialize the Domino prefix GRU state from [verified, first draft]."""

        if embed_module is None:
            raise RuntimeError(
                "DFLASH Domino prefix init requires the target embedding module."
            )
        prefix_embeds = embed_module(prefix_ids).to(z_dtype)
        return self.domino_helper.init_gru_hidden(prefix_embeds)

    def _get_or_capture_domino_loop_graph(
        self,
        *,
        bs: int,
        num_draft: int,
        emb_dim: int,
        gru_hidden: int,
        num_org: int,
        org_vocab_start: int,
        z_dtype: torch.dtype,
        logits_dtype: torch.dtype,
        device: torch.device,
        state: dict,
        gru_input_table: torch.Tensor,
    ):
        """Capture or return the CUDA graph for the full-vocab Domino rollout loop."""

        pool = getattr(self, "_domino_loop_graph_pool", None)
        if pool is None:
            pool = {}
            self._domino_loop_graph_pool = pool

        pool_key = (
            bs,
            num_draft,
            emb_dim,
            gru_hidden,
            num_org,
            org_vocab_start,
            z_dtype,
            logits_dtype,
        )
        entry = pool.get(pool_key)
        if entry is not None:
            return entry

        G = gru_hidden
        z_proj_buf = torch.empty((bs, num_draft, emb_dim), dtype=z_dtype, device=device)
        base_logits_buf = torch.empty(
            (bs, num_draft, num_org), dtype=logits_dtype, device=device
        )
        gru_h_buf = torch.empty((bs, G), dtype=z_dtype, device=device)
        out_buf = torch.empty((bs, num_draft), dtype=torch.long, device=device)

        block_v = _DOMINO_BLOCK_V
        block_m = _DOMINO_BLOCK_M
        score_num_warps = _DOMINO_SCORE_NUM_WARPS
        score_num_stages = _DOMINO_SCORE_NUM_STAGES
        num_score_blocks = (num_org + block_v - 1) // block_v

        w_sh_T = torch.cat(
            [state["w_s"].T.contiguous(), state["w_hh"].T.contiguous()],
            dim=1,
        ).contiguous()
        sh_buf = torch.empty((bs, emb_dim + 3 * G), dtype=z_dtype, device=device)
        s_proj_buf = sh_buf[:, :emb_dim]
        gh_buf = sh_buf[:, emb_dim:]
        h_new_buf = torch.empty((bs, G), dtype=z_dtype, device=device)
        argmax_val_buf = torch.empty(
            (bs, num_score_blocks), dtype=torch.float32, device=device
        )
        argmax_idx_buf = torch.empty(
            (bs, num_score_blocks), dtype=torch.int32, device=device
        )
        local_tok_buf = torch.empty((bs,), dtype=torch.long, device=device)
        fc2_w_shard, fc2_b_shard = self._slice_domino_fc2(
            state=state,
            start=org_vocab_start,
            length=num_org,
        )
        b_hh_static = state["b_hh"]

        static_refs = [
            z_proj_buf,
            base_logits_buf,
            gru_h_buf,
            out_buf,
            sh_buf,
            h_new_buf,
            argmax_val_buf,
            argmax_idx_buf,
            local_tok_buf,
            fc2_w_shard,
            w_sh_T,
            gru_input_table,
        ]
        if fc2_b_shard is not None:
            static_refs.append(fc2_b_shard)
        if b_hh_static is not None:
            static_refs.append(b_hh_static)

        def run_loop(out_target):
            h_state = gru_h_buf
            for k in range(1, num_draft):
                torch.matmul(h_state, w_sh_T, out=sh_buf)
                fused_silu_fc2_argmax(
                    z_proj=z_proj_buf[:, k, :],
                    s_proj=s_proj_buf,
                    fc2_weight=fc2_w_shard,
                    fc2_bias=fc2_b_shard,
                    base_logits=base_logits_buf[:, k, :],
                    out_val=argmax_val_buf,
                    out_idx=argmax_idx_buf,
                    final_token=local_tok_buf,
                    block_v=block_v,
                    block_m=block_m,
                    num_warps=score_num_warps,
                    num_stages=score_num_stages,
                )
                tok_full = local_tok_buf + org_vocab_start
                out_target[:, k] = tok_full
                if k + 1 < num_draft:
                    fused_gru_cell_from_table(
                        tok_full=tok_full,
                        gru_input_table=gru_input_table,
                        gh=gh_buf,
                        gh_bias=b_hh_static,
                        h_state=h_state,
                        h_out=h_new_buf,
                    )
                    h_state = h_new_buf
            return None

        warmup_out = torch.empty((bs, num_draft), dtype=torch.long, device=device)
        for _ in range(2):
            run_loop(warmup_out)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_loop(out_buf)

        entry = {
            "graph": graph,
            "z_proj_buf": z_proj_buf,
            "base_logits_buf": base_logits_buf,
            "gru_h_buf": gru_h_buf,
            "out_buf": out_buf,
            "_static_refs": static_refs,
        }
        pool[pool_key] = entry
        logger.info(
            "[DFLASH Domino] captured CUDA graph (Triton full-vocab) for rollout "
            "loop bs=%d num_draft=%d",
            bs,
            num_draft,
        )
        return entry

    def rollout_draft_block(
        self,
        *,
        draft_hidden: torch.Tensor,
        verified_id: torch.Tensor,
        target_model,
        lm_head,
    ) -> torch.Tensor:
        """Sequential Domino rollout to produce `block_size - 1` draft tokens.

        Algorithm mirrors the reference Domino rollout for both shift_label=True
        and shift_label=False. SGLang's draft block reserves slot 0 for the
        current verified token and emits slots 1..block_size-1:
          1. Select z so z[:, 0] predicts slot_1 under the checkpoint's label
             alignment.
          2. slot_1 = argmax(base_logits[0]).
          3. Initialize prefix_gru on embeddings of [verified_id, slot_1] to get gru_h.
          4. For k = 1..block_size-2:
                 bias = embed_proj(cat(z_k, gru_h))
                 slot_{k+1} = argmax(base_logits[k] + bias)   # full vocab
                 gru_h = prefix_gru_step(embed(slot_{k+1}))

        Implementation note: the per-step embed_proj/GRU calls have non-trivial
        Python + cuDNN launch overhead.  We pre-split fc1.weight along its input
        axis so z @ W_z (the part that doesn't depend on the GRU state) is
        batched outside the loop, and we replace nn.GRU(seq_len=1) with a manual
        GRU cell to skip cuDNN setup.

        Args:
            draft_hidden: [B, block_size, hidden_size] draft model output (post-norm).
            verified_id: [B] current verified token per request (int64).
            target_model: the SGLang target model (for embed_tokens).
            lm_head: vocab-parallel lm_head (used to compute dense base logits).

        Returns:
            [B, block_size - 1] int64 tensor of sampled draft tokens.
        """
        bs, total_slots, hidden_size = draft_hidden.shape
        if total_slots != self.block_size:
            raise RuntimeError(
                f"DFLASH Domino expected draft_hidden block dim={self.block_size}, "
                f"got {total_slots}."
            )
        num_draft = self.block_size - 1  # 15 for block_size=16
        if num_draft <= 0:
            raise RuntimeError(
                f"DFLASH Domino requires block_size > 1, got {self.block_size}."
            )

        device = draft_hidden.device
        # Defensive: TP>1 is rejected at worker init, but the CUDA-graph + Triton
        # path also requires a CUDA device. Fail clearly instead of silently
        # falling back on CPU/other backends.
        if device.type != "cuda":
            raise NotImplementedError(
                "DFLASH Domino rollout requires a CUDA device (uses CUDA graphs and "
                f"Triton kernels), got device.type={device.type!r}."
            )

        weight = lm_head.weight
        shard = lm_head.shard_indices
        num_added = int(shard.num_added_elements)
        if num_added != 0:
            raise NotImplementedError(
                "DFLASH Domino rollout does not yet handle added-vocab lm_head shards."
            )
        org_vocab_start = int(shard.org_vocab_start_index)
        num_org = int(shard.num_org_elements)
        if num_org <= 0:
            raise RuntimeError("DFLASH lm_head has empty base vocab shard.")

        draft_model = self.domino_helper.draft_model
        embed_module = target_model.get_input_embeddings()
        domino_helper = self.domino_helper
        if domino_helper is None:
            raise RuntimeError("DFLASH Domino rollout called without a Domino helper.")

        # shift_label=True: draft_hidden[:, i, :] predicts token at position i+1.
        # shift_label=False: draft_hidden[:, i, :] predicts token at position i.
        # For the draft block, position 0 is verified_id; positions 1..block_size-1
        # are the draft slots.  With shift_label we need draft_hidden[:, 0, :]
        # to predict slot_1, so we slice [:num_draft]; otherwise we slice [1:].
        if getattr(draft_model, "shift_label", False):
            z = draft_hidden[:, :num_draft, :].contiguous()  # [B, num_draft, hidden]
        else:
            z = draft_hidden[:, 1:, :].contiguous()  # [B, num_draft, hidden]

        state = domino_helper.get_rollout_state()

        G = state["gru_hidden_size"]
        emb_dim = int(state["w_z"].shape[0])
        z_for_dtype = z.to(weight.dtype) if z.dtype != weight.dtype else z
        gru_input_table = domino_helper.get_gru_input_proj_table(embed_module.weight)

        # Dense base logits over the full (org) vocab shard. Full-vocab scoring
        # guarantees the per-step argmax of (base_logits + domino_bias) is exact.
        z_flat = z_for_dtype.reshape(bs * num_draft, hidden_size)
        base_logits = torch.matmul(z_flat, weight[:num_org].T).view(
            bs, num_draft, num_org
        )

        slot_local_arg = torch.argmax(base_logits[:, 0, :], dim=-1)
        slot_1 = (slot_local_arg + org_vocab_start).to(torch.long)

        gru_h = self._init_domino_prefix_gru_hidden(
            prefix_ids=torch.stack([verified_id.to(torch.long), slot_1], dim=1),
            z_dtype=z.dtype,
            embed_module=embed_module,
        )

        z_proj_all = torch.nn.functional.linear(z, state["w_z"], state["b1"])

        graph_entry = self._get_or_capture_domino_loop_graph(
            bs=bs,
            num_draft=num_draft,
            emb_dim=emb_dim,
            gru_hidden=G,
            num_org=num_org,
            org_vocab_start=org_vocab_start,
            z_dtype=z.dtype,
            logits_dtype=base_logits.dtype,
            device=device,
            state=state,
            gru_input_table=gru_input_table,
        )
        graph_entry["z_proj_buf"].copy_(z_proj_all)
        graph_entry["base_logits_buf"].copy_(base_logits)
        graph_entry["gru_h_buf"].copy_(gru_h)
        graph_entry["out_buf"][:, 0].copy_(slot_1)
        graph_entry["graph"].replay()
        out = graph_entry["out_buf"]
        return out
