import logging
from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import resolve_dflash_mask_token
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

logger = logging.getLogger(__name__)


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp>=1/pp=1)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = target_worker.device

        self._warned_forced_greedy = False
        self._logged_first_verify = False

        # Native (SGLang) draft runner (separate KV cache + attention backend).
        # Share req_to_token_pool + token_to_kv_pool_allocator with the target worker (EAGLE3-style),
        # while keeping a separate draft KV cache pool (the draft model has different KV values).
        shared_req_to_token_pool, shared_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_backend = draft_server_args.speculative_draft_attention_backend
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = "flashinfer"
        elif draft_backend == "trtllm_mha":
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' yet; "
                "falling back to 'flashinfer'."
            )
            draft_backend = "flashinfer"
        elif draft_backend not in ("flashinfer", "fa3"):
            logger.warning(
                "DFLASH draft worker only supports attention_backend in {'flashinfer', 'fa3'} for now, "
                "but got %r. Falling back to 'flashinfer'.",
                draft_backend,
            )
            draft_backend = "flashinfer"

        # Make the draft worker backend explicit and self-contained (no further overrides).
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        # Keep draft context length aligned with the target.
        draft_server_args.context_length = target_worker.model_runner.model_config.context_len
        self.native_draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=shared_req_to_token_pool,
            token_to_kv_pool_allocator=shared_token_to_kv_pool_allocator,
        )
        self.native_draft_model_runner = self.native_draft_worker.model_runner
        self.native_draft_model = self.native_draft_model_runner.model
        if server_args.speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            self.block_size = int(getattr(self.native_draft_model, "block_size", 16))
        else:
            self.block_size = int(server_args.speculative_num_draft_tokens)
            model_block_size = getattr(self.native_draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(self.block_size):
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )

        self._mask_token = resolve_dflash_mask_token(
            draft_hf_config=self.native_draft_model_runner.model_config.hf_config
        )
        self._mask_token_id = self._resolve_mask_token_id(mask_token=self._mask_token)
        if self.tp_rank == 0:
            logger.info(
                "Initialized native DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.native_draft_model.__class__.__name__,
                self.block_size,
            )
            logger.info(
                "DFLASH draft impl selected. impl=native, mask_token=%s, mask_token_id=%s",
                self._mask_token,
                self._mask_token_id,
            )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # allocator and req_to_token_pool are shared with target worker
        pass

    def on_req_finished(self, req):
        # allocator and req_to_token_pool are shared with the target worker;
        # there is no separate draft allocation to release here.
        if hasattr(req, "dflash_draft_seq_len"):
            req.dflash_draft_seq_len = 0

    def _resolve_mask_token_id(self, *, mask_token: str) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(f"DFLASH mask_token must be a non-empty string, got {mask_token!r}.")

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("DFLASH requires tokenizer initialization (skip_tokenizer_init is not supported).")

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        mask_token_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            mask_token_id = getattr(tokenizer, "mask_token_id", None)

        if mask_token_id is None:
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            mask_token_id = vocab.get(mask_token, None)

        if mask_token_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            mask_token_id = getattr(tokenizer, "mask_token_id", None)
            if mask_token_id is None:
                mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    mask_token_id,
                    len(tokenizer),
                    vocab_size,
                )

        if mask_token_id is None or int(mask_token_id) < 0:
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        if mask_token_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={mask_token_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(mask_token_id)

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch, draft_input: DFlashDraftInput):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise ValueError("DFLASH does not support grammar-constrained decoding yet.")
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            if not self._warned_forced_greedy and self.tp_rank == 0:
                logger.warning(
                    "DFLASH currently supports greedy verification only; "
                    "ignoring non-greedy sampling params (e.g. temperature/top_p/top_k) and using argmax."
                )
                self._warned_forced_greedy = True

        bs = batch.batch_size()
        device = self.model_runner.device

        # --- 1) Append any newly committed tokens into the native draft KV cache.
        self._append_target_hidden_to_native_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # --- 2) Draft a non-causal block with the native draft model.
        block_ids = torch.full(
            (bs, self.block_size),
            self._mask_token_id,
            dtype=torch.long,
            device=device,
        )
        block_ids[:, 0] = draft_input.verified_id.to(torch.long)

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        prefix_lens_cpu = [int(x) for x in draft_input.draft_seq_lens_cpu]
        prefix_lens = torch.tensor(prefix_lens_cpu, dtype=torch.int32, device=device)
        extend_lens = torch.full(
            (bs,), int(self.block_size), dtype=torch.int32, device=device
        )
        positions, extend_start_loc = compute_position(
            self.native_draft_model_runner.server_args.attention_backend,
            prefix_lens,
            extend_lens,
            bs * self.block_size,
        )

        block_start = prefix_lens.to(torch.int64)
        block_end = block_start + int(self.block_size)
        allocator = self.native_draft_model_runner.token_to_kv_pool_allocator
        token_to_kv_pool_state_backup = allocator.backup_state()
        try:
            block_cache_loc = allocator.alloc(bs * self.block_size)
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DFLASH native draft OOM when allocating {bs * self.block_size} block tokens."
                )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.native_draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )

            # Use TARGET_VERIFY mode (cuda-graphable) to run a fixed-size draft block.
            # In this mode, `seq_lens` stores the prefix lengths; attention backends
            # derive kv_len by adding `draft_token_num`.
            draft_spec_info = DFlashVerifyInput(
                draft_token=torch.empty((0,), dtype=torch.long, device=device),
                positions=torch.empty((0,), dtype=torch.int64, device=device),
                draft_token_num=int(self.block_size),
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            seq_lens = prefix_lens.to(torch.int32)
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids.flatten(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=int(seq_lens.sum().item()),
                seq_lens_cpu=torch.tensor(prefix_lens_cpu, dtype=torch.int32),
                positions=positions,
                req_to_token_pool=self.native_draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.native_draft_model_runner.token_to_kv_pool,
                attn_backend=self.native_draft_model_runner.attn_backend,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                draft_hidden = self.native_draft_model_runner.forward(
                    forward_batch
                ).logits_output
        finally:
            # Drop the speculative block from the shared allocator (EAGLE3-style).
            allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, self.block_size - 1)
        draft_tokens = torch.cat([block_ids[:, :1], draft_next], dim=1)  # [bs, block_size]
        positions = (
            batch.seq_lens.to(torch.long).unsqueeze(1)
            + torch.arange(self.block_size, device=device, dtype=torch.long)[None, :]
        ).flatten()

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.flatten(),
            positions=positions,
            draft_token_num=self.block_size,
        )
        verify_input.prepare_for_verify(batch, self.page_size)

        batch.forward_mode = ForwardMode.TARGET_VERIFY if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH greedy sampling requires a vocab-parallel head with `weight` and `shard_indices`."
            )

        shard = lm_head.shard_indices
        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype

        # Valid ranges in the local shard (excluding padding):
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = int(hidden_states.shape[0])
        out_token_ids = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = hidden_states[start:end].to(weight_dtype)
            chunk_len = int(hs.shape[0])

            # Base vocab logits.
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = torch.max(base_logits, dim=-1)
            else:
                local_max = torch.full(
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(hs, weight[added_slice_start:added_slice_end].T)
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            # Convert local argmax indices to global token ids.
            global_ids = torch.empty(
                (chunk_len,), dtype=torch.int64, device=hs.device
            )
            is_base = local_arg < num_org
            global_ids[is_base] = org_vocab_start + local_arg[is_base]
            if num_added > 0:
                global_ids[~is_base] = added_vocab_start + (local_arg[~is_base] - num_org_padded)

            if tp_size == 1:
                out_token_ids[start:end] = global_ids.to(torch.long)
                continue

            # Gather per-rank maxima and associated global ids, then select the global max.
            gathered_max = torch.empty(
                (tp_size * chunk_len,),
                dtype=local_max.dtype,
                device=hs.device,
            )
            gathered_ids = torch.empty(
                (tp_size * chunk_len,),
                dtype=global_ids.dtype,
                device=hs.device,
            )
            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = torch.argmax(gathered_max, dim=0)
            idx = torch.arange(chunk_len, device=hs.device)
            out_token_ids[start:end] = gathered_ids[best_rank, idx].to(torch.long)

        return out_token_ids

    def _append_target_hidden_to_native_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        """Materialize the target hidden-state features into the native draft KV cache.

        This must be run before exposing new tokens to radix cache (prefix hits), otherwise
        another request could reuse target KV indices without having draft KV values.
        """

        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError("DFLASH draft state missing target_hidden context features.")
        if len(draft_input.ctx_lens_cpu) != bs:
            raise RuntimeError(
                f"DFLASH ctx_lens_cpu length mismatch: got {len(draft_input.ctx_lens_cpu)} for bs={bs}."
            )
        if len(draft_input.draft_seq_lens_cpu) != bs:
            raise RuntimeError(
                "DFLASH draft_seq_lens_cpu length mismatch: "
                f"got {len(draft_input.draft_seq_lens_cpu)} for bs={bs}."
            )

        # Invariant: draft_seq_len + ctx_len == current target prefix length.
        start_pos_cpu = batch.seq_lens_cpu.tolist()
        for cache_len, ctx_len, start_pos in zip(
            draft_input.draft_seq_lens_cpu,
            draft_input.ctx_lens_cpu,
            start_pos_cpu,
            strict=True,
        ):
            if int(cache_len) + int(ctx_len) != int(start_pos):
                raise RuntimeError(
                    "DFLASH native draft cache length mismatch. "
                    f"cache_len={int(cache_len)}, ctx_len={int(ctx_len)}, start_pos={int(start_pos)}."
                )

        total_ctx = int(sum(int(x) for x in draft_input.ctx_lens_cpu))
        if total_ctx <= 0:
            return

        req_to_token = self.native_draft_model_runner.req_to_token_pool.req_to_token
        req_pool_indices_cpu = batch.req_pool_indices.tolist()

        ctx_cache_loc_chunks: List[torch.Tensor] = []
        ctx_positions_chunks: List[torch.Tensor] = []
        new_draft_seq_lens_cpu: List[int] = []
        for req_pool_idx, cache_len, ctx_len in zip(
            req_pool_indices_cpu,
            draft_input.draft_seq_lens_cpu,
            draft_input.ctx_lens_cpu,
            strict=True,
        ):
            cache_len_i = int(cache_len)
            ctx_len_i = int(ctx_len)
            new_draft_seq_lens_cpu.append(cache_len_i + ctx_len_i)
            if ctx_len_i <= 0:
                continue
            s = cache_len_i
            e = cache_len_i + ctx_len_i
            ctx_cache_loc_chunks.append(req_to_token[req_pool_idx, s:e].to(torch.int64))
            ctx_positions_chunks.append(torch.arange(s, e, device=device, dtype=torch.int64))

        ctx_cache_loc = (
            torch.cat(ctx_cache_loc_chunks, dim=0)
            if ctx_cache_loc_chunks
            else torch.empty((0,), dtype=torch.int64, device=device)
        )

        ctx_positions = (
            torch.cat(ctx_positions_chunks, dim=0)
            if ctx_positions_chunks
            else torch.empty((0,), dtype=torch.int64, device=device)
        )

        with torch.inference_mode():
            ctx_hidden = self.native_draft_model.project_target_hidden(
                draft_input.target_hidden
            )  # [sum(ctx), hidden]

            for layer in self.native_draft_model.layers:
                attn = layer.self_attn
                qkv, _ = attn.qkv_proj(ctx_hidden)
                q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

                q, k = apply_qk_norm(
                    q=q,
                    k=k,
                    q_norm=attn.q_norm,
                    k_norm=attn.k_norm,
                    head_dim=attn.head_dim,
                )
                q, k = attn.rotary_emb(ctx_positions, q, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
                self.native_draft_model_runner.token_to_kv_pool.set_kv_buffer(
                    attn.attn,
                    ctx_cache_loc,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

        draft_input.draft_seq_lens_cpu = new_draft_seq_lens_cpu
        draft_input.ctx_lens_cpu = [0] * bs
        draft_input.target_hidden = draft_input.target_hidden[:0]

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError("DFLASH speculative decoding does not support return_logprob yet.")

        if isinstance(batch, ModelWorkerBatch):
            # Should not happen for spec-v1 (non-overlap) scheduling, but keep a sane fallback.
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if model_worker_batch.extend_seq_lens is None or model_worker_batch.extend_prefix_lens is None:
                raise RuntimeError(
                    "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

            # Materialize the prompt tokens into the draft KV cache immediately. This is required
            # for radix cache support, since the scheduler may update radix after prefill returns.
            draft_input = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens_cpu=[int(x) for x in model_worker_batch.extend_seq_lens],
                draft_seq_lens_cpu=[int(x) for x in model_worker_batch.extend_prefix_lens],
            )
            self._append_target_hidden_to_native_draft_kv(batch, draft_input)
            batch.spec_info = draft_input
            for req, draft_len in zip(batch.reqs, draft_input.draft_seq_lens_cpu, strict=True):
                req.dflash_draft_seq_len = int(draft_len)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # Decode / target-verify stage.
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )

        # Update draft state for the next iteration. Also materialize the committed verify tokens
        # into the draft KV cache immediately so radix cache entries are safe to reuse.
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens_cpu = commit_lens.cpu().tolist()
        self._append_target_hidden_to_native_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DFLASH verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
