import logging
from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

logger = logging.getLogger(__name__)


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp=1/pp=1)."""

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

        self._mask_token_id = self._resolve_mask_token_id()
        self._warned_forced_greedy = False
        self._logged_first_verify = False

        # Native (SGLang) draft runner (separate KV cache + attention backend).
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_server_args.disable_cuda_graph = True
        draft_backend = draft_server_args.speculative_draft_attention_backend
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = "flashinfer"
        if draft_backend not in ("flashinfer", "fa3"):
            raise ValueError(
                "DFLASH draft worker only supports attention_backend in {'flashinfer', 'fa3'} for now, "
                f"but got {draft_backend!r}. "
                "Use `--speculative-draft-attention-backend` to override the draft backend."
            )

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
        )
        self.native_draft_model_runner = self.native_draft_worker.model_runner
        self.native_draft_model = self.native_draft_model_runner.model
        self.block_size = int(getattr(self.native_draft_model, "block_size", 16))
        if self.tp_rank == 0:
            logger.info(
                "Initialized native DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.native_draft_model.__class__.__name__,
                self.block_size,
            )
            logger.info(
                "DFLASH draft impl selected. impl=native, mask_token_id=%s",
                self._mask_token_id,
            )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        if self.native_draft_model_runner is None:
            return
        self.native_draft_model_runner.req_to_token_pool.clear()
        self.native_draft_model_runner.token_to_kv_pool_allocator.clear()

    def on_req_finished(self, req):
        """Release native-draft KV cache for a finished request.

        The native draft path uses a separate KV pool that is not managed by the
        scheduler/tree-cache. We must explicitly free per-request draft KV slots
        when the request completes to avoid leaking draft KV memory across
        requests.
        """
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            return
        draft_len = getattr(req, "dflash_draft_seq_len", None)
        if draft_len is None:
            return
        draft_len = int(draft_len)
        if draft_len <= 0:
            return
        kv_indices = self.native_draft_model_runner.req_to_token_pool.req_to_token[
            req_pool_idx, :draft_len
        ]
        self.native_draft_model_runner.token_to_kv_pool_allocator.free(kv_indices)
        req.dflash_draft_seq_len = 0

    def _resolve_mask_token_id(self) -> int:
        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("DFLASH requires tokenizer initialization (skip_tokenizer_init is not supported).")

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            # `convert_tokens_to_ids` can return `None` (or an unk id) depending on tokenizer.
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            mask_token_id = vocab.get("<|MASK|>", None)

        if mask_token_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
            mask_token_id = getattr(tokenizer, "mask_token_id", None)
            if mask_token_id is None:
                mask_token_id = tokenizer.convert_tokens_to_ids("<|MASK|>")

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    "<|MASK|>",
                    mask_token_id,
                    len(tokenizer),
                    vocab_size,
                )

        if mask_token_id is None or int(mask_token_id) < 0:
            raise ValueError("DFLASH requires a `<|MASK|>` token id, but it could not be resolved.")

        if mask_token_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={mask_token_id}, vocab_size={vocab_size}. "
                "This likely means `<|MASK|>` requires vocab expansion beyond the model's embedding size. "
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

        embed_weight, head_weight = self.target_worker.model_runner.model.get_embed_and_head()

        # --- 1) Append new context tokens into the native draft KV cache.
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
                    f"cache_len={int(cache_len)}, ctx_len={int(ctx_len)}, start_pos={int(start_pos)}. "
                    "This can happen if prefix caching is enabled; start with `--disable-radix-cache` for now."
                )

        total_ctx = int(sum(int(x) for x in draft_input.ctx_lens_cpu))
        if total_ctx > 0:
            ctx_start = torch.tensor(
                draft_input.draft_seq_lens_cpu, dtype=torch.int64, device=device
            )
            ctx_len = torch.tensor(draft_input.ctx_lens_cpu, dtype=torch.int64, device=device)
            ctx_end = ctx_start + ctx_len

            ctx_cache_loc = self.native_draft_model_runner.token_to_kv_pool_allocator.alloc(
                total_ctx
            )
            if ctx_cache_loc is None:
                raise RuntimeError(
                    f"DFLASH native draft OOM when allocating {total_ctx} context tokens."
                )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.native_draft_model_runner.req_to_token_pool.req_to_token,
                ctx_start,
                ctx_end,
                ctx_cache_loc,
                bs,
            )

            ctx_positions_chunks: List[torch.Tensor] = []
            for s, e in zip(ctx_start.tolist(), ctx_end.tolist(), strict=True):
                if e > s:
                    ctx_positions_chunks.append(
                        torch.arange(s, e, device=device, dtype=torch.int64)
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
                    k = attn.k_proj(ctx_hidden)
                    v = attn.v_proj(ctx_hidden)
                    k = attn.k_norm(k.view(-1, attn.head_dim)).view_as(k)
                    dummy_q = k.new_empty((k.shape[0], attn.head_dim))
                    _, k = attn.rotary_emb(ctx_positions, dummy_q, k)
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

            draft_input.draft_seq_lens_cpu = ctx_end.to(torch.int64).cpu().tolist()
            for req, seq_len in zip(
                batch.reqs, draft_input.draft_seq_lens_cpu, strict=True
            ):
                req.dflash_draft_seq_len = int(seq_len)
            draft_input.ctx_lens_cpu = [0] * bs
            draft_input.target_hidden = draft_input.target_hidden[:0]

        # --- 2) Draft a non-causal block with the native draft model.
        block_ids = torch.full(
            (bs, self.block_size),
            self._mask_token_id,
            dtype=torch.long,
            device=device,
        )
        block_ids[:, 0] = draft_input.verified_id.to(torch.long)

        noise_embedding = F.embedding(block_ids, embed_weight)
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
        block_cache_loc = self.native_draft_model_runner.token_to_kv_pool_allocator.alloc(
            bs * self.block_size
        )
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

        seq_lens = block_end.to(torch.int64)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=block_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=torch.tensor(seq_lens.cpu().tolist(), dtype=torch.int64),
            positions=positions,
            extend_num_tokens=bs * self.block_size,
            extend_seq_lens=extend_lens,
            extend_prefix_lens=prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=prefix_lens_cpu,
            extend_seq_lens_cpu=[int(self.block_size)] * bs,
            extend_logprob_start_lens_cpu=[0] * bs,
            req_to_token_pool=self.native_draft_model_runner.req_to_token_pool,
            token_to_kv_pool=self.native_draft_model_runner.token_to_kv_pool,
            attn_backend=self.native_draft_model_runner.attn_backend,
            input_embeds=input_embeds,
        )

        with torch.inference_mode():
            draft_hidden = self.native_draft_model_runner.forward(forward_batch).logits_output

        # Crop: drop the speculative block from the draft KV cache (context stays).
        self.native_draft_model_runner.token_to_kv_pool_allocator.free(block_cache_loc)

        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        draft_logits = F.linear(draft_hidden[:, 1:, :], head_weight)
        draft_next = torch.argmax(draft_logits, dim=-1).to(torch.long)
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
            if any(len(req.prefix_indices) > 0 for req in batch.reqs):
                raise ValueError(
                    "DFLASH currently does not support radix/prefix cache hits (prefix_indices != 0). "
                    "Start with `--disable-radix-cache` for now."
                )

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

            ctx_lens_cpu = model_worker_batch.seq_lens_cpu.tolist()

            batch.spec_info = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens_cpu=ctx_lens_cpu,
                draft_seq_lens_cpu=[0] * len(ctx_lens_cpu),
            )
            for req in batch.reqs:
                req.dflash_draft_seq_len = 0

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

        # Update draft state for the next iteration.
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens_cpu = commit_lens.cpu().tolist()
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
