import logging
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus
from sglang.srt.speculative.dspark_info import (
    DSparkDraftBlockInput,
    DSparkDraftInputV2,
    DSparkVerifyInput,
)
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.speculative.triton_ops.dspark import (
    _compute_dspark_accept_bonus_triton_unchecked,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.common import empty_context

logger = logging.getLogger(__name__)


class DSparkWorkerV2(BaseSpecWorker):
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
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device

        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()
        with (
            empty_context(),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._draft_worker = TpModelWorker(
                server_args=draft_server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                pp_rank=0,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                dp_rank=dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
            )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self._draft_worker.model_runner
        self._draft_worker.draft_runner = self.draft_model_runner
        self.draft_model = self.draft_model_runner.model
        self._draft_inner = self.draft_model.model

        self.draft_model.model.embed_tokens.weight = (
            self.target_worker.model_runner.model.model.embed_tokens.weight
        )
        self.draft_model.lm_head.weight = (
            self.target_worker.model_runner.model.lm_head.weight
        )

        self.block_size = int(server_args.speculative_num_draft_tokens)
        model_block_size = int(getattr(self.draft_model, "block_size", self.block_size))
        if model_block_size != self.block_size:
            logger.warning(
                "DSpark block size mismatch: using speculative_num_draft_tokens=%s "
                "but draft model block_size=%s.",
                self.block_size,
                model_block_size,
            )
        self.speculative_num_draft_tokens = int(self.block_size)

        self.noise_token_id = int(self._draft_inner.noise_token_id)
        self.markov_rank = int(self._draft_inner.markov_rank)
        self.num_dspark_layers = int(self.draft_model.num_dspark_layers)
        self.confidence_threshold = float(
            server_args.speculative_dspark_confidence_threshold
        )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self._use_triton_accept_bonus = is_cuda() or is_hip()
        self._accept_bonus_buffer_cap: int = 0
        self._accept_bonus_buffer_slot: int = 0
        self._commit_lens_bufs: List[torch.Tensor] = []
        self._bonus_id_bufs: List[torch.Tensor] = []
        self._out_tokens_bufs: List[torch.Tensor] = []
        self._new_seq_lens_bufs: List[torch.Tensor] = []
        self._markov_refine_buffer_cap: int = 0
        self._markov_candidates_buf: Optional[torch.Tensor] = None
        self._markov_embeds_buf: Optional[torch.Tensor] = None

        if self.tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. model=%s, block_size=%s, "
                "num_dspark_layers=%s, noise_token_id=%s, markov_rank=%s, "
                "confidence_threshold=%s",
                self.draft_model.__class__.__name__,
                self.block_size,
                self.num_dspark_layers,
                self.noise_token_id,
                self.markov_rank,
                self.confidence_threshold,
            )

    def _get_dp_decode_global_num_tokens(
        self, batch: ScheduleBatch
    ) -> Optional[list[int]]:
        if not self.server_args.enable_dp_attention or batch.global_num_tokens is None:
            return None

        global_num_tokens = [int(x) for x in batch.global_num_tokens]
        if any(x > 0 for x in global_num_tokens):
            return [max(1, x) for x in global_num_tokens]
        return global_num_tokens

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._draft_worker.init_cuda_graphs()

    def clear_cache_pool(self):
        pass

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _materialize_main_hidden_to_draft_kv(
        self,
        *,
        main_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if main_hidden is None:
            raise RuntimeError("DSpark missing target main_hidden context features.")
        if main_hidden.numel() == 0:
            return

        device = self.device
        if main_hidden.device != device:
            main_hidden = main_hidden.to(device, non_blocking=True)
        if cache_loc.device != device:
            cache_loc = cache_loc.to(device, non_blocking=True)
        if positions.device != device:
            positions = positions.to(device, non_blocking=True)
        if cache_loc.dtype != torch.int64:
            cache_loc = cache_loc.to(torch.int64)
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)

        attn_backend = self.draft_model_runner.attn_backend
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            torch.inference_mode(),
        ):
            main_x = self.draft_model.project_main_hidden(main_hidden)
            for layer in self._draft_inner.layers:
                layer.self_attn.kv_from_hidden(
                    main_x, positions, cache_loc, attn_backend
                )

    def _run_draft_block(
        self,
        *,
        batch: ScheduleBatch,
        bs: int,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        verify_out_cache_loc: torch.Tensor,
        dp_decode_global_num_tokens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        draft_block_spec_info = DSparkDraftBlockInput(
            draft_token=block_ids.reshape(-1),
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        draft_forward_batch = draft_block_spec_info.prepare_for_draft_block(
            batch=batch,
            draft_model_runner=self.draft_model_runner,
            out_cache_loc=verify_out_cache_loc,
            dp_decode_global_num_tokens=dp_decode_global_num_tokens,
        )

        from sglang.srt.layers.attention import deepseek_v4_backend as _dsv4_be

        _dsv4_be._DSPARK_BLOCK_FULL_ATTN = int(self.block_size)
        try:
            with torch.inference_mode():
                draft_runner_out = self.draft_model_runner.forward(draft_forward_batch)
        finally:
            _dsv4_be._DSPARK_BLOCK_FULL_ATTN = 0

        raw = draft_runner_out.logits_output
        block_hidden = raw if isinstance(raw, torch.Tensor) else raw.hidden_states
        if block_hidden is None:
            raise RuntimeError("DSpark draft model returned no block hidden states.")
        reshape_bs = bs
        keep_tokens = bs * int(self.block_size)
        if bs == 0 and dp_decode_global_num_tokens is not None:
            reshape_bs = 1
            keep_tokens = int(self.block_size)
            if block_hidden.numel() == 0:
                block_hidden = block_hidden.new_zeros(
                    int(self.block_size), self._draft_inner.hidden_size
                )
        block_hidden = block_hidden[:keep_tokens]
        return block_hidden.reshape(
            reshape_bs, int(self.block_size), block_hidden.shape[-1]
        )

    def _ensure_markov_refine_buffers(self, bs: int, device: torch.device) -> None:
        cap = self._markov_refine_buffer_cap
        if (
            cap >= int(bs)
            and self._markov_candidates_buf is not None
            and self._markov_embeds_buf is not None
            and self._markov_candidates_buf.device == device
            and self._markov_embeds_buf.device == device
        ):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        markov_weight = getattr(self._draft_inner.markov_head.markov_w1, "weight", None)
        markov_dtype = (
            markov_weight.dtype
            if markov_weight is not None
            else self.draft_model.lm_head.weight.dtype
        )
        self._markov_candidates_buf = torch.empty(
            (new_cap, int(self.block_size)), dtype=torch.int64, device=device
        )
        self._markov_embeds_buf = torch.empty(
            (new_cap, int(self.block_size), int(self.markov_rank)),
            dtype=markov_dtype,
            device=device,
        )
        self._markov_refine_buffer_cap = new_cap

    def _refine_block_markov(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        output_bs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = int(block_hidden.shape[0])
        output_bs = bs if output_bs is None else int(output_bs)
        block_size = int(self.block_size)
        if bs == 0:
            empty_tokens = torch.empty(
                (output_bs, block_size), dtype=torch.int64, device=block_hidden.device
            )
            empty_confidence = torch.empty(
                (output_bs, block_size), dtype=torch.float32, device=block_hidden.device
            )
            return empty_tokens, empty_confidence

        self._ensure_markov_refine_buffers(bs, block_hidden.device)
        assert self._markov_candidates_buf is not None
        assert self._markov_embeds_buf is not None
        candidates = self._markov_candidates_buf[:bs]
        markov_embeds = self._markov_embeds_buf[:bs]

        markov_head = self._draft_inner.markov_head
        confidence_head = self._draft_inner.confidence_head
        lm_head = self.draft_model.lm_head

        tp_size = get_tensor_model_parallel_world_size()
        vocab_size = int(self._draft_inner.vocab_size)

        def _gather_full_vocab(logits_shard: torch.Tensor) -> torch.Tensor:
            if logits_shard.shape[-1] >= vocab_size:
                return logits_shard[..., :vocab_size]
            if tp_size == 1:
                return logits_shard
            return tensor_model_parallel_all_gather(logits_shard, dim=-1)[
                ..., :vocab_size
            ]

        if bonus_tokens.numel() == bs:
            first_tokens = bonus_tokens.view(-1).to(torch.int64)
        else:
            first_tokens = torch.full(
                (bs,), self.noise_token_id, dtype=torch.int64, device=block_hidden.device
            )
        candidates[:, 0].copy_(first_tokens)

        with torch.inference_mode():
            base_logits = _gather_full_vocab(F.linear(block_hidden, lm_head.weight))
            prev_tokens = candidates[:, 0]
            for i in range(block_size):
                prev_embed = markov_head.get_prev_embeddings(prev_tokens)
                markov_embeds[:, i].copy_(prev_embed)
                bias = _gather_full_vocab(markov_head.project_bias(prev_embed))
                bias.add_(base_logits[:, i])
                next_tokens = torch.argmax(bias, dim=-1)
                if i + 1 < block_size:
                    candidates[:, i + 1].copy_(next_tokens)
                prev_tokens = next_tokens

            confidence = confidence_head(block_hidden, markov_embeds)

        return candidates[:output_bs], confidence[:output_bs]

    def _confident_prefix(self, confidence: torch.Tensor) -> torch.Tensor:
        keep = torch.sigmoid(confidence) >= self.confidence_threshold
        return keep.to(torch.int32).cumprod(dim=1).sum(dim=1)

    def _ensure_accept_bonus_buffers(self, bs: int) -> None:
        if self._accept_bonus_buffer_cap >= int(bs):
            return

        new_cap = max(
            int(bs),
            (
                self._accept_bonus_buffer_cap * 2
                if self._accept_bonus_buffer_cap > 0
                else int(bs)
            ),
        )
        device = self.device
        block_size = int(self.block_size)
        self._commit_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._bonus_id_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._out_tokens_bufs = [
            torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
            for _ in range(2)
        ]
        self._new_seq_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._accept_bonus_buffer_cap = new_cap

    def _next_accept_bonus_buffers(self, bs: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._ensure_accept_bonus_buffers(bs)
        slot = self._accept_bonus_buffer_slot
        self._accept_bonus_buffer_slot = (slot + 1) % 2
        return (
            self._commit_lens_bufs[slot][:bs],
            self._bonus_id_bufs[slot][:bs],
            self._out_tokens_bufs[slot][:bs],
            self._new_seq_lens_bufs[slot][:bs],
        )

    def _compute_accept_bonus_eager(
        self,
        *,
        candidates: torch.Tensor,
        target_predict: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, block_size = candidates.shape
        correct_len, _ = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        confident_prefix = self._confident_prefix(confidence)
        correct_len = torch.minimum(
            correct_len.to(torch.int64), confident_prefix.to(torch.int64)
        )
        bonus_tokens = target_predict.gather(1, correct_len.unsqueeze(1)).squeeze(1)
        commit_lens = correct_len.to(torch.int32) + 1

        out_tokens = torch.empty(
            (bs, block_size), dtype=torch.int64, device=candidates.device
        )
        if block_size > 1:
            out_tokens[:, : block_size - 1].copy_(candidates[:, 1:])
        out_tokens[:, block_size - 1].fill_(0)
        out_tokens.scatter_(
            1,
            correct_len.unsqueeze(1),
            bonus_tokens.unsqueeze(1).to(torch.int64),
        )
        return commit_lens, bonus_tokens, out_tokens

    def _make_next_draft_input_prefill(
        self,
        *,
        bonus_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DSparkDraftInputV2:
        return DSparkDraftInputV2(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=seq_lens.to(dtype=torch.int64),
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def _make_next_draft_input_decode(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DSparkDraftInputV2:
        return DSparkDraftInputV2(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(model_worker_batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        sampling_info = getattr(model_worker_batch, "sampling_info", None)
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and self.tp_rank == 0
            and not getattr(self, "_warned_sampling", False)
        ):
            self._warned_sampling = True
            logger.warning(
                "DSpark verifies greedily; temperature>0 requests are served with "
                "greedy verification. Rejection-sampling support is a follow-up."
            )

        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            return self._forward_prefill(model_worker_batch, on_publish)

        return self._forward_decode(model_worker_batch, on_publish)

    def _forward_prefill(
        self, model_worker_batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(model_worker_batch)

        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = model_worker_batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DSpark target layers configured."
            )
        if model_worker_batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        device = next_token_ids.device
        extend_lens = model_worker_batch.extend_lens
        prefix_lens = model_worker_batch.prefix_lens
        if extend_lens is None or prefix_lens is None:
            reqs = getattr(model_worker_batch, "reqs", None) or []
            if len(reqs) != len(model_worker_batch.seq_lens):
                raise RuntimeError(
                    "DSpark expected extend_lens / prefix_lens in extend mode, "
                    "and could not rebuild them from batch requests."
                )
            prefix_lens = [len(req.prefix_indices) for req in reqs]
            extend_lens = [req.extend_range.length for req in reqs]

        ctx_lens = torch.tensor(extend_lens, dtype=torch.int32, device=device)
        draft_seq_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(extend_lens)),
        )
        self._materialize_main_hidden_to_draft_kv(
            main_hidden=logits_output.hidden_states,
            cache_loc=model_worker_batch.out_cache_loc,
            positions=positions,
        )

        logits_output.hidden_states = None

        batch_output.next_draft_input = self._make_next_draft_input_prefill(
            bonus_tokens=next_token_ids,
            seq_lens=model_worker_batch.seq_lens,
            cur_allocated_seq_lens_cpu=model_worker_batch.seq_lens_cpu,
        )
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        batch_output.next_draft_input.verify_done = verify_done
        return batch_output

    def _forward_decode(
        self, model_worker_batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = DSparkDraftInputV2.create_idle_input(
                device=self.device
            )
        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DSparkDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DSparkDraftInputV2 state on the running batch."
            )

        participates_in_dp_decode = (
            self.server_args.enable_dp_attention
            and model_worker_batch.forward_mode.is_idle()
            and model_worker_batch.global_num_tokens is not None
            and any(int(x) > 0 for x in model_worker_batch.global_num_tokens)
        )
        if model_worker_batch.forward_mode.is_idle() and not participates_in_dp_decode:
            return self._forward_idle(on_publish)
        dp_decode_global_num_tokens = self._get_dp_decode_global_num_tokens(
            model_worker_batch
        )

        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        device = self.device
        bs = len(model_worker_batch.seq_lens)
        block_size = int(self.block_size)
        prefix_lens = model_worker_batch.seq_lens
        req_pool_indices = model_worker_batch.req_pool_indices

        block_ids = torch.full(
            (bs, block_size), self.noise_token_id, dtype=torch.int64, device=device
        )
        block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))

        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        positions = positions_2d.reshape(-1).to(torch.int64)

        end_offset = prefix_lens + block_size
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=end_offset,
            batch_size=bs,
            draft_token_num=block_size,
            device=device,
        )

        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            block_hidden = self._run_draft_block(
                batch=model_worker_batch,
                bs=bs,
                block_ids=block_ids,
                positions=positions,
                verify_out_cache_loc=verify_out_cache_loc,
                dp_decode_global_num_tokens=dp_decode_global_num_tokens,
            )

            candidates, confidence = self._refine_block_markov(
                block_hidden=block_hidden,
                bonus_tokens=draft_input.bonus_tokens,
                output_bs=bs,
            )

        verify_input = DSparkVerifyInput(
            draft_token=candidates.reshape(-1),
            positions=positions,
            draft_token_num=block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        model_worker_batch.out_cache_loc = verify_out_cache_loc
        if participates_in_dp_decode:
            model_worker_batch.forward_mode = ForwardMode.DECODE
        original_global_num_tokens = model_worker_batch.global_num_tokens
        original_global_num_tokens_for_logprob = (
            model_worker_batch.global_num_tokens_for_logprob
        )
        if dp_decode_global_num_tokens is not None:
            model_worker_batch.global_num_tokens = dp_decode_global_num_tokens
            if original_global_num_tokens_for_logprob is not None:
                model_worker_batch.global_num_tokens_for_logprob = (
                    dp_decode_global_num_tokens
                )
        try:
            verify_forward_batch, _ = verify_input.prepare_for_verify(
                model_worker_batch, self.target_worker
            )
        finally:
            model_worker_batch.global_num_tokens = original_global_num_tokens
            model_worker_batch.global_num_tokens_for_logprob = (
                original_global_num_tokens_for_logprob
            )
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, block_size
        )

        new_seq_lens = None
        if bs == 0:
            bonus_tokens = torch.empty((0,), dtype=torch.int64, device=device)
            commit_lens = torch.empty((0,), dtype=torch.int32, device=device)
            out_tokens = torch.empty((0, block_size), dtype=torch.int64, device=device)
        elif self._use_triton_accept_bonus:
            try:
                (
                    commit_lens,
                    bonus_tokens,
                    out_tokens,
                    new_seq_lens,
                ) = self._next_accept_bonus_buffers(bs)
                _compute_dspark_accept_bonus_triton_unchecked(
                    candidates=candidates,
                    target_top1=target_predict,
                    confidence=confidence,
                    commit_lens_out=commit_lens,
                    bonus_ids_out=bonus_tokens,
                    out_tokens_out=out_tokens,
                    prefix_lens=prefix_lens,
                    new_seq_lens_out=new_seq_lens,
                    confidence_threshold=self.confidence_threshold,
                )
            except Exception as e:
                self._use_triton_accept_bonus = False
                logger.warning(
                    "DSPARK Triton accept/bonus failed; falling back to eager path: %s",
                    e,
                )
                commit_lens, bonus_tokens, out_tokens = self._compute_accept_bonus_eager(
                    candidates=candidates,
                    target_predict=target_predict,
                    confidence=confidence,
                )
        else:
            commit_lens, bonus_tokens, out_tokens = self._compute_accept_bonus_eager(
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )

        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DSpark verify requires target main_hidden states, but got None."
            )
        if bs > 0:
            hidden = hidden.view(bs, block_size, -1)
            commit_mask = (
                self._block_pos_offsets.unsqueeze(0)
                < commit_lens.unsqueeze(1).to(torch.int64)
            ).reshape(-1)
            self._materialize_main_hidden_to_draft_kv(
                main_hidden=hidden.reshape(-1, hidden.shape[-1])[commit_mask],
                cache_loc=verify_out_cache_loc[commit_mask],
                positions=positions[commit_mask],
            )

        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
        )
        next_draft_input.carry_prepare_buffers_from(draft_input)
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=block_size,
            new_seq_lens=new_seq_lens,
        )

    def _forward_idle(self, on_publish) -> GenerationBatchResult:
        empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
        empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=empty_ids,
            accept_lens=empty_lens,
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.block_size),
            new_seq_lens=next_draft_input.new_seq_lens,
        )
