import logging
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_correct_drafts_and_bonus,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.dspark_info import DSparkDraftInputV2, DSparkVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.utils import get_available_gpu_memory, is_cuda

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
        self._sampling_verify_logged = False

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
        self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        capture_decode_cuda_graph = not self.server_args.disable_cuda_graph
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        self._draft_worker.init_cuda_graphs(
            capture_decode_cuda_graph=capture_decode_cuda_graph
        )

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
        if not main_hidden.shape[0] == cache_loc.shape[0] == positions.shape[0]:
            raise RuntimeError(
                "DSpark draft KV materialization row mismatch: "
                f"main_hidden={main_hidden.shape[0]}, cache_loc={cache_loc.shape[0]}, "
                f"positions={positions.shape[0]}. Under prefill context parallelism "
                "the captured target hidden states must be gathered to full token order."
            )

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
        with torch.inference_mode():
            main_x = self.draft_model.project_main_hidden(main_hidden)
            for layer in self._draft_inner.layers:
                layer.self_attn.kv_from_hidden(
                    main_x, positions, cache_loc, attn_backend
                )

    def _run_draft_block(
        self,
        *,
        bs: int,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        verify_out_cache_loc: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        seq_lens_sum: Optional[int] = None,
    ) -> torch.Tensor:
        device = self.device
        # Host seq lengths must be committed (mirror the committed device
        # seq_lens=prefix_lens), not the reserved over-allocation:
        # DeepseekV4AttnBackend.init_forward_metadata adds +block itself for
        # TARGET_VERIFY, so committed+block is the true verify extent. Passing the
        # reserved (committed + 2*block) host bound would overshoot the allocated
        # req_to_token range by one block.
        if seq_lens_cpu is None:
            seq_lens_cpu = prefix_lens.to(device="cpu", dtype=torch.int32)
        if seq_lens_sum is None:
            seq_lens_sum = int(seq_lens_cpu.sum().item())
        draft_block_spec_info = DSparkVerifyInput(
            draft_token=block_ids.reshape(-1),
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            block_full_attn=int(self.block_size),
        )
        draft_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.reshape(-1),
            req_pool_indices=req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        with torch.inference_mode():
            draft_runner_out = self.draft_model_runner.forward(draft_forward_batch)

        raw = draft_runner_out.logits_output
        block_hidden = raw if isinstance(raw, torch.Tensor) else raw.hidden_states
        if block_hidden is None:
            raise RuntimeError("DSpark draft model returned no block hidden states.")
        return block_hidden.view(bs, int(self.block_size), -1)

    def _refine_block_markov(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = int(block_hidden.shape[0])
        block_size = int(self.block_size)
        markov_head = self._draft_inner.markov_head
        confidence_head = self._draft_inner.confidence_head
        lm_head = self.draft_model.lm_head

        tp_size = get_tensor_model_parallel_world_size()
        vocab_size = int(self._draft_inner.vocab_size)

        def _gather_full_vocab(logits_shard: torch.Tensor) -> torch.Tensor:
            if tp_size == 1:
                return logits_shard
            return tensor_model_parallel_all_gather(logits_shard, dim=-1)[
                ..., :vocab_size
            ]

        out_tokens = torch.empty(
            (bs, block_size + 1), dtype=torch.int64, device=block_hidden.device
        )
        out_tokens[:, 0] = bonus_tokens.view(-1).to(torch.int64)
        markov_embeds = []
        with torch.inference_mode():
            normed_hidden = self._draft_inner.shared_head.norm(block_hidden)
            base_logits = _gather_full_vocab(F.linear(normed_hidden, lm_head.weight))
            for i in range(block_size):
                prev_embed = markov_head.get_prev_embeddings(out_tokens[:, i])
                bias = _gather_full_vocab(markov_head.project_bias(prev_embed))
                refined = base_logits[:, i] + bias
                out_tokens[:, i + 1] = torch.argmax(refined, dim=-1)
                markov_embeds.append(prev_embed)

            stacked_embed = torch.stack(markov_embeds, dim=1)
            confidence = confidence_head(block_hidden, stacked_embed)

        candidates = out_tokens[:, :block_size].contiguous()
        return candidates, confidence

    def _confident_prefix(self, confidence: torch.Tensor) -> torch.Tensor:
        keep = torch.sigmoid(confidence) >= self.confidence_threshold
        return keep.to(torch.int32).cumprod(dim=1).sum(dim=1)

    def _make_next_draft_input_prefill(
        self,
        *,
        bonus_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> DSparkDraftInputV2:
        return DSparkDraftInputV2(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=seq_lens.to(dtype=torch.int64),
        )

    def _make_next_draft_input_decode(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
    ) -> DSparkDraftInputV2:
        return DSparkDraftInputV2(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
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
            and not is_dflash_sampling_verify_available()
            and self.tp_rank == 0
            and not getattr(self, "_warned_sampling", False)
        ):
            self._warned_sampling = True
            logger.warning(
                "DSpark sampling verification is unavailable on this build; "
                "temperature>0 requests fall back to greedy verification."
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
        if (
            model_worker_batch.extend_lens is None
            or model_worker_batch.prefix_lens is None
        ):
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if model_worker_batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        device = next_token_ids.device
        ctx_lens = torch.tensor(
            model_worker_batch.extend_lens, dtype=torch.int32, device=device
        )
        draft_seq_lens = torch.tensor(
            model_worker_batch.prefix_lens, dtype=torch.int32, device=device
        )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(model_worker_batch.extend_lens)),
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

        if model_worker_batch.forward_mode.is_idle():
            return self._forward_idle(on_publish)

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

        block_hidden = self._run_draft_block(
            bs=bs,
            block_ids=block_ids,
            positions=positions,
            verify_out_cache_loc=verify_out_cache_loc,
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            seq_lens_cpu=model_worker_batch.seq_lens_cpu,
            seq_lens_sum=model_worker_batch.seq_lens_sum,
        )

        candidates, confidence = self._refine_block_markov(
            block_hidden=block_hidden,
            bonus_tokens=draft_input.bonus_tokens,
        )

        verify_input = DSparkVerifyInput(
            draft_token=candidates.reshape(-1),
            positions=positions,
            draft_token_num=block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        model_worker_batch.out_cache_loc = verify_out_cache_loc
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            model_worker_batch, self.target_worker
        )
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        sampling_info = getattr(model_worker_batch, "sampling_info", None)
        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=block_size,
            )
        confident_prefix = self._confident_prefix(confidence)

        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            if self.tp_rank == 0 and not self._sampling_verify_logged:
                self._sampling_verify_logged = True
                logger.info(
                    "DSpark target-only sampling verify is engaged for "
                    "temperature>0 requests."
                )
            accept_len, sampled_bonus = (
                compute_dflash_sampling_correct_drafts_and_bonus(
                    candidates=candidates,
                    next_token_logits=logits_output.next_token_logits,
                    sampling_info=sampling_info,
                    max_top_k=draft_input.max_top_k,
                    uniform_top_k_value=draft_input.uniform_top_k_value,
                )
            )
            # TP consistency: per-rank float nondeterminism in softmax/top_k/top_p
            # can select different sampled tokens on each rank; broadcast rank 0's
            # accept length and bonus token so every rank commits identical tokens
            # and seq_lens stay in lockstep (diverging lengths hang later
            # collectives). DSpark rejects dp_attention, so use the plain TP group.
            if get_tensor_model_parallel_world_size() > 1:
                packed = torch.stack(
                    [accept_len.to(torch.int64), sampled_bonus.to(torch.int64)],
                    dim=0,
                )
                get_tp_group().broadcast(packed, src=0)
                accept_len, sampled_bonus = packed[0], packed[1]
            accept_len = accept_len.to(torch.int64)
            # Losslessness under confidence truncation: the sampling-verify kernel
            # already accepted `accept_len` drafts against the target distribution.
            # When confidence truncates to correct_len < accept_len, the draft at
            # candidates[correct_len + 1] is exactly the token the kernel accepted
            # at that position (prob p(t)); emitting it as the bonus composes with
            # the kernel's rejection-residual (target minus the rejected mass) to
            # reproduce the target distribution, so truncation stays lossless.
            correct_len = torch.minimum(accept_len, confident_prefix.to(torch.int64))
            truncated = correct_len < accept_len
            next_draft = (
                candidates.gather(
                    1, (correct_len + 1).clamp(max=block_size - 1).unsqueeze(1)
                )
                .squeeze(1)
                .to(torch.int64)
            )
            bonus_tokens = torch.where(
                truncated, next_draft, sampled_bonus.to(torch.int64)
            )
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, block_size
            )
            correct_len, _ = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )
            correct_len = torch.minimum(
                correct_len.to(torch.int64), confident_prefix.to(torch.int64)
            )
            bonus_tokens = target_predict.gather(1, correct_len.unsqueeze(1)).squeeze(1)

        commit_lens = correct_len.to(torch.int32) + 1

        out_tokens = torch.empty((bs, block_size), dtype=torch.int64, device=device)
        if block_size > 1:
            out_tokens[:, : block_size - 1].copy_(candidates[:, 1:])
        out_tokens[:, block_size - 1].fill_(0)
        out_tokens.scatter_(
            1, correct_len.unsqueeze(1), bonus_tokens.unsqueeze(1).to(torch.int64)
        )

        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DSpark verify requires target main_hidden states, but got None."
            )
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
        )
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
