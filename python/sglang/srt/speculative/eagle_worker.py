import logging
import os
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    draft_tp_context,
    fast_topk,
    generate_token_bitmask,
    get_last_loc_large_page_size_large_top_k,
    load_token_map,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    empty_context,
    get_available_gpu_memory,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_hip = is_hip()
_is_npu = is_npu()
# Set HIP_DIAG_ENABLE=1 to enable verbose GPU-sync diagnostic logging (dev only).
# Off by default — .tolist() calls force GPU stream sync on every forward pass.
_hip_diag = _is_hip and os.environ.get("HIP_DIAG_ENABLE", "0") == "1"

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


class EAGLEWorker(TpModelWorker):

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
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.is_p_eagle = self.speculative_algorithm.is_p_eagle()

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            logger.info("EAGLE draft worker init: entering TpModelWorker super().__init__")
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )
            logger.info("EAGLE draft worker init: TpModelWorker super().__init__ complete")

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                logger.info("EAGLE draft worker init: sharing target embed + lm_head")
                embed, head = self.target_worker.model_runner.model.get_embed_and_head()
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                embed = None
                skip_target_embed_share = os.getenv(
                    "SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE", ""
                ).lower() in {"1", "true", "yes", "on"}
                if skip_target_embed_share:
                    logger.info(
                        "EAGLE draft worker init: skipping target embedding share "
                        "because SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE is enabled"
                    )
                else:
                    # BUG-FIX: Do not share embeddings from a GGUF-quantised target.
                    # Q1_0_G128 dequantised weights have ~70x smaller magnitude than
                    # the full-precision fp16 weights that EAGLE3 was trained with.
                    # Feeding near-zero embeddings into the midlayer produces a nearly
                    # uniform logit distribution → 0% spec-decode acceptance.
                    # The draft model's own safetensors embed_tokens are the correct
                    # full-precision weights; just leave them in place.
                    _target_quant = getattr(
                        self.target_worker.model_runner.server_args, "quantization", None
                    )
                    if _target_quant == "gguf":
                        logger.info(
                            "EAGLE draft worker init: skipping embed share — target "
                            "uses GGUF quantisation (Q1_0 magnitudes are ~70x smaller "
                            "than the fp16 embeddings EAGLE3 was trained with). "
                            "Using draft-local full-precision embed_tokens instead."
                        )
                    else:
                        try:
                            logger.info(
                                "EAGLE draft worker init: attempting target embedding share"
                            )
                            embed = self.target_worker.model_runner.model.get_embed()
                            self.draft_model_runner.model.set_embed(embed)
                            logger.info(
                                "EAGLE draft worker init: target embedding share complete"
                            )
                        except torch.OutOfMemoryError:
                            logger.warning(
                                "Skipping target embedding share for EAGLE3 draft because "
                                "the target embedding export exhausted GPU memory. "
                                "Continuing with the draft-local embedding."
                            )
                            torch.cuda.empty_cache()
                        except Exception as exc:
                            logger.warning(
                                "Skipping target embedding share for EAGLE3 draft and "
                                "continuing with the draft-local embedding: %s",
                                exc,
                            )
                if embed is None:
                    logger.info("EAGLE draft worker init: using draft-local embedding")

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_id is not None:
                embed_device = (
                    embed.device
                    if embed is not None
                    else next(self.draft_model_runner.model.parameters()).device
                )
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(
                    embed_device
                )

        else:
            embed, head = self.target_worker.model_runner.model.get_embed_and_head()
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            self.eagle_use_aux_hidden_state = True
            eagle_config = getattr(
                self.draft_model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        logger.info("EAGLE draft worker init: initializing attention backend and cuda graphs")
        if self.is_p_eagle:
            logger.info(
                "P_EAGLE mode: parallel drafting enabled (depth-1 tree, K=%d candidates per pass)",
                self.topk,
            )
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()
        logger.info("EAGLE draft worker init: attention backend and cuda graphs ready")

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # Capture extend
        if self.draft_extend_attn_backend and not _is_npu:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                self
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            (
                logits_output,
                next_token_ids,
                seq_lens_cpu,
                can_run_cuda_graph,
            ) = self.forward_target_extend(batch)
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.forward_draft_extend(
                    batch,
                    logits_output.hidden_states,
                    next_token_ids,
                    seq_lens_cpu,
                    logits_output.mm_input_embeds,
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )
        else:
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                spec_info = self.draft(batch)
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )

            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                if (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                ):
                    # decode is not finished
                    self.forward_draft_extend_after_decode(batch)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def check_forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        local_need_forward = batch.spec_info.verified_id.shape[0] > 0
        if not self.server_args.enable_dp_attention:
            return local_need_forward

        global_need_forward = torch.tensor(
            [
                (local_need_forward),
            ],
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            global_need_forward, group=get_tp_group().cpu_group
        )
        global_need_forward_cnt = global_need_forward[0].item()
        need_forward = global_need_forward_cnt > 0
        return need_forward

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor], bool]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            seq_lens_cpu: CPU copy of sequence lengths for the draft prefill path.
            can_run_cuda_graph: Whether the target prefill ran with cuda graph.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        if _hip_diag:
            logger.info(
                "EAGLE target extend entry: batch_size=%s seq_lens_sum=%s "
                "extend_lens=%s prefix_lens=%s forward_mode=%s",
                batch.batch_size(),
                batch.seq_lens_sum,
                batch.extend_lens,
                batch.prefix_lens,
                batch.forward_mode,
            )
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        if _hip_diag:
            logger.info(
                "EAGLE target extend start: batch_size=%s seq_lens_sum=%s",
                batch.batch_size(),
                batch.seq_lens_sum,
            )
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        if _hip_diag:
            logger.info(
                "EAGLE target extend done: next_token_shape=%s hidden_states_shape=%s",
                tuple(next_token_ids.shape),
                None
                if logits_output.hidden_states is None
                else tuple(logits_output.hidden_states.shape),
            )
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
            batch_result.can_run_cuda_graph,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        if self.page_size == 1:
            alloc_len_per_decode = self.speculative_num_steps * self.topk
            # TODO: We only need self.speculative_num_steps - 1 * topk cache loc
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        if self.page_size > 1 and self.topk > 1:
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (self.topk - 1)
            target_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            # When source_cache_loc is not needed, simply skip
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + self.page_size),
        )

        if self.page_size > 1 and self.topk > 1:
            if duplicate_cache_len > 0:
                self.draft_model_runner.token_to_kv_pool.move_kv_cache(
                    target_cache_loc, source_cache_loc
                )
            # Remove padded slots
            # TODO: We only need self.speculative_num_steps - 1 cache loc
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self.model_config.hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def draft(self, batch: ScheduleBatch):
        # Parse args
        if _hip_diag:
            logger.info(
                "EAGLE draft ENTRY: req_pool_indices=%s dtype=%s device=%s",
                batch.req_pool_indices.tolist() if batch.req_pool_indices is not None and batch.req_pool_indices.numel() > 0 else batch.req_pool_indices,
                batch.req_pool_indices.dtype if batch.req_pool_indices is not None else None,
                batch.req_pool_indices.device if batch.req_pool_indices is not None else None,
            )
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)
        if _hip_diag:
            logger.info(
                "EAGLE draft POST-PREPROCESS: req_pool_indices=%s",
                batch.req_pool_indices.tolist() if batch.req_pool_indices is not None and batch.req_pool_indices.numel() > 0 else batch.req_pool_indices,
            )

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        batch.return_hidden_states = False

        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        can_cuda_graph = (
            self.cuda_graph_runner
            and self.cuda_graph_runner.can_run(forward_batch)
            and not self.is_p_eagle  # P_EAGLE uses a different forward path
        )
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Multi-step EAGLE3 needs explicit attn backend init
                self.draft_attn_backend.init_forward_metadata(forward_batch)

            if self.is_p_eagle and not forward_batch.forward_mode.is_idle():
                # P_EAGLE: single parallel forward pass instead of sequential loop
                parent_list, top_scores_index, draft_tokens = (
                    self.draft_forward_p_eagle(forward_batch)
                )
            else:
                # EAGLE / EAGLE3: sequential multi-step draft
                parent_list, top_scores_index, draft_tokens = self.draft_forward(
                    forward_batch
                )

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # P_EAGLE produces a depth-1 tree; spec_steps for tree building = 1
        effective_spec_steps = 1 if self.is_p_eagle else self.speculative_num_steps

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            effective_spec_steps,
            self.speculative_num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=effective_spec_steps,
            topk=self.topk,
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        # TODO: We only need self.speculative_num_steps - 1 cache loc
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            # This is a temporary fix for the case that the user is using standalone
            # speculative decoding and the draft model architecture is gpt-oss. gpt-oss
            # rope kernel needs cache_loc to be contiguous.
            if (
                self.server_args.speculative_algorithm == "STANDALONE"
                and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens

    def draft_forward_p_eagle(self, forward_batch: ForwardBatch):
        """P_EAGLE parallel drafting: single forward pass producing K candidates.

        Instead of EAGLE3's sequential multi-step loop, P_EAGLE generates all K
        draft candidates in one shot using mask_hidden for slots 1..K-1.

        The draft model's prepare_p_eagle_inputs() builds:
          - slot 0:      real last token + fused hidden state from target
          - slots 1..K-1: mask token + learned mask_hidden parameter

        This produces a depth-1 tree: K independent branches from the root.
        organize_draft_results handles len(parents_list)==1 by returning
        parent_list=empty, which build_tree_kernel_efficient interprets as a
        flat fan-out tree.
        """
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        # --- Retrieve target model's hidden states and the verified token ---
        hidden_states = spec_info.hidden_states       # [bs, hidden*3] or [bs, 1, hidden*3]
        verified_id = spec_info.verified_id            # [bs] last accepted token

        if hidden_states is None or hidden_states.numel() == 0:
            # Idle batch — produce empty results matching sequential path format
            bs = forward_batch.batch_size
            device = forward_batch.input_ids.device
            empty_parents = torch.empty(bs, 0, device=device, dtype=torch.long)
            empty_scores = torch.empty(bs, 0, device=device)
            empty_tokens = torch.empty(bs, 0, device=device, dtype=torch.long)
            return empty_parents, empty_scores, empty_tokens

        bs = verified_id.shape[0]
        device = verified_id.device

        # After prefill, hidden_states may contain all prefill tokens (num_tokens > bs).
        # P_EAGLE only needs the last token's fused hidden state per request.
        if hidden_states.dim() == 2 and hidden_states.shape[0] > bs:
            hidden_states = hidden_states[-bs:]

        # Reshape hidden_states to [bs, 1, feat_dim] for prepare_p_eagle_inputs
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)

        # Call the model's P_EAGLE input builder
        draft_model = self.draft_model_runner.model
        token_ids = verified_id.unsqueeze(-1)  # [bs, 1]
        embeds, projected_hidden = draft_model.prepare_p_eagle_inputs(
            token_ids, hidden_states, k=self.topk
        )
        # embeds:            [bs, K, hidden_size]
        # projected_hidden:  [bs, K, hidden_size]

        # Flatten for the draft model forward: [bs*K, hidden_size]
        flat_embeds = embeds.reshape(-1, embeds.shape[-1])
        flat_hidden = projected_hidden.reshape(-1, projected_hidden.shape[-1])

        # Set up forward_batch for the single P_EAGLE forward pass
        # All K positions are at the SAME sequence position (parallel alternatives)
        positions_per_req = spec_info.positions  # [bs*topk] from preprocess
        forward_batch.input_ids = draft_model.embed_tokens(
            torch.cat([token_ids, torch.full(
                (bs, self.topk - 1), draft_model.mask_token_id,
                dtype=token_ids.dtype, device=device
            )], dim=1).flatten()
        ) if False else token_ids.flatten().repeat_interleave(self.topk)
        # Actually: set input_embeds and hidden_states directly
        forward_batch.input_ids = torch.cat([
            token_ids,
            torch.full(
                (bs, self.topk - 1), draft_model.mask_token_id,
                dtype=token_ids.dtype, device=device
            )
        ], dim=1).flatten()  # [bs*K]

        # Override hidden states with the projected P_EAGLE hidden states
        spec_info.hidden_states = flat_hidden

        # Use the first-step out_cache_loc
        out_cache_loc = forward_batch.out_cache_loc
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        forward_batch.out_cache_loc = out_cache_loc[:, :, 0].reshape(-1)

        # P-EAGLE: K candidates per request share the same KV cache.
        # Replicate batch metadata so the model_runner's standard attention
        # init builds correct kv_indptr/kv_indices for bs*K decode tokens.
        # spec_info is kept (model reads hidden_states from it) but its
        # kv_indptr/kv_indices stay None — triton_backend falls through to
        # the standard build-from-batch-metadata path.
        bs_k = forward_batch.batch_size * self.topk
        forward_batch.batch_size = bs_k
        forward_batch.seq_lens = forward_batch.seq_lens.repeat_interleave(self.topk)
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu.repeat(self.topk) if forward_batch.seq_lens_cpu is not None else None
        forward_batch.seq_lens_sum = forward_batch.seq_lens.sum().item()
        forward_batch.req_pool_indices = forward_batch.req_pool_indices.repeat_interleave(self.topk)

        # Run ONE forward pass through the draft model
        logits_output = self.draft_model_runner.forward(
            forward_batch, skip_attn_backend_init=False
        ).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "p_eagle_draft: NaN in logits")

        # Get top-K predictions from each of the K parallel positions
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)  # [bs*K, vocab]
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)  # [bs*K, topk]

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        # Reshape to [bs, K, topk] and take the FIRST token from each parallel slot
        topk_p = topk_p.reshape(bs, self.topk, self.topk)
        topk_index = topk_index.reshape(bs, self.topk, self.topk)

        # For the depth-1 tree: each of K parallel positions contributes its top-1 token
        # Scores: probability of each candidate
        # Parents: all point to root (-1)
        draft_scores = topk_p[:, :, 0]    # [bs, K] — top-1 prob from each slot
        draft_tokens = topk_index[:, :, 0]  # [bs, K] — top-1 token from each slot

        # Build tree_info in the format organize_draft_results expects:
        # score_list[0]: (bs, 1, topk) — the initial fan-out scores
        # token_list[0]: (bs, topk) — the token IDs
        # parents_list[0]: (bs, topk+1) — parent indices, starting with -1 for root
        score_list = [draft_scores.unsqueeze(1)]  # [bs, 1, K]
        token_list = [draft_tokens]                # [bs, K]
        parents_list = [
            torch.arange(-1, self.topk, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(bs, -1)
        ]  # [bs, K+1]

        parent_list, top_scores_index, final_draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, final_draft_tokens

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker
        pass

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        if _hip_diag:
            logger.info(
                "EAGLE verify ENTRY: req_pool_indices=%s seq_lens=%s",
                batch.req_pool_indices.tolist() if batch.req_pool_indices is not None and batch.req_pool_indices.numel() > 0 else batch.req_pool_indices,
                batch.seq_lens.tolist() if batch.seq_lens is not None and batch.seq_lens.numel() > 0 else None,
            )
        seq_lens_pre_verify = batch.seq_lens.clone()
        spec_info.prepare_for_verify(batch, self.page_size)
        if _hip_diag:
            logger.info(
                "EAGLE verify POST-PREPARE: req_pool_indices=%s seq_lens=%s",
                batch.req_pool_indices.tolist() if batch.req_pool_indices is not None and batch.req_pool_indices.numel() > 0 else batch.req_pool_indices,
                batch.seq_lens.tolist() if batch.seq_lens is not None and batch.seq_lens.numel() > 0 else None,
            )
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")

        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Diagnostic: show acceptance for first few verify steps
        import sys
        _accept_cpu = res.accept_length_per_req_cpu
        if not hasattr(self, '_diag_ct'):
            self._diag_ct = 0
        if self._diag_ct < 5:
            _draft_tok = spec_info.draft_token[:20].tolist() if spec_info.draft_token is not None else None
            print(f"[E3-VERIFY] accept_per_req={_accept_cpu} draft_tokens={_draft_tok}", file=sys.stderr, flush=True)
            self._diag_ct += 1

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        # Under DP attention, some ranks can be IDLE during target verify and never
        # initialize mamba forward metadata for this step.
        if batch.forward_mode.is_idle():
            return

        accepted_length = (
            torch.tensor(
                res.accept_length_per_req_cpu,
                device=logits_output.hidden_states.device,
                dtype=torch.int64,
            )
            + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        # prepend 0 to the cumulative_accepted_lengths
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # res.accepted_indices.shape[0] > 0 skips DP attn idle batch
        if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
            # accepted_indices=[0,2,3,4,5,7,9,10,11], accepted_length=[4, 3, 2], cumulative_accepted_lengths=[4, 7, 9]
            # first_token_indices_per_req=prepend(0, accepted_indices[cumulative_accepted_lengths[:-1]]) = [0, 5, 10]
            # last_token_indices_per_req=accepted_indices[cumulative_accepted_lengths - 1] = [4, 9, 11] (last token ID of each req)
            # max_relative_indices_per_req = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
            # first_token_indices_per_req = res.accepted_indices[accepted_indices_start]
            accepted_steps = (
                res.accepted_indices[cumulative_accepted_lengths - 1]
                - accepted_indices_offset
            )
        else:
            accepted_steps = accepted_length - 1

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=seq_lens_cpu
        )
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        if _is_hip and forward_batch.extend_seq_lens_cpu is not None:
            recomputed_positions = torch.cat(
                [
                    torch.arange(
                        int(prefix_len),
                        int(prefix_len) + int(extend_len),
                        dtype=torch.int64,
                    )
                    for prefix_len, extend_len in zip(
                        forward_batch.extend_prefix_lens_cpu,
                        forward_batch.extend_seq_lens_cpu,
                    )
                ],
                dim=0,
            ).to(forward_batch.input_ids.device, non_blocking=False)
            forward_batch.positions = recomputed_positions
        if _hip_diag:
            logger.info(
                "EAGLE draft extend start: forward_mode=%s hidden_states_shape=%s next_token_shape=%s seq_lens_cpu=%s",
                forward_batch.forward_mode,
                tuple(hidden_states.shape),
                tuple(next_token_ids.shape),
                None if seq_lens_cpu is None else seq_lens_cpu.tolist(),
            )
            positions_cpu = forward_batch.positions.detach().to(device="cpu", dtype=torch.int64)
            logger.info(
                "EAGLE draft extend metadata: extend_prefix_lens_cpu=%s extend_seq_lens_cpu=%s positions_shape=%s positions_head=%s positions_tail=%s positions_min=%s positions_max=%s",
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                tuple(forward_batch.positions.shape),
                positions_cpu[: min(16, positions_cpu.numel())].tolist(),
                positions_cpu[-min(16, positions_cpu.numel()) :].tolist(),
                None if positions_cpu.numel() == 0 else int(positions_cpu.min().item()),
                None if positions_cpu.numel() == 0 else int(positions_cpu.max().item()),
            )
        logits_output = self.draft_model_runner.forward(forward_batch).logits_output
        if _hip_diag:
            torch.cuda.synchronize()
            logger.info(
                "EAGLE draft extend forward done: logits_shape=%s hidden_states_shape=%s",
                tuple(logits_output.next_token_logits.shape),
                None
                if logits_output.hidden_states is None
                else tuple(logits_output.hidden_states.shape),
            )
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        self.capture_for_decode(logits_output, forward_batch.spec_info)

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        assert isinstance(batch.spec_info, EagleDraftInput)
        # Backup fields that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
        req_pool_indices_backup = batch.req_pool_indices
        accept_length_backup = batch.spec_info.accept_length
        return_logprob_backup = batch.return_logprob
        # extend_lens/prefix_lens are modified by prepare_extend_after_decode;
        # restore them so stale draft-extend values don't leak into the next
        # TARGET_VERIFY or EXTEND batch.
        extend_lens_backup = list(batch.extend_lens) if batch.extend_lens is not None else None
        extend_num_tokens_backup = batch.extend_num_tokens
        prefix_lens_backup = list(batch.prefix_lens) if batch.prefix_lens is not None else None

        input_is_idle = batch.forward_mode.is_idle()

        if not input_is_idle and batch.spec_info.verified_id.numel() == 0:
            batch = batch.copy()
            batch.prepare_for_idle()
            hidden_size = (
                self.model_config.hidden_size * 3
                if self.speculative_algorithm.is_eagle3()
                and self.eagle_use_aux_hidden_state
                else self.model_config.hidden_size
            )
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        batch.spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.spec_info.num_tokens_for_logprob_per_req = 1
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        batch.forward_mode = (
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()

        # Run
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
            forward_batch.spec_info.topk_p, forward_batch.spec_info.topk_index = (
                logits_output.topk_p,
                logits_output.topk_index,
            )
            # P_EAGLE preserves target model's fused aux hidden states (hidden*3)
            if not self.is_p_eagle:
                forward_batch.spec_info.hidden_states = logits_output.hidden_states
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if not forward_batch.forward_mode.is_idle():
                self.draft_model_runner.attn_backend.init_forward_metadata(
                    forward_batch
                )
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            self.capture_for_decode(logits_output, forward_batch.spec_info)

        maybe_detect_nan(
            logits_output.next_token_logits,
            f"draft_extend_after_decode (cuda_graph={can_cuda_graph})",
        )

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = (
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup
        # Restore extend_lens/prefix_lens to prevent stale draft-extend values from
        # leaking into the next EXTEND or TARGET_VERIFY batch build.
        if extend_lens_backup is not None:
            batch.extend_lens = extend_lens_backup
            batch.extend_num_tokens = extend_num_tokens_backup
        if prefix_lens_backup is not None:
            batch.prefix_lens = prefix_lens_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        logits = logits_output.next_token_logits
        # Always run softmax+topk on GPU. The _is_hip CPU path was a workaround
        # for a broken sgl_kernel wheel; fast_topk falls back to torch.topk which
        # runs fine on ROCm. Copying 151K logits to CPU every decode step was the
        # primary throughput bottleneck (~2s/token on gfx1031).
        probs = torch.softmax(logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        # P_EAGLE needs the target model's fused aux hidden states (hidden*3)
        # for prepare_p_eagle_inputs each round. The draft model outputs only
        # hidden_size, so overwriting would lose the fused representation.
        if not self.is_p_eagle:
            draft_input.hidden_states = logits_output.hidden_states

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message


@torch.compile(dynamic=True, disable=_is_npu)
def get_last_loc_large_page_size_top_k_1(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens,
    speculative_num_steps: int,
):
    prefix_lens = seq_lens
    seq_lens = prefix_lens + speculative_num_steps
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return prefix_lens, seq_lens, last_loc
