"""A tensor parallel worker."""

import logging
import multiprocessing
import pickle
import time
import warnings
from typing import List, Optional

import torch
import torch.distributed as dist

from sglang.global_config import global_config
from sglang.srt.constrained.fsm_cache import FSMCache
from sglang.srt.constrained.jump_forward import JumpForwardCache
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.controller.infer_batch import (
    FINISH_ABORT,
    BaseFinishReason,
    Batch,
    ForwardMode,
    Req,
)
from sglang.srt.managers.controller.model_runner import ModelRunner
from sglang.srt.managers.controller.radix_cache import RadixCache
from sglang.srt.managers.controller.schedule_heuristic import ScheduleHeuristic
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchTokenIDOut,
    FlushCacheReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    get_int_token_logit_bias,
    is_multimodal_model,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger("srt.tp_worker")


class ModelTpServer:
    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        model_overide_args: dict,
    ):
        suppress_other_loggers()

        # Copy arguments
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.schedule_heuristic = server_args.schedule_heuristic
        self.disable_regex_jump_forward = server_args.disable_regex_jump_forward

        # Init model and tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_overide_args=model_overide_args,
        )
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

        if is_multimodal_model(server_args.model_path):
            self.processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = (
            16384
            if server_args.max_prefill_tokens is None
            else server_args.max_prefill_tokens
        )
        self.max_running_requests = (
            self.max_total_num_tokens // 2
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )
        self.max_running_requests = min(
            self.max_running_requests, self.model_runner.req_to_token_pool.size - 1
        )
        self.int_token_logit_bias = torch.tensor(
            get_int_token_logit_bias(self.tokenizer, self.model_config.vocab_size)
        )
        set_random_seed(server_args.random_seed)

        # Print info
        logger.info(
            f"[gpu_id={self.gpu_id}] "
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        # Init cache
        self.tree_cache = RadixCache(
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            disable=server_args.disable_radix_cache,
        )
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.scheduler = ScheduleHeuristic(
            self.schedule_heuristic,
            self.max_running_requests,
            self.max_prefill_tokens,
            self.max_total_num_tokens,
            self.tree_cache,
        )
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool = self.model_runner.token_to_kv_pool

        # Init running status
        self.forward_queue: List[Req] = []
        self.running_batch: Batch = None
        self.out_pyobjs = []
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()

        # Init the FSM cache for constrained generation
        self.regex_fsm_cache = FSMCache(
            server_args.tokenizer_path,
            {
                "tokenizer_mode": server_args.tokenizer_mode,
                "trust_remote_code": server_args.trust_remote_code,
            },
        )
        self.jump_forward_cache = JumpForwardCache()

        # Init new token estimation
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.min_new_token_ratio = min(
            global_config.base_min_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.new_token_ratio = self.min_new_token_ratio
        self.new_token_ratio_decay = global_config.new_token_ratio_decay
        self.new_token_ratio_recovery = global_config.new_token_ratio_recovery

    def exposed_step(self, recv_reqs):
        try:
            # Recv requests
            for recv_req in recv_reqs:
                if isinstance(recv_req, TokenizedGenerateReqInput):
                    self.handle_generate_request(recv_req)
                elif isinstance(recv_req, FlushCacheReq):
                    self.flush_cache()
                elif isinstance(recv_req, AbortReq):
                    self.abort_request(recv_req)
                else:
                    raise ValueError(f"Invalid request: {recv_req}")

            # Forward
            self.forward_step()
        except Exception:
            logger.error("Exception in ModelTpServer:\n" + get_exception_traceback())
            raise

        # Return results
        ret = self.out_pyobjs
        self.out_pyobjs = []
        return ret

    @torch.inference_mode()
    def forward_step(self):
        new_batch = self.get_new_prefill_batch()

        if new_batch is not None:
            # Run a new prefill batch
            self.forward_prefill_batch(new_batch)
            self.cache_filled_batch(new_batch)

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
        else:
            # Run a decode batch
            if self.running_batch is not None:
                # Run a few decode batches continuously for reducing overhead
                for _ in range(global_config.num_continue_decode_steps):
                    self.num_generated_tokens += len(self.running_batch.reqs)
                    self.forward_decode_batch(self.running_batch)

                    # Print stats
                    if self.tp_rank == 0 and self.decode_forward_ct % 40 == 0:
                        self.print_stats()

                    if self.running_batch.is_empty():
                        self.running_batch = None
                        break

                    if self.out_pyobjs and self.running_batch.has_stream():
                        break
            else:
                self.check_memory()
                self.new_token_ratio = global_config.init_new_token_ratio

    def print_stats(self):
        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        throughput = self.num_generated_tokens / (time.time() - self.last_stats_tic)
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()
        logger.info(
            f"[gpu_id={self.gpu_id}] Decode batch. "
            f"#running-req: {len(self.running_batch.reqs)}, "
            f"#token: {num_used}, "
            f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
            f"gen throughput (token/s): {throughput:.2f}, "
            f"#queue-req: {len(self.forward_queue)}"
        )

    def check_memory(self):
        available_size = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        if available_size != self.max_total_num_tokens:
            warnings.warn(
                "Warning: "
                f"available_size={available_size}, max_total_num_tokens={self.max_total_num_tokens}\n"
                "KV cache pool leak detected!"
            )

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        req = Req(recv_req.rid, recv_req.input_text, recv_req.input_ids)
        req.pixel_values = recv_req.pixel_values
        if req.pixel_values is not None:
            req.pad_value = [
                (recv_req.image_hash) % self.model_config.vocab_size,
                (recv_req.image_hash >> 16) % self.model_config.vocab_size,
                (recv_req.image_hash >> 32) % self.model_config.vocab_size,
                (recv_req.image_hash >> 64) % self.model_config.vocab_size,
            ]
            req.image_size = recv_req.image_size
            (
                req.origin_input_ids,
                req.image_offset,
            ) = self.model_runner.model.pad_input_ids(
                req.origin_input_ids_unpadded,
                req.pad_value,
                req.pixel_values.shape,
                req.image_size,
            )
        req.sampling_params = recv_req.sampling_params
        req.return_logprob = recv_req.return_logprob
        req.logprob_start_len = recv_req.logprob_start_len
        req.top_logprobs_num = recv_req.top_logprobs_num
        req.stream = recv_req.stream
        req.tokenizer = self.tokenizer

        # Init regex fsm
        if req.sampling_params.regex is not None:
            req.regex_fsm = self.regex_fsm_cache.query(req.sampling_params.regex)
            if not self.disable_regex_jump_forward:
                req.jump_forward_map = self.jump_forward_cache.query(
                    req.sampling_params.regex
                )

        # Truncate prompts that are too long
        req.origin_input_ids = req.origin_input_ids[: self.model_config.context_len - 1]
        req.sampling_params.max_new_tokens = min(
            req.sampling_params.max_new_tokens,
            self.model_config.context_len - 1 - len(req.origin_input_ids),
            self.max_total_num_tokens - 128 - len(req.origin_input_ids),
        )
        if req.sampling_params.max_new_tokens < 0:
            req.origin_input_ids = req.origin_input_ids[
                : self.max_total_num_tokens - 128
            ]
            logger.error("Request longer than memory pool size, truncated!!!")

        self.forward_queue.append(req)

    def get_new_prefill_batch(self) -> Optional[Batch]:
        running_bs = (
            len(self.running_batch.reqs) if self.running_batch is not None else 0
        )
        if running_bs >= self.max_running_requests:
            return

        # Compute matched prefix length
        for req in self.forward_queue:
            req.input_ids = req.origin_input_ids + req.output_ids
            prefix_indices, last_node = self.tree_cache.match_prefix(req.input_ids)
            if req.return_logprob:
                prefix_indices = prefix_indices[: req.logprob_start_len]
            req.extend_input_len = len(req.input_ids) - len(prefix_indices)
            req.prefix_indices = prefix_indices
            req.last_node = last_node

        # Get priority queue
        self.forward_queue = self.scheduler.get_priority_queue(self.forward_queue)

        # Add requests if there is available space
        can_run_list = []
        new_batch_total_tokens = 0
        new_batch_input_tokens = 0

        available_size = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        if self.running_batch:
            available_size -= sum(
                [
                    (r.sampling_params.max_new_tokens - len(r.output_ids))
                    * self.new_token_ratio
                    for r in self.running_batch.reqs
                ]
            )

        for req in self.forward_queue:
            if req.return_logprob and req.normalized_prompt_logprob is None:
                # Need at least two tokens to compute normalized logprob
                if req.extend_input_len < 2:
                    delta = 2 - req.extend_input_len
                    req.extend_input_len += delta
                    req.prefix_indices = req.prefix_indices[:-delta]
                    if req.image_offset is not None:
                        req.image_offset += delta
            if req.extend_input_len == 0 and req.sampling_params.max_new_tokens > 0:
                # Need at least one token to compute logits
                req.extend_input_len = 1
                req.prefix_indices = req.prefix_indices[:-1]
                if req.image_offset is not None:
                    req.image_offset += 1

            if (
                req.extend_input_len
                + req.sampling_params.max_new_tokens
                + new_batch_total_tokens
                < available_size
                and (
                    req.extend_input_len + new_batch_input_tokens
                    <= self.max_prefill_tokens
                    or len(can_run_list) == 0
                )
            ):
                delta = self.tree_cache.inc_lock_ref(req.last_node)
                available_size += delta

                if not (
                    req.extend_input_len
                    + req.sampling_params.max_new_tokens
                    + new_batch_total_tokens
                    < available_size
                ):
                    # Undo locking
                    delta = self.tree_cache.dec_lock_ref(req.last_node)
                    available_size += delta
                    break
                else:
                    # Add this request to the running batch
                    can_run_list.append(req)
                    new_batch_total_tokens += (
                        req.extend_input_len + req.sampling_params.max_new_tokens
                    )
                    new_batch_input_tokens += req.extend_input_len
            else:
                break

            if running_bs + len(can_run_list) >= self.max_running_requests:
                break

        if len(can_run_list) == 0:
            return None

        # Print stats
        if self.tp_rank == 0:
            hit_tokens = sum(len(x.prefix_indices) for x in can_run_list)
            self.tree_cache_metrics["total"] += (
                hit_tokens + new_batch_input_tokens
            ) / 10**9
            self.tree_cache_metrics["hit"] += hit_tokens / 10**9
            tree_cache_hit_rate = (
                self.tree_cache_metrics["hit"] / self.tree_cache_metrics["total"]
            )
            logger.info(
                f"[gpu_id={self.gpu_id}] Prefill batch. "
                f"#new-seq: {len(can_run_list)}, "
                f"#new-token: {new_batch_input_tokens}, "
                f"#cached-token: {hit_tokens}, "
                f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                f"#running-req: {running_bs}, "
                f"#queue-req: {len(self.forward_queue) - len(can_run_list)}"
            )

        # Return the new batch
        new_batch = Batch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
        )
        self.forward_queue = [x for x in self.forward_queue if x not in can_run_list]
        return new_batch

    def forward_prefill_batch(self, batch: Batch):
        # Build batch tensors
        batch.prepare_for_extend(
            self.model_config.vocab_size, self.int_token_logit_bias
        )

        # Forward and sample the next tokens
        if batch.extend_num_tokens != 0:
            output = self.model_runner.forward(batch, ForwardMode.EXTEND)
            next_token_ids = batch.sample(output.next_token_logits)

            # Move logprobs to cpu
            if output.next_token_logprobs is not None:
                output.next_token_logprobs = output.next_token_logprobs[
                    torch.arange(len(next_token_ids), device=next_token_ids.device),
                    next_token_ids,
                ].tolist()
                output.prefill_token_logprobs = output.prefill_token_logprobs.tolist()
                output.normalized_prompt_logprobs = (
                    output.normalized_prompt_logprobs.tolist()
                )

            next_token_ids = next_token_ids.tolist()
        else:
            next_token_ids = [self.tokenizer.eos_token_id] * len(batch.reqs)

        # Check finish conditions
        pt = 0
        for i, req in enumerate(batch.reqs):
            req.completion_tokens_wo_jump_forward += 1
            req.output_ids.append(next_token_ids[i])
            req.check_finished()

            if req.return_logprob:
                self.add_logprob_return_values(i, req, pt, next_token_ids, output)
                pt += req.extend_input_len

        self.handle_finished_requests(batch)

    def add_logprob_return_values(self, i, req, pt, next_token_ids, output):
        if req.normalized_prompt_logprob is None:
            req.normalized_prompt_logprob = output.normalized_prompt_logprobs[i]

        if req.prefill_token_logprobs is None:
            # If logprob_start_len > 0, then first logprob_start_len prompt tokens will be ignored.
            req.prefill_token_logprobs = list(
                zip(
                    output.prefill_token_logprobs[pt : pt + req.extend_input_len - 1],
                    req.input_ids[-req.extend_input_len + 1 :],
                )
            )
            if req.logprob_start_len == 0:
                req.prefill_token_logprobs = [
                    (None, req.input_ids[0])
                ] + req.prefill_token_logprobs

        if req.last_update_decode_tokens != 0:
            req.decode_token_logprobs.extend(
                list(
                    zip(
                        output.prefill_token_logprobs[
                            pt
                            + req.extend_input_len
                            - req.last_update_decode_tokens : pt
                            + req.extend_input_len
                            - 1
                        ],
                        req.input_ids[-req.last_update_decode_tokens + 1 :],
                    )
                )
            )

        req.decode_token_logprobs.append(
            (output.next_token_logprobs[i], next_token_ids[i])
        )

        if req.top_logprobs_num > 0:
            if req.prefill_top_logprobs is None:
                req.prefill_top_logprobs = output.prefill_top_logprobs[i]
                if req.logprob_start_len == 0:
                    req.prefill_top_logprobs = [None] + req.prefill_top_logprobs

            if req.last_update_decode_tokens != 0:
                req.decode_top_logprobs.extend(
                    output.prefill_top_logprobs[i][-req.last_update_decode_tokens + 1 :]
                )
            req.decode_top_logprobs.append(output.decode_top_logprobs[i])

    def cache_filled_batch(self, batch: Batch):
        req_pool_indices_cpu = batch.req_pool_indices.cpu().numpy()
        for i, req in enumerate(batch.reqs):
            new_prefix_indices, new_last_node = self.tree_cache.cache_req(
                token_ids=tuple(req.origin_input_ids + req.output_ids)[:-1],
                last_uncached_pos=len(req.prefix_indices),
                req_pool_idx=req_pool_indices_cpu[i],
                del_in_memory_pool=False,
                old_last_node=req.last_node,
            )
            req.prefix_indices, req.last_node = new_prefix_indices, new_last_node

    def forward_decode_batch(self, batch: Batch):
        # Check if decode out of memory
        if not batch.check_decode_mem():
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode()
            self.new_token_ratio = new_token_ratio

            logger.info(
                "decode out of memory happened, "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self.forward_queue.extend(retracted_reqs)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if not self.disable_regex_jump_forward:
            # Check for jump-forward
            jump_forward_reqs = batch.check_for_jump_forward(self.model_runner)
            self.forward_queue.extend(jump_forward_reqs)
            if batch.is_empty():
                return

        # Update batch tensors
        self.decode_forward_ct = (self.decode_forward_ct + 1) % (1 << 30)
        batch.prepare_for_decode()

        # Forward and sample the next tokens
        output = self.model_runner.forward(batch, ForwardMode.DECODE)
        next_token_ids = batch.sample(output.next_token_logits)

        # Move logprobs to cpu
        if output.next_token_logprobs is not None:
            next_token_logprobs = output.next_token_logprobs[
                torch.arange(len(next_token_ids), device=next_token_ids.device),
                next_token_ids,
            ].tolist()

        next_token_ids = next_token_ids.tolist()

        # Check finish condition
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req.completion_tokens_wo_jump_forward += 1
            req.output_ids.append(next_token_id)
            req.check_finished()

            if req.return_logprob:
                req.decode_token_logprobs.append(
                    (next_token_logprobs[i], next_token_id)
                )
                if req.top_logprobs_num > 0:
                    req.decode_top_logprobs.append(output.decode_top_logprobs[i])

        self.handle_finished_requests(batch)

    def handle_finished_requests(self, batch: Batch):
        output_rids = []
        output_vids = []
        decoded_texts = []
        output_read_ids = []
        output_read_offsets = []
        output_skip_special_tokens = []
        output_spaces_between_special_tokens = []
        output_meta_info = []
        output_finished_reason: List[BaseFinishReason] = []
        finished_indices = []
        unfinished_indices = []
        for i, req in enumerate(batch.reqs):
            if req.finished():
                finished_indices.append(i)
            else:
                unfinished_indices.append(i)

            if req.finished() or (
                (
                    req.stream
                    and (
                        self.decode_forward_ct % self.stream_interval == 0
                        or len(req.output_ids) == 1
                    )
                )
            ):
                output_rids.append(req.rid)
                output_vids.append(req.vid)
                decoded_texts.append(req.decoded_text)
                read_ids, read_offset = req.init_incremental_detokenize()
                output_read_ids.append(read_ids)
                output_read_offsets.append(read_offset)
                output_skip_special_tokens.append(
                    req.sampling_params.skip_special_tokens
                )
                output_spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )

                meta_info = {
                    "prompt_tokens": len(req.origin_input_ids),
                    "completion_tokens": len(req.output_ids),
                    "completion_tokens_wo_jump_forward": req.completion_tokens_wo_jump_forward,
                    "finish_reason": str(req.finished_reason),
                }
                if req.return_logprob:
                    (
                        meta_info["prefill_token_logprobs"],
                        meta_info["decode_token_logprobs"],
                        meta_info["prefill_top_logprobs"],
                        meta_info["decode_top_logprobs"],
                        meta_info["normalized_prompt_logprob"],
                    ) = (
                        req.prefill_token_logprobs,
                        req.decode_token_logprobs,
                        req.prefill_top_logprobs,
                        req.decode_top_logprobs,
                        req.normalized_prompt_logprob,
                    )
                output_meta_info.append(meta_info)
                output_finished_reason.append(req.finished_reason)

        # Send to detokenizer
        if output_rids:
            self.out_pyobjs.append(
                BatchTokenIDOut(
                    output_rids,
                    output_vids,
                    decoded_texts,
                    output_read_ids,
                    output_read_offsets,
                    output_skip_special_tokens,
                    output_spaces_between_special_tokens,
                    output_meta_info,
                    output_finished_reason,
                )
            )

        # Remove finished reqs
        if finished_indices:
            # Update radix cache
            req_pool_indices_cpu = batch.req_pool_indices.tolist()
            for i in finished_indices:
                req = batch.reqs[i]
                self.tree_cache.cache_req(
                    token_ids=tuple(req.origin_input_ids + req.output_ids)[:-1],
                    last_uncached_pos=len(req.prefix_indices),
                    req_pool_idx=req_pool_indices_cpu[i],
                )

                self.tree_cache.dec_lock_ref(req.last_node)

            # Update batch tensors
            if unfinished_indices:
                batch.filter_batch(unfinished_indices)
            else:
                batch.reqs = []

    def flush_cache(self):
        if len(self.forward_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.tree_cache.reset()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            self.regex_fsm_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool.clear()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
        else:
            warnings.warn(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.forward_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )

    def abort_request(self, recv_req):
        # Delete requests in the waiting queue
        to_del = None
        for i, req in enumerate(self.forward_queue):
            if req.rid == recv_req.rid:
                to_del = i
                break

        if to_del is not None:
            del self.forward_queue[to_del]

        # Delete requests in the running batch
        if self.running_batch:
            for req in self.running_batch.reqs:
                if req.rid == recv_req.rid:
                    req.finished_reason = FINISH_ABORT()
                    break


def run_tp_server(
    gpu_id: int,
    tp_rank: int,
    server_args: ServerArgs,
    nccl_port: int,
    model_overide_args: dict,
):
    """Run a tensor parallel server."""
    try:
        model_server = ModelTpServer(
            gpu_id,
            tp_rank,
            server_args,
            nccl_port,
            model_overide_args,
        )
        tp_cpu_group = model_server.model_runner.tp_group.cpu_group

        while True:
            recv_reqs = broadcast_recv_input(None, tp_rank, tp_cpu_group)
            model_server.exposed_step(recv_reqs)
    except Exception:
        logger.error("Exception in run_tp_server:\n" + get_exception_traceback())
        raise


def launch_tp_servers(
    gpu_ids, tp_rank_range, server_args, nccl_port, model_overide_args
):
    """Launch multiple tensor parallel servers."""
    procs = []
    for i in tp_rank_range:
        proc = multiprocessing.Process(
            target=run_tp_server,
            args=(gpu_ids[i], i, server_args, nccl_port, model_overide_args),
        )
        proc.start()
        procs.append(proc)

    return procs


def broadcast_recv_input(data, rank, dist_group):
    """Broadcast inputs from rank=0 to all other ranks with torch.dist backend."""

    if rank == 0:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long)
            dist.broadcast(tensor_size, src=0, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(list(serialized_data))
            tensor_size = torch.tensor([size], dtype=torch.long)

            dist.broadcast(tensor_size, src=0, group=dist_group)
            dist.broadcast(tensor_data, src=0, group=dist_group)
    else:
        tensor_size = torch.tensor([0], dtype=torch.long)
        dist.broadcast(tensor_size, src=0, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8)
        dist.broadcast(tensor_data, src=0, group=dist_group)

        serialized_data = bytes(tensor_data.tolist())
        data = pickle.loads(serialized_data)
        return data
