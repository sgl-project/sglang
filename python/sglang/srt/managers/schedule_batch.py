from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Meta data for requests and batches"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.global_config import global_config
from sglang.srt.constrained import RegexGuide
from sglang.srt.constrained.jump_forward import JumpForwardMap
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

if TYPE_CHECKING:
    from sglang.srt.layers.sampler import SampleOutput


INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# Put some global args for easy access
global_server_args_dict = {
    "disable_flashinfer": False,
    "disable_flashinfer_sampling": False,
    "triton_attention_reduce_in_fp32": False,
    "enable_mla": False,
}


logger = logging.getLogger(__name__)


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def __str__(self) -> str:
        return f"FINISH_MATCHED_TOKEN: {self.matched}"


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def __str__(self) -> str:
        return f"FINISH_LENGTH: {self.length}"


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def __str__(self) -> str:
        return f"FINISH_MATCHED_STR: {self.matched}"


class FINISH_ABORT(BaseFinishReason):
    def __init__(self):
        super().__init__(is_error=True)

    def __str__(self) -> str:
        return "FINISH_ABORT"


class Req:
    """Store all inforamtion of a request."""

    def __init__(self, rid, origin_input_text, origin_input_ids):
        # Input and output info
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = origin_input_ids  # Before image padding
        self.origin_input_ids = origin_input_ids
        self.output_ids = []  # Each decode stage's output ids
        self.fill_ids = None  # fill_ids = origin_input_ids + output_ids

        # Memory info
        self.req_pool_idx = None

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.vid = 0  # version id to sync decode status with in detokenizer_manager
        self.decoded_text = ""
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None

        # The number of decoded tokens for token usage report. Note that
        # this does not include the jump forward tokens.
        self.completion_tokens_wo_jump_forward = 0

        # For vision input
        self.pixel_values = None
        self.image_sizes = None
        self.image_offsets = None
        self.pad_value = None
        self.modalities = None

        # Prefix info
        self.extend_input_len = 0
        self.prefix_indices = []
        self.last_node = None

        # Sampling parameters
        self.sampling_params = None
        self.stream = False

        # Check finish
        self.tokenizer = None
        self.finished_reason = None

        # Logprobs
        self.return_logprob = False
        self.embedding = None
        self.logprob_start_len = 0
        self.top_logprobs_num = 0
        self.normalized_prompt_logprob = None
        self.input_token_logprobs = None
        self.input_top_logprobs = None
        self.output_token_logprobs = []
        self.output_top_logprobs = []
        # The tokens is prefilled but need to be considered as decode tokens
        # and should be updated for the decode logprobs
        self.last_update_decode_tokens = 0

        # Constrained decoding
        self.regex_fsm: RegexGuide = None
        self.regex_fsm_state: int = 0
        self.jump_forward_map: JumpForwardMap = None

    # whether request reached finished condition
    def finished(self) -> bool:
        return self.finished_reason is not None

    def init_next_round_input(self, tree_cache: Optional[BasePrefixCache] = None):
        self.fill_ids = self.origin_input_ids + self.output_ids
        if tree_cache is not None:
            self.prefix_indices, self.last_node = tree_cache.match_prefix(
                rid=self.rid, key=self.adjust_max_prefix_ids()
            )
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)

    def adjust_max_prefix_ids(self):
        self.fill_ids = self.origin_input_ids + self.output_ids
        input_len = len(self.fill_ids)

        # FIXME: To work around some bugs in logprob computation, we need to ensure each
        # request has at least one token. Later, we can relax this requirement and use `input_len`.
        max_prefix_len = input_len - 1

        if self.sampling_params.max_new_tokens > 0:
            # Need at least one token to compute logits
            max_prefix_len = min(max_prefix_len, input_len - 1)

        if self.return_logprob:
            if self.normalized_prompt_logprob is None:
                # Need at least two tokens to compute normalized logprob
                max_prefix_len = min(max_prefix_len, input_len - 2)
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)

        max_prefix_len = max(max_prefix_len, 0)
        return self.fill_ids[:max_prefix_len]

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )

        all_ids = self.origin_input_ids_unpadded + self.output_ids
        return all_ids[self.surr_offset :], self.read_offset - self.surr_offset

    def get_next_inc_detokenization(self):
        if self.tokenizer is None:
            return False, ""
        read_ids, read_offset = self.init_incremental_detokenize()
        surr_ids = read_ids[:read_offset]

        surr_text = self.tokenizer.decode(
            surr_ids,
            skip_special_tokens=self.sampling_params.skip_special_tokens,
            spaces_between_special_tokens=self.sampling_params.spaces_between_special_tokens,
        )
        new_text = self.tokenizer.decode(
            read_ids,
            skip_special_tokens=self.sampling_params.skip_special_tokens,
            spaces_between_special_tokens=self.sampling_params.spaces_between_special_tokens,
        )

        if len(new_text) > len(surr_text) and not new_text.endswith("�"):
            return True, new_text[len(surr_text) :]

        return False, ""

    def check_finished(self):
        if self.finished():
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            return

        last_token_id = self.output_ids[-1]

        matched_eos = last_token_id in self.sampling_params.stop_token_ids

        if self.tokenizer is not None:
            matched_eos |= last_token_id == self.tokenizer.eos_token_id

        if matched_eos and not self.sampling_params.ignore_eos:
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
            return

        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                    return

    def jump_forward_and_retokenize(self, jump_forward_str, next_state):
        if self.origin_input_text is None:
            # Recovering text can only use unpadded ids
            self.origin_input_text = self.tokenizer.decode(
                self.origin_input_ids_unpadded
            )

        all_text = self.origin_input_text + self.decoded_text + jump_forward_str
        all_ids = self.tokenizer.encode(all_text)
        if not all_ids:
            logger.warning("Encoded all_text resulted in empty all_ids")
            return False

        prompt_tokens = len(self.origin_input_ids_unpadded)
        if prompt_tokens > len(all_ids):
            logger.warning("prompt_tokens is larger than encoded all_ids")
            return False

        if all_ids[prompt_tokens - 1] != self.origin_input_ids_unpadded[-1]:
            # TODO(lsyin): fix token fusion
            logger.warning(
                "Token fusion between input and output, try to avoid this by removing the space at the end of the input."
            )
            return False

        old_output_ids = self.output_ids
        self.output_ids = all_ids[prompt_tokens:]
        self.decoded_text = self.decoded_text + jump_forward_str
        self.surr_offset = prompt_tokens
        self.read_offset = len(all_ids)

        # NOTE: A trick to reduce the surrouding tokens decoding overhead
        for i in range(0, INIT_INCREMENTAL_DETOKENIZATION_OFFSET):
            surr_text_ = self.tokenizer.decode(
                all_ids[self.read_offset - i : self.read_offset]
            )
            if not surr_text_.endswith("�"):
                self.surr_offset = self.read_offset - i
                break

        self.regex_fsm_state = next_state

        if self.return_logprob:
            # For fast-forward part's logprobs
            k = 0
            for i, old_id in enumerate(old_output_ids):
                if old_id == self.output_ids[i]:
                    k = k + 1
                else:
                    break
            self.output_token_logprobs = self.output_token_logprobs[:k]
            self.output_top_logprobs = self.output_top_logprobs[:k]
            self.logprob_start_len = prompt_tokens + k
            self.last_update_decode_tokens = len(self.output_ids) - k

        return True

    def __repr__(self):
        return f"rid(n={self.rid}, " f"input_ids={self.origin_input_ids}, "


@dataclass
class ScheduleBatch:
    """Store all inforamtion of a batch."""

    # Request, memory pool, and cache
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: BaseTokenToKVPool
    tree_cache: BasePrefixCache

    # Batched arguments to model runner
    input_ids: torch.Tensor = None
    req_pool_indices: torch.Tensor = None
    seq_lens: torch.Tensor = None
    position_ids_offsets: torch.Tensor = None
    out_cache_loc: torch.Tensor = None
    extend_num_tokens: int = None

    # For mixed chunekd prefill
    prefix_lens_cpu: List[int] = None

    # For processing logprobs
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None

    @classmethod
    def init_new(cls, reqs, req_to_token_pool, token_to_kv_pool, tree_cache):
        return_logprob = any(req.return_logprob for req in reqs)

        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            tree_cache=tree_cache,
            return_logprob=return_logprob,
        )

    def batch_size(self):
        return len(self.reqs) if self.reqs is not None else 0

    def is_empty(self):
        return len(self.reqs) == 0

    def has_stream(self) -> bool:
        # Return whether batch has at least 1 streaming request
        return any(r.stream for r in self.reqs)

    def alloc_req_slots(self, num_reqs):
        req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
        if req_pool_indices is None:
            raise RuntimeError(
                "Out of memory. "
                "Please set a smaller number for `--max-running-requests`."
            )
        return req_pool_indices

    def alloc_token_slots(self, num_tokens: int):
        out_cache_loc = self.token_to_kv_pool.alloc(num_tokens)

        if out_cache_loc is None:
            if self.tree_cache is not None:
                self.tree_cache.evict(num_tokens, self.token_to_kv_pool.free)
                out_cache_loc = self.token_to_kv_pool.alloc(num_tokens)

            if out_cache_loc is None:
                logger.error("Prefill out of memory. Try to lower your batch size.")
                if self.tree_cache is not None:
                    self.tree_cache.pretty_print()
                exit(1)

        return out_cache_loc

    def prepare_for_extend(self, vocab_size: int):
        bs = self.batch_size()
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []

        # Allocate memory
        req_pool_indices_cpu = self.alloc_req_slots(bs)
        out_cache_loc = self.alloc_token_slots(extend_num_tokens)

        pt = 0
        for i, req in enumerate(reqs):
            req.req_pool_idx = req_pool_indices_cpu[i]
            pre_len, seq_len = len(req.prefix_indices), len(req.fill_ids)
            ext_len = seq_len - pre_len
            seq_lens.append(seq_len)

            if pre_len > 0:
                self.req_to_token_pool.req_to_token[req.req_pool_idx][
                    :pre_len
                ] = req.prefix_indices

            self.req_to_token_pool.req_to_token[req.req_pool_idx][pre_len:seq_len] = (
                out_cache_loc[pt : pt + ext_len]
            )
            pt += ext_len

        # Set fields
        with torch.device("cuda"):
            self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int32)
            self.req_pool_indices = torch.tensor(req_pool_indices_cpu)
            self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
            self.position_ids_offsets = torch.zeros((bs,), dtype=torch.int64)

        self.extend_num_tokens = extend_num_tokens
        self.out_cache_loc = out_cache_loc
        self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
        self.prefix_lens_cpu = [len(r.prefix_indices) for r in reqs]

        self.sampling_info = SamplingBatchInfo.from_schedule_batch(self, vocab_size)

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        prefix_lens_cpu = [len(r.prefix_indices) for r in self.reqs]
        prefix_lens_cpu.extend(
            [
                len(r.origin_input_ids) + len(r.output_ids) - 1
                for r in running_batch.reqs
            ]
        )

        for req in running_batch.reqs:
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.extend_input_len = 1

        input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])
        extend_num_tokens = self.extend_num_tokens + running_batch.batch_size()
        self.merge(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc
        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens_cpu = prefix_lens_cpu

    def check_decode_mem(self):
        bs = self.batch_size()
        if self.token_to_kv_pool.available_size() >= bs:
            return True

        self.tree_cache.evict(bs, self.token_to_kv_pool.free)

        if self.token_to_kv_pool.available_size() >= bs:
            return True

        return False

    def retract_decode(self):
        sorted_indices = [i for i in range(len(self.reqs))]

        # TODO(lsyin): improve retraction policy for radix cache
        sorted_indices.sort(
            key=lambda i: (
                len(self.reqs[i].output_ids),
                -len(self.reqs[i].origin_input_ids),
            ),
            reverse=True,
        )

        retracted_reqs = []
        seq_lens_cpu = self.seq_lens.cpu().numpy()
        while (
            self.token_to_kv_pool.available_size()
            < len(sorted_indices) * global_config.retract_decode_steps
        ):
            if len(sorted_indices) == 1:
                # Corner case: only one request left
                assert (
                    self.token_to_kv_pool.available_size() > 0
                ), "No space left for only one request"
                break

            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            if isinstance(self.tree_cache, ChunkCache):
                # ChunkCache does not have eviction
                token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                    : seq_lens_cpu[idx]
                ]
                self.token_to_kv_pool.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)
                del self.tree_cache.entries[req.rid]
            else:
                # TODO: apply more fine-grained retraction
                last_uncached_pos = len(req.prefix_indices)
                token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                    last_uncached_pos : seq_lens_cpu[idx]
                ]
                self.token_to_kv_pool.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)

                # release the last node
                self.tree_cache.dec_lock_ref(req.last_node)

                # NOTE(lsyin): we should use the newly evictable memory instantly.
                residual_size = (
                    len(sorted_indices) * global_config.retract_decode_steps
                    - self.token_to_kv_pool.available_size()
                )
                residual_size = max(0, residual_size)
                self.tree_cache.evict(residual_size, self.token_to_kv_pool.free)

            req.prefix_indices = []
            req.last_node = None
            req.extend_input_len = 0

            # For incremental logprobs
            req.last_update_decode_tokens = 0
            req.logprob_start_len = 10**9

        self.filter_batch(sorted_indices)

        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens + global_config.retract_decode_steps * len(self.reqs)
        ) / total_max_new_tokens
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio

    def check_for_jump_forward(self, model_runner):
        jump_forward_reqs = []
        filter_indices = [i for i in range(len(self.reqs))]

        for i, req in enumerate(self.reqs):
            if req.jump_forward_map is not None:
                jump_forward_bytes = req.jump_forward_map.jump_forward_byte(
                    req.regex_fsm_state
                )
                if jump_forward_bytes is not None and len(jump_forward_bytes) > 1:
                    suffix_bytes = []
                    continuation_range = range(0x80, 0xC0)
                    cur_state = req.regex_fsm_state
                    while (
                        len(jump_forward_bytes)
                        and jump_forward_bytes[0][0] in continuation_range
                    ):
                        # continuation bytes
                        byte_edge = jump_forward_bytes.pop(0)
                        suffix_bytes.append(byte_edge[0])
                        cur_state = byte_edge[1]

                    suffix_tokens = [f"<0x{hex(b)[2:].upper()}>" for b in suffix_bytes]
                    suffix_ids = req.tokenizer.convert_tokens_to_ids(suffix_tokens)

                    # Current ids, for cache and revert
                    cur_all_ids = tuple(req.origin_input_ids + req.output_ids)[:-1]
                    cur_output_ids = req.output_ids

                    req.output_ids.extend(suffix_ids)
                    decode_res, new_text = req.get_next_inc_detokenization()
                    if not decode_res:
                        req.output_ids = cur_output_ids
                        continue

                    (
                        jump_forward_str,
                        next_state,
                    ) = req.jump_forward_map.jump_forward_symbol(cur_state)

                    # Make the incrementally decoded text part of jump_forward_str
                    # so that the UTF-8 will not corrupt
                    jump_forward_str = new_text + jump_forward_str
                    if not req.jump_forward_and_retokenize(
                        jump_forward_str, next_state
                    ):
                        req.output_ids = cur_output_ids
                        continue

                    # The decode status has diverged from detokenizer_manager
                    req.vid += 1

                    # insert the old request into tree_cache
                    self.tree_cache.cache_finished_req(req, cur_all_ids)

                    # re-applying image padding
                    if req.pixel_values is not None:
                        (
                            req.origin_input_ids,
                            req.image_offsets,
                        ) = model_runner.model.pad_input_ids(
                            req.origin_input_ids_unpadded,
                            req.pad_value,
                            req.pixel_values,
                            req.image_sizes,
                        )

                    jump_forward_reqs.append(req)
                    filter_indices.remove(i)

        self.filter_batch(filter_indices)

        return jump_forward_reqs

    def prepare_for_decode(self, input_ids=None):
        if input_ids is None:
            input_ids = [
                r.output_ids[-1] if r.output_ids else r.origin_input_ids[-1]
                for r in self.reqs
            ]
        else:
            self.sampling_info.penalizer_orchestrator.cumulate_input_tokens(input_ids)

        self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        self.seq_lens.add_(1)

        # Alloc mem
        bs = self.batch_size()
        self.out_cache_loc = self.alloc_token_slots(bs)

        self.req_to_token_pool.req_to_token[
            self.req_pool_indices, self.seq_lens - 1
        ] = self.out_cache_loc

        self.sampling_info.update_regex_vocab_mask(self)

    def filter_batch(self, unfinished_indices: List[int]):
        if unfinished_indices is None or len(unfinished_indices) == 0:
            # Filter out all requests
            self.reqs = []
            return

        if len(unfinished_indices) == len(self.reqs):
            # No need to filter
            return

        self.reqs = [self.reqs[i] for i in unfinished_indices]
        new_indices = torch.tensor(unfinished_indices, dtype=torch.int32, device="cuda")
        self.seq_lens = self.seq_lens[new_indices]
        self.input_ids = None
        self.req_pool_indices = self.req_pool_indices[new_indices]
        self.position_ids_offsets = self.position_ids_offsets[new_indices]
        self.out_cache_loc = None
        self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in unfinished_indices]
        self.return_logprob = any(req.return_logprob for req in self.reqs)

        self.sampling_info.filter(unfinished_indices, new_indices)

    def merge(self, other: "ScheduleBatch"):
        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge(other.sampling_info)

        self.reqs.extend(other.reqs)

        self.req_pool_indices = torch.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.concat([self.seq_lens, other.seq_lens])
        self.position_ids_offsets = torch.concat(
            [self.position_ids_offsets, other.position_ids_offsets]
        )
        self.out_cache_loc = None
        self.top_logprobs_nums.extend(other.top_logprobs_nums)
        self.return_logprob = any(req.return_logprob for req in self.reqs)

    def check_sample_results(self, sample_output: SampleOutput):
        if not torch.all(sample_output.success):
            probs = sample_output.probs
            batch_next_token_ids = sample_output.batch_next_token_ids
            logging.warning("Sampling failed, fallback to top_k=1 strategy")
            probs = probs.masked_fill(torch.isnan(probs), 0.0)
            argmax_ids = torch.argmax(probs, dim=-1)
            batch_next_token_ids = torch.where(
                sample_output.success, batch_next_token_ids, argmax_ids
            )
            sample_output.probs = probs
            sample_output.batch_next_token_ids = batch_next_token_ids

        return sample_output.batch_next_token_ids
