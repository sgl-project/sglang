"""Meta data for requests and batches"""

import warnings
from dataclasses import dataclass
from enum import IntEnum, auto
import itertools
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from flashinfer.sampling import top_k_top_p_sampling_from_probs

from sglang.srt.constrained import RegexGuide
from sglang.srt.constrained.jump_forward import JumpForwardMap
from sglang.srt.managers.controller.radix_cache import RadixCache
from sglang.srt.memory_pool import ReqToTokenPool, TokenToKVPool

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the first part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()


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
        self.input_ids = None  # input_ids = origin_input_ids + output_ids

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.decoded_text = ""
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None

        # The number of decoded tokens for token usage report. Note that
        # this does not include the jump forward tokens.
        self.completion_tokens_wo_jump_forward = 0

        # For vision input
        self.pixel_values = None
        self.image_size = None
        self.image_offset = 0
        self.pad_value = None

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
        self.logprob_start_len = 0
        self.top_logprobs_num = 0
        self.normalized_prompt_logprob = None
        self.prefill_token_logprobs = None
        self.prefill_top_logprobs = None
        self.decode_token_logprobs = []
        self.decode_top_logprobs = []
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
            self.finished_reason = FINISH_LENGTH(len(self.output_ids))
            return

        if (
            self.output_ids[-1] == self.tokenizer.eos_token_id
            and not self.sampling_params.ignore_eos
        ):
            self.finished_reason = FINISH_MATCHED_TOKEN(
                matched=self.tokenizer.eos_token_id
            )
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
        prompt_tokens = len(self.origin_input_ids_unpadded)

        if all_ids[prompt_tokens - 1] != self.origin_input_ids_unpadded[-1]:
            # TODO(lsyin): fix token fusion
            warnings.warn(
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
            self.decode_token_logprobs = self.decode_token_logprobs[:k]
            self.decode_top_logprobs = self.decode_top_logprobs[:k]
            self.logprob_start_len = prompt_tokens + k
            self.last_update_decode_tokens = len(self.output_ids) - k

        return True

    def __repr__(self):
        return f"rid(n={self.rid}, " f"input_ids={self.origin_input_ids}, "


@dataclass
class Batch:
    """Store all inforamtion of a batch."""

    # Request, memory pool, and cache
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: TokenToKVPool
    tree_cache: RadixCache

    # Batched arguments to model runner
    input_ids: torch.Tensor = None
    req_pool_indices: torch.Tensor = None
    seq_lens: torch.Tensor = None
    prefix_lens: torch.Tensor = None
    position_ids_offsets: torch.Tensor = None
    out_cache_loc: torch.Tensor = None
    extend_num_tokens: int = None

    # For processing logprobs
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None

    # For multimodal
    pixel_values: List[torch.Tensor] = None
    image_sizes: List[List[int]] = None
    image_offsets: List[int] = None

    # Batched sampling params
    temperatures: torch.Tensor = None
    top_ps: torch.Tensor = None
    top_ks: torch.Tensor = None
    frequency_penalties: torch.Tensor = None
    presence_penalties: torch.Tensor = None
    logit_bias: torch.Tensor = None

    # Sequence Parallel params
    sp_size: int = None
    sp_rank: int = None
    padded_sp_len: int = None

    @classmethod
    def init_new(cls, reqs, req_to_token_pool, token_to_kv_pool, tree_cache,
                 sp_size: int = 1, sp_rank: int = 0):
        return_logprob = any(req.return_logprob for req in reqs)

        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            tree_cache=tree_cache,
            return_logprob=return_logprob,
            sp_size=sp_size,
            sp_rank=sp_rank,
        )

    def is_empty(self):
        return len(self.reqs) == 0

    def has_stream(self) -> bool:
        # Return whether batch has at least 1 streaming request
        return any(r.stream for r in self.reqs)

    def prepare_for_extend(self, vocab_size: int, int_token_logit_bias: torch.Tensor):
        device = "cuda"
        bs = len(self.reqs)
        reqs = self.reqs
        input_ids = [r.input_ids[len(r.prefix_indices) :] for r in reqs]
        prefix_indices = [r.prefix_indices for r in reqs]

        # Handle prefix
        # Note: The flatten input ids with Sequence Parallel is in form of:
        # [req_0_sp_0, req_1_sp_0, ... req_n_sp_0,
        #   req_0_sp_1, ..., req_n_sp_1, padding_sp_1,
        #   ...
        # ]
        # The padding is for collection primitives which needs each candidate to
        # have the same size. Since we don't expect too many requests in SP,
        # the extra compute caused by this is affordable.
        flatten_input_ids = [[] for _ in range(self.sp_size)]
        extend_lens = []
        prefix_lens = []
        seq_lens = []

        req_pool_indices = self.req_to_token_pool.alloc(bs)

        if req_pool_indices is None:
            raise RuntimeError(
                "Out of memory. "
                "Please set a smaller number for `--max-running-requests`."
            )

        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        for i in range(bs):
            for sp_rank in range(self.sp_size):
                ids = input_ids[i]
                local_slice = _get_local_token_slices(sp_rank, self.sp_size,
                                                      len(ids))
                flatten_input_ids[sp_rank].extend(ids[local_slice])
            extend_lens.append(len(input_ids[i]))

            if len(prefix_indices[i]) == 0:
                prefix_lens.append(0)
            else:
                prefix_lens.append(len(prefix_indices[i]))
                self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
                    : len(prefix_indices[i])
                ] = prefix_indices[i]

            seq_lens.append(prefix_lens[-1] + extend_lens[-1])
        # For sequence parallel, add padding zeros for each rank.
        padded_sp_len = max(len(ids) for ids in flatten_input_ids)
        for flatten_ids in flatten_input_ids:
            if len(flatten_ids) < padded_sp_len:
                flatten_ids.extend([0] * (padded_sp_len - len(flatten_ids)))
        flatten_input_ids = list(itertools.chain(flatten_input_ids))
        self.padded_sp_len = padded_sp_len

        position_ids_offsets = torch.zeros((bs,), dtype=torch.int32, device=device)

        # Allocate memory
        seq_lens, prefix_lens = np.array(seq_lens), np.array(prefix_lens)
        extend_num_tokens = seq_lens.sum() - prefix_lens.sum()
        if self.sp_size > 1:
            extend_seq_lens = seq_lens - prefix_lens
            # FIXME(yonghao): _extend_num_tokens -> extend_num_tokens once kv cache store is ready for SP
            extend_local_token_nums = _get_local_token_nums(self.sp_rank,
                                                            self.sp_size,
                                                            extend_seq_lens)
            _extend_num_tokens = int(np.sum(extend_local_token_nums))

        out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)
        if out_cache_loc is None:
            self.tree_cache.evict(extend_num_tokens, self.token_to_kv_pool.free)
            out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)

            if out_cache_loc is None:
                print("Prefill out of memory. This should never happen.")
                self.tree_cache.pretty_print()
                exit()

        pt = 0
        for i in range(bs):
            extend_len = extend_lens[i]
            if self.sp_size > 1:
                # FIXME(yonghao): _extend_len > extend_len once SP Attn is ready
                _extend_len = extend_local_token_nums[i]
            self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
                prefix_lens[i] : prefix_lens[i] + extend_len
            ] = out_cache_loc[pt : pt + extend_len]
            pt += extend_len

        # Handle logit bias but only allocate when needed
        logit_bias = None
        for i in range(bs):
            if reqs[i].sampling_params.dtype == "int":
                if logit_bias is None:
                    logit_bias = torch.zeros(
                        (bs, vocab_size), dtype=torch.float32, device=device
                    )
                logit_bias[i] = int_token_logit_bias

        # Set fields
        self.input_ids = torch.tensor(
            flatten_input_ids, dtype=torch.int32, device=device
        )
        self.pixel_values = [r.pixel_values for r in reqs]
        self.image_sizes = [r.image_size for r in reqs]
        self.image_offsets = [
            r.image_offset - p_len for r, p_len in zip(reqs, prefix_lens)
        ]
        self.req_pool_indices = req_pool_indices
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        self.prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        self.position_ids_offsets = position_ids_offsets
        self.extend_num_tokens = extend_num_tokens
        self.out_cache_loc = out_cache_loc
        self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]

        self.temperatures = torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            device=device,
        ).view(-1, 1)
        self.top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float, device=device
        )
        self.top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int, device=device
        )
        self.frequency_penalties = torch.tensor(
            [r.sampling_params.frequency_penalty for r in reqs],
            dtype=torch.float,
            device=device,
        )
        self.presence_penalties = torch.tensor(
            [r.sampling_params.presence_penalty for r in reqs],
            dtype=torch.float,
            device=device,
        )
        self.logit_bias = logit_bias

    def check_decode_mem(self):
        bs = len(self.reqs)
        if self.token_to_kv_pool.available_size() >= bs:
            return True

        self.tree_cache.evict(bs, self.token_to_kv_pool.free)

        if self.token_to_kv_pool.available_size() >= bs:
            return True

        return False

    def retract_decode(self):
        sorted_indices = [i for i in range(len(self.reqs))]
        # TODO(lsyin): improve the priority of retraction
        sorted_indices.sort(
            key=lambda i: (
                len(self.reqs[i].output_ids),
                -len(self.reqs[i].origin_input_ids),
            ),
            reverse=True,
        )

        retracted_reqs = []
        seq_lens_cpu = self.seq_lens.cpu().numpy()
        req_pool_indices_cpu = self.req_pool_indices.cpu().numpy()
        while self.token_to_kv_pool.available_size() < len(self.reqs):
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            # TODO: apply more fine-grained retraction
            last_uncached_pos = len(req.prefix_indices)
            token_indices = self.req_to_token_pool.req_to_token[
                req_pool_indices_cpu[idx]
            ][last_uncached_pos : seq_lens_cpu[idx]]
            self.token_to_kv_pool.free(token_indices)

            # release the last node
            self.tree_cache.dec_lock_ref(req.last_node)

            req.prefix_indices = None
            req.last_node = None
            req.extend_input_len = 0

            # For incremental logprobs
            req.last_update_decode_tokens = 0
            req.logprob_start_len = 10**9

        self.filter_batch(sorted_indices)

        return retracted_reqs

    def check_for_jump_forward(self, model_runner):
        jump_forward_reqs = []
        filter_indices = [i for i in range(len(self.reqs))]

        req_pool_indices_cpu = None

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

                    # insert the old request into tree_cache
                    if req_pool_indices_cpu is None:
                        req_pool_indices_cpu = self.req_pool_indices.tolist()
                    self.tree_cache.cache_req(
                        token_ids=cur_all_ids,
                        last_uncached_pos=len(req.prefix_indices),
                        req_pool_idx=req_pool_indices_cpu[i],
                    )

                    # unlock the last node
                    self.tree_cache.dec_lock_ref(req.last_node)

                    # re-applying image padding
                    if req.pixel_values is not None:
                        (
                            req.origin_input_ids,
                            req.image_offset,
                        ) = model_runner.model.pad_input_ids(
                            req.origin_input_ids_unpadded,
                            req.pad_value,
                            req.pixel_values.shape,
                            req.image_size,
                        )

                    jump_forward_reqs.append(req)
                    filter_indices.remove(i)

        if len(filter_indices) < len(self.reqs):
            self.filter_batch(filter_indices)

        return jump_forward_reqs

    def prepare_for_decode(self, input_ids=None):
        if input_ids is None:
            input_ids = [
                r.output_ids[-1] if r.output_ids else r.input_ids[-1] for r in self.reqs
            ]
        self.seq_lens.add_(1)
        input_ids_sp = [[] for _ in range(self.sp_size)]
        seq_lens_cpu = (self.seq_lens - self.prefix_lens).cpu().numpy()
        for sp_rank in range(self.sp_size):
            # TODO(yonghao): double check moving the seq lens adds one to above.
            input_ids_sp[sp_rank].append(input_ids[
                get_decode_indices(sp_rank, self.sp_size, seq_lens_cpu)
            ])
        padded_sp_len = max(len(ids) for ids in input_ids_sp)
        for flatten_ids in input_ids_sp:
            if len(flatten_ids) < padded_sp_len:
                flatten_ids.extend([0] * (padded_sp_len - len(flatten_ids)))
        self.padded_sp_len = padded_sp_len
        
        input_ids = itertools.chain(input_ids_sp)
        self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        self.prefix_lens = None

        # Alloc mem
        bs = len(self.reqs)
        if self.sp_size > 1:
            sp_local_indices = get_decode_indices(self.sp_rank, self.sp_size,
                                                  seq_lens_cpu)
            # FIXME(yonghao): _bs -> bs once SP kv cache store is ready
            _bs = len(sp_local_indices)

        self.out_cache_loc = self.token_to_kv_pool.alloc(bs)

        if self.out_cache_loc is None:
            print("Decode out of memory. This should never happen.")
            self.tree_cache.pretty_print()
            exit()

        if self.sp_size > 1:
            return  # FIXME(yonghao): remove it once SP kv cache store is ready
            local_req_indices = self.req_pool_indices[sp_local_indices]
            # NOTE(yonghao): here the seqlen is still the total seq len but not
            # the local lens
            local_req_seqlens = self.seq_lens[sp_local_indices]
            # local_req_local_lens = _get_local_token_nums(self.sp_rank, self.sp_size, local_req_seqlens)
            self.req_to_token_pool.req_to_token[
                local_req_indices, local_req_seqlens - 1
            ] = self.out_cache_loc
            return
        self.req_to_token_pool.req_to_token[
            self.req_pool_indices, self.seq_lens - 1
        ] = self.out_cache_loc

    def filter_batch(self, unfinished_indices: List[int]):
        self.reqs = [self.reqs[i] for i in unfinished_indices]
        new_indices = torch.tensor(unfinished_indices, dtype=torch.int32, device="cuda")
        self.seq_lens = self.seq_lens[new_indices]
        self.input_ids = None
        self.req_pool_indices = self.req_pool_indices[new_indices]
        self.prefix_lens = None
        self.position_ids_offsets = self.position_ids_offsets[new_indices]
        self.out_cache_loc = None
        self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in unfinished_indices]
        self.return_logprob = any(req.return_logprob for req in self.reqs)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "frequency_penalties",
            "presence_penalties",
            "logit_bias",
        ]:
            self_val = getattr(self, item, None)
            if self_val is not None:  # logit_bias can be None
                setattr(self, item, self_val[new_indices])

    def merge(self, other: "Batch"):
        self.reqs.extend(other.reqs)

        self.req_pool_indices = torch.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.concat([self.seq_lens, other.seq_lens])
        self.prefix_lens = None
        self.position_ids_offsets = torch.concat(
            [self.position_ids_offsets, other.position_ids_offsets]
        )
        self.out_cache_loc = None
        self.top_logprobs_nums.extend(other.top_logprobs_nums)
        self.return_logprob = any(req.return_logprob for req in self.reqs)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "frequency_penalties",
            "presence_penalties",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            setattr(self, item, torch.concat([self_val, other_val]))

        # logit_bias can be None
        if self.logit_bias is not None or other.logit_bias is not None:
            vocab_size = (
                self.logit_bias.shape[1]
                if self.logit_bias is not None
                else other.logit_bias.shape[1]
            )
            if self.logit_bias is None:
                self.logit_bias = torch.zeros(
                    (len(self.reqs), vocab_size), dtype=torch.float32, device="cuda"
                )
            if other.logit_bias is None:
                other.logit_bias = torch.zeros(
                    (len(other.reqs), vocab_size), dtype=torch.float32, device="cuda"
                )
            self.logit_bias = torch.concat([self.logit_bias, other.logit_bias])

    def sample(self, logits: torch.Tensor):
        # Post process logits
        logits = logits.contiguous()
        logits.div_(self.temperatures)
        if self.logit_bias is not None:
            logits.add_(self.logit_bias)

        has_regex = any(req.regex_fsm is not None for req in self.reqs)
        if has_regex:
            allowed_mask = torch.empty_like(logits[0], dtype=torch.bool)
            for i, req in enumerate(self.reqs):
                if req.regex_fsm is not None:
                    allowed_mask.zero_()
                    allowed_mask[
                        req.regex_fsm.get_next_instruction(req.regex_fsm_state).tokens
                    ] = 1
                    logits[i].masked_fill_(~allowed_mask, float("-inf"))

        # TODO(lmzheng): apply penalty
        probs = torch.softmax(logits, dim=-1)
        try:
            max_top_k_round, batch_size = 32, probs.shape[0]
            uniform_samples = torch.rand(
                (max_top_k_round, batch_size), device=probs.device
            )
            batch_next_token_ids, _ = top_k_top_p_sampling_from_probs(
                probs, uniform_samples, self.top_ks, self.top_ps
            )
        except RuntimeError as e:
            warnings.warn(f"Ignore errors in sampling: {e}")
            batch_next_token_ids = torch.argmax(probs, dim=-1)

        if has_regex:
            batch_next_token_ids_cpu = batch_next_token_ids.cpu().numpy()
            for i, req in enumerate(self.reqs):
                if req.regex_fsm is not None:
                    req.regex_fsm_state = req.regex_fsm.get_next_state(
                        req.regex_fsm_state, batch_next_token_ids_cpu[i]
                    )

        return batch_next_token_ids


@dataclass
class InputMetadata:
    """Store all inforamtion of a forward pass."""

    forward_mode: ForwardMode
    batch_size: int
    total_num_tokens: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: TokenToKVPool

    # For extend
    extend_seq_lens: torch.Tensor
    extend_start_loc: torch.Tensor
    extend_no_prefix: bool

    # Output location of the KV cache
    out_cache_loc: torch.Tensor = None

    # Output options
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None

    # Trition attention backend
    triton_max_seq_len: int = 0
    triton_max_extend_len: int = 0
    triton_start_loc: torch.Tensor = None
    triton_prefix_lens: torch.Tensor = None

    # FlashInfer attention backend
    flashinfer_prefill_wrapper_ragged: "BatchPrefillWithRaggedKVCacheWrapper" = None
    flashinfer_prefill_wrapper_paged: "BatchPrefillWithPagedKVCacheWrapper" = None
    flashinfer_decode_wrapper: "BatchDecodeWithPagedKVCacheWrapper" = None

    # For Sequence Parallel
    sp_rank: int = None
    sp_size: int = None
    local_token_indices: np.ndarray
    sp_to_normal_indices: np.ndarray
    sp_local_token_length: int

    @classmethod
    def create(
        cls,
        model_runner,
        forward_mode,
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
        top_logprobs_nums=None,
        return_logprob=False,
        skip_flashinfer_init=False,
        padded_sp_len=None,
    ):
        if not skip_flashinfer_init and not model_runner.server_args.disable_flashinfer:
            init_flashinfer_args(
                forward_mode,
                model_runner,
                req_pool_indices,
                seq_lens,
                prefix_lens,
                model_runner.flashinfer_decode_wrapper,
            )

        batch_size = len(req_pool_indices)

        if forward_mode == ForwardMode.DECODE:
            positions = ((seq_lens - 1) + position_ids_offsets).to(torch.int64)
            extend_seq_lens = extend_start_loc = extend_no_prefix = None
            if not model_runner.server_args.disable_flashinfer:
                # This variable is not needed in this case,
                # we do not compute it to make it compatbile with cuda graph.
                total_num_tokens = None
            else:
                total_num_tokens = int(torch.sum(seq_lens))
        else:
            seq_lens_cpu = seq_lens.cpu().numpy()
            prefix_lens_cpu = prefix_lens.cpu().numpy()
            position_ids_offsets_cpu = position_ids_offsets.cpu().numpy()
            positions = torch.tensor(
                np.concatenate(
                    [
                        np.arange(
                            prefix_lens_cpu[i] + position_ids_offsets_cpu[i],
                            seq_lens_cpu[i] + position_ids_offsets_cpu[i],
                        )
                        for i in range(batch_size)
                    ],
                    axis=0,
                ),
                device="cuda",
            )
            extend_seq_lens = seq_lens - prefix_lens
            extend_start_loc = torch.zeros_like(seq_lens)
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
            extend_no_prefix = torch.all(prefix_lens == 0)
            total_num_tokens = int(torch.sum(seq_lens))

        sp_rank = model_runner.sp_rank
        sp_size = model_runner.sp_size
        if sp_size > 1:
            # During the runtime, we should use positions[local_token_indices]
            # to get positions for each SP shard.
            extend_seq_lens_cpu = (seq_lens - prefix_lens).cpu().numpy()
            if forward_mode == ForwardMode.DECODE:
                local_token_indices = get_decode_indices(sp_rank, sp_size,
                                                         extend_seq_lens_cpu)
                sp_to_normal_indices = sp_to_normal_indices_decode(
                    sp_size, extend_seq_lens_cpu, padded_sp_len
                )
            else:
                local_token_indices = get_prefill_indices(sp_rank, sp_size,
                                                          extend_seq_lens_cpu)
                sp_to_normal_indices = sp_to_normal_indices_prefill(
                    sp_size, extend_seq_lens_cpu, padded_sp_len
                )
        else:
            local_token_indices = np.arange(positions.numel())
            sp_to_normal_indices = np.arange(extend_seq_lens.numel())
        sp_local_token_length = len(local_token_indices)

        ret = cls(
            forward_mode=forward_mode,
            batch_size=batch_size,
            total_num_tokens=total_num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            positions=positions,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            extend_no_prefix=extend_no_prefix,
            return_logprob=return_logprob,
            top_logprobs_nums=top_logprobs_nums,
            flashinfer_prefill_wrapper_ragged=model_runner.flashinfer_prefill_wrapper_ragged,
            flashinfer_prefill_wrapper_paged=model_runner.flashinfer_prefill_wrapper_paged,
            flashinfer_decode_wrapper=model_runner.flashinfer_decode_wrapper,
            sp_rank=sp_rank,
            sp_size=sp_size,
            local_token_indices=local_token_indices,
            sp_to_normal_indices=sp_to_normal_indices,
            sp_local_token_length=sp_local_token_length,
        )

        if model_runner.server_args.disable_flashinfer:
            (
                ret.triton_max_seq_len,
                ret.triton_max_extend_len,
                ret.triton_start_loc,
                ret.triton_prefix_lens,
            ) = init_triton_args(forward_mode, seq_lens, prefix_lens)

        return ret


def init_flashinfer_args(
    forward_mode,
    model_runner,
    req_pool_indices,
    seq_lens,
    prefix_lens,
    flashinfer_decode_wrapper,
):
    """Init auxiliary variables for FlashInfer attention backend."""
    num_qo_heads = model_runner.model_config.num_attention_heads // model_runner.tp_size
    num_kv_heads = model_runner.model_config.get_num_kv_heads(model_runner.tp_size)
    head_dim = model_runner.model_config.head_dim
    batch_size = len(req_pool_indices)

    if forward_mode == ForwardMode.DECODE:
        paged_kernel_lens = seq_lens
    else:
        paged_kernel_lens = prefix_lens

    kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
    req_pool_indices_cpu = req_pool_indices.cpu().numpy()
    paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
    kv_indices = torch.cat(
        [
            model_runner.req_to_token_pool.req_to_token[
                req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]
            ]
            for i in range(batch_size)
        ],
        dim=0,
    ).contiguous()
    kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

    if forward_mode == ForwardMode.DECODE:
        flashinfer_decode_wrapper.end_forward()
        flashinfer_decode_wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
        )
    else:
        # extend part
        qo_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

        model_runner.flashinfer_prefill_wrapper_ragged.end_forward()
        model_runner.flashinfer_prefill_wrapper_ragged.begin_forward(
            qo_indptr,
            qo_indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
        )

        # cached part
        model_runner.flashinfer_prefill_wrapper_paged.end_forward()
        model_runner.flashinfer_prefill_wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
        )


def init_triton_args(forward_mode, seq_lens, prefix_lens):
    """Init auxiliary variables for triton attention backend."""
    batch_size = len(seq_lens)
    max_seq_len = int(torch.max(seq_lens))
    start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    if forward_mode == ForwardMode.DECODE:
        max_extend_len = None
    else:
        extend_seq_lens = seq_lens - prefix_lens
        max_extend_len = int(torch.max(extend_seq_lens))

    return max_seq_len, max_extend_len, start_loc, prefix_lens


def get_prefill_indices(sp_rank, sp_size, extend_seq_lens: np.ndarray,
                        extend_start_loc):
    """
    Get indices from the normal layout to the sequence parallel layout of all
    requests.
    """
    # For the first few ranks, they have one more token to compute
    sp_req_lens = _get_local_token_nums(sp_rank, sp_size, extend_seq_lens)
    # the offset of each request in the batch. Only the first few ranks may get
    # 1 more token (for each). For sp_rank=r, therere r peers ahread (0-based),
    # each will get one token
    sp_in_req_offset = (
        extend_seq_lens // sp_size * sp_rank +
        np.clip(extend_seq_lens % sp_size, a_min=None, a_max=sp_rank)
    )
    sp_req_start = extend_start_loc + sp_in_req_offset
    sp_indices = np.concatenate([
        np.arange(s, s + l) for s, l in zip(sp_req_start, sp_req_lens)
    ])
    return sp_indices


def _get_local_token_nums(sp_rank, sp_size,
                          extend_seq_lens: Union[int, np.ndarray]):
    """Get the number of tokens in this SP. Padding is not considered."""
    has_remainder = (extend_seq_lens % sp_size) > sp_rank
    return extend_seq_lens // sp_size + has_remainder


def get_decode_indices(sp_rank, sp_size, seq_lens: np.ndarray, offset=0):
    """Get Indices from the normal layout to the sequence parallel layout."""
    return np.nonzero((seq_lens % sp_size) == sp_rank) + offset


def _get_local_token_slices(sp_rank, sp_size, seq_len: int):
    """Get the SP local slice for a single request's extended input ids."""
    start = seq_len // sp_size * sp_rank + min(seq_len % sp_size, sp_rank)
    length = _get_local_token_nums(sp_rank, sp_size, seq_len)
    return slice(start, start + length)


def sp_to_normal_indices_prefill(sp_size, extend_seq_lens: np.ndarray,
                                 padded_sp_len: int):
    """
    Indices from the Sequence Parallel layout (padded) to the normal layout.
    """
    indices = []
    sp_offset = [padded_sp_len * sp_rank for sp_rank in range(sp_size)]
    sp_local_token_nums = [
        _get_local_token_nums(sp_rank, sp_size, extend_seq_lens)
        for sp_rank in range(sp_size)
    ]
    for req_id in range(len(extend_seq_lens)):
        for sp_rank in range(sp_size):
            sp_len = sp_local_token_nums[sp_rank][req_id]
            indices.extend(range(sp_offset[sp_rank], sp_offset + sp_len))
            sp_offset[sp_rank] += sp_len
    return np.asarray(indices)


def sp_to_normal_indices_decode(sp_size, seq_lens_cpu: np.ndarray,
                                 padded_sp_len: int):
    """
    Indices from the Sequence Parallel layout (padded) to the normal layout.
    """
    req_sp_rank = seq_lens_cpu % sp_size
    req_sp_offset = req_sp_rank * padded_sp_len
    for sp_rank in range(sp_size):
        local_reqs = req_sp_rank == sp_rank
        req_sp_index = np.cumsum(local_reqs) - 1
        req_sp_offset += req_sp_index * local_reqs  # mask out reqs not here.
    return req_sp_offset


def _debug_normal_to_sp_indices(mode, sp_size, seq_lens, sp_padded_len):
    """(Debug only) Indices from normal layout to the SP layout (padded)."""
    get_indices_fn = (get_decode_indices
                      if mode == ForwardMode.DECODE else get_prefill_indices)
    def get_offset(sp_rank):
        offset = sp_rank * sp_padded_len
        if mode == ForwardMode.DECODE:
            return offset
        return [offset] * len(seq_lens)
    indices = [
        get_indices_fn(sp_rank, sp_size, seq_lens, get_offset(sp_rank))
        for sp_rank in range(sp_size)
    ]
    return indices


def _debug_normal_to_sp(indices, output_tensor, tensor):
    """
    Use the indices generated above to translate from a normal layout to a
    SP layout (padded). Due to the padding, `output_tensor`'s shape is different
    from the input `tensor`'s.
    """
    for idxs in indices:
        output_tensor[idxs] = tensor
    output_tensor = output_tensor.contiguous()
    return output_tensor
