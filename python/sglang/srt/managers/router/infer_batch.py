from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy as np
import torch
from sglang.srt.managers.router.radix_cache import RadixCache
from sglang.srt.memory_pool import ReqToTokenPool, TokenToKVPool
from sglang.srt.speculate.speculate_engine import SpeculateEngine, SpeculateTries


class ForwardMode(Enum):
    PREFILL = auto()
    EXTEND = auto()
    DECODE = auto()


class FinishReason(Enum):
    LENGTH = auto()
    EOS_TOKEN = auto()
    STOP_STR = auto()


class Req:
    def __init__(self, rid, input_text, input_ids):
        self.rid = rid
        self.input_text = input_text
        self.input_ids = input_ids
        self.output_ids = []

        # Since jump forward may retokenize the prompt with partial outputs,
        # we maintain the original prompt length to report the correct usage.
        self.prompt_tokens = len(input_ids)
        # The number of decoded tokens for token usage report. Note that
        # this does not include the jump forward tokens.
        self.completion_tokens_wo_jump_forward = 0

        # For vision input
        self.pixel_values = None
        self.image_size = None
        self.image_offset = 0
        self.pad_value = None

        self.sampling_params = None
        self.return_logprob = False
        self.logprob_start_len = 0
        self.stream = False

        self.tokenizer = None
        self.finished = False
        self.finish_reason = None
        self.hit_stop_str = None

        self.extend_input_len = 0
        self.prefix_indices = []
        self.last_node = None

        self.logprob = None
        self.token_logprob = None
        self.normalized_logprob = None

        # For constrained decoding
        self.regex_fsm = None
        self.regex_fsm_state = 0
        self.jump_forward_map = None
        # TODO(lsyin): rename this and improve style
        self.output_and_jump_forward_str = ""

        # For speculative decoding
        # FIXME: speculate engine inside per request or globally?
        self.speculate_engine: SpeculateEngine = None
        self.speculate_tries: SpeculateTries = None

    def max_new_tokens(self):
        return self.sampling_params.max_new_tokens

    def jump_forward_and_retokenize(self, jump_forward_str, next_state):
        old_output_str = self.tokenizer.decode(self.output_ids)
        # FIXME: This logic does not really solve the problem of determining whether
        # there should be a leading space.
        first_token = self.tokenizer.convert_ids_to_tokens(self.output_ids[0])
        first_token = (
            first_token.decode() if isinstance(first_token, bytes) else first_token
        )
        if first_token.startswith("▁"):
            old_output_str = " " + old_output_str
        new_input_string = (
            self.input_text
            + self.output_and_jump_forward_str
            + old_output_str
            + jump_forward_str
        )
        new_input_ids = self.tokenizer.encode(new_input_string)
        if self.pixel_values is not None:
            # NOTE: This is a hack because the old input_ids contains the image padding
            jump_forward_tokens_len = len(self.tokenizer.encode(jump_forward_str))
        else:
            jump_forward_tokens_len = (
                len(new_input_ids) - len(self.input_ids) - len(self.output_ids)
            )

        # print("=" * 100)
        # print(f"Catch jump forward:\n{jump_forward_str}")
        # print(self.tokenizer.convert_ids_to_tokens(self.input_ids))
        # print(self.tokenizer.convert_ids_to_tokens(new_input_ids))

        self.input_ids = new_input_ids
        self.output_ids = []
        self.sampling_params.max_new_tokens = max(
            self.sampling_params.max_new_tokens - jump_forward_tokens_len, 0
        )
        self.regex_fsm_state = next_state
        self.output_and_jump_forward_str = (
            self.output_and_jump_forward_str + old_output_str + jump_forward_str
        )

        # print(f"Output and jump forward str:\n{self.output_and_jump_forward_str}")
        # print("*" * 100)

    def check_finished(self):
        if self.finished:
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished = True
            self.finish_reason = FinishReason.LENGTH
            return

        if (
            self.output_ids[-1] == self.tokenizer.eos_token_id
            and self.sampling_params.ignore_eos == False
        ):
            self.finished = True
            self.finish_reason = FinishReason.EOS_TOKEN
            return

        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str:
                    self.finished = True
                    self.finish_reason = FinishReason.STOP_STR
                    self.hit_stop_str = stop_str
                    return

    def __repr__(self):
        return f"rid(n={self.rid}, " f"input_ids={self.input_ids}, "


@dataclass
class Batch:
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: TokenToKVPool
    tree_cache: RadixCache

    # batched arguments to model runner
    input_ids: torch.Tensor = None  # flatten input_ids for extend/decode part
    req_pool_indices: torch.Tensor = None
    seq_lens: torch.Tensor = None
    prefix_lens: torch.Tensor = None
    position_ids_offsets: torch.Tensor = None
    out_cache_loc: torch.Tensor = None
    out_cache_cont_start: torch.Tensor = None
    out_cache_cont_end: torch.Tensor = None
    return_logprob: bool = False

    # for multimodal
    pixel_values: List[torch.Tensor] = None
    image_sizes: List[List[int]] = None
    image_offsets: List[int] = None

    # other arguments for control
    output_ids: torch.Tensor = None
    extend_num_tokens: int = None

    # batched sampling params
    temperatures: torch.Tensor = None
    top_ps: torch.Tensor = None
    top_ks: torch.Tensor = None
    frequency_penalties: torch.Tensor = None
    presence_penalties: torch.Tensor = None
    logit_bias: torch.Tensor = None

    # batched tree mask
    tree_mask: torch.Tensor = None
    tree_mask_start: torch.Tensor = None
    tree_mask_idx: torch.Tensor = None
    tree_depths: List[List[int]] = None

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

    def is_empty(self):
        return len(self.reqs) == 0

    def prepare_for_extend(self, vocab_size: int, int_token_logit_bias: torch.Tensor):
        # Set some alias
        device = "cuda"
        bs = len(self.reqs)

        # Batched arguments
        flatten_input_ids = []  # flatten all extend input ids
        extend_lens = []
        prefix_lens = []
        seq_lens = []
        req_pool_indices = self.req_to_token_pool.alloc(bs)
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()

        max_spec_tokens_len = max(
            len(r.speculate_tries.tree_ids) if r.speculate_tries is not None else 0
            for r in self.reqs
        )
        num_spec_reqs = sum(
            not r.speculate_tries.is_empty() if r.speculate_tries is not None else 0
            for r in self.reqs
        )

        tree_mask, tree_mask_start, tree_mask_idx, tree_depths = (
            None,
            None,
            None,
            None,
        )
        if num_spec_reqs > 0:
            tree_mask = np.zeros(
                (num_spec_reqs, max_spec_tokens_len, max_spec_tokens_len),
                dtype=np.int32,
            )
            tree_mask_start, tree_mask_idx, tree_depths = [], [], []

        # Handle prefix indices && flatten input ids
        for i, r in enumerate(self.reqs):
            prefix_len, seq_len = len(r.prefix_indices), len(r.input_ids)
            flatten_input_ids.extend(r.input_ids[prefix_len:])
            extend_lens.append(seq_len - prefix_len)
            prefix_lens.append(prefix_len)
            seq_lens.append(seq_len)

            # avoid GPU overhead
            if prefix_len != 0:
                self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
                    :prefix_len
                ] = r.prefix_indices

            # handle tree mask
            if r.speculate_tries is not None and not r.speculate_tries.is_empty():
                tree_ids_len = len(r.speculate_tries.tree_ids)
                cur_tree_mask_idx = len(tree_mask_idx)
                tree_mask[cur_tree_mask_idx, :tree_ids_len, :tree_ids_len] = (
                    r.speculate_tries.tree_mask
                )
                tree_mask_start.append(seq_len - prefix_len - tree_ids_len)
                tree_mask_idx.append(cur_tree_mask_idx)
                tree_depths.append(r.speculate_tries.tree_depths)

        # Alloc mem
        seq_lens, prefix_lens = np.array(seq_lens), np.array(prefix_lens)
        extend_num_tokens = seq_lens.sum() - prefix_lens.sum()
        out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)
        if out_cache_loc is None:
            if not self.tree_cache.disable:
                self.tree_cache.evict(extend_num_tokens, self.token_to_kv_pool.free)
                out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)

            if out_cache_loc is None:
                print("Prefill out of memory. This should nerver happen.")
                self.tree_cache.pretty_print()
                exit()

        # Map out_cache_loc to memory pool
        pt = 0
        for i in range(bs):
            self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
                prefix_lens[i] : prefix_lens[i] + extend_lens[i]
            ] = out_cache_loc[pt : pt + extend_lens[i]]
            pt += extend_lens[i]

        # Handle logit bias
        logit_bias = torch.zeros((bs, vocab_size), dtype=torch.float32, device=device)
        for i in range(bs):
            if self.reqs[i].sampling_params.dtype == "int":
                logit_bias[i] = int_token_logit_bias

        # Set fields
        self.input_ids = torch.tensor(
            flatten_input_ids, dtype=torch.int32, device=device
        )
        self.pixel_values = [r.pixel_values for r in self.reqs]
        self.image_sizes = [r.image_size for r in self.reqs]
        self.image_offsets = [
            r.image_offset - p_len for r, p_len in zip(self.reqs, prefix_lens)
        ]
        self.req_pool_indices = req_pool_indices
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        self.prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        self.position_ids_offsets = torch.zeros((bs,), dtype=torch.int32, device=device)
        self.extend_num_tokens = extend_num_tokens
        self.out_cache_loc = out_cache_loc

        self.temperatures = torch.tensor(
            [r.sampling_params.temperature for r in self.reqs],
            dtype=torch.float,
            device=device,
        ).view(-1, 1)
        self.top_ps = torch.tensor(
            [r.sampling_params.top_p for r in self.reqs],
            dtype=torch.float,
            device=device,
        ).view(-1, 1)
        self.top_ks = torch.tensor(
            [r.sampling_params.top_k for r in self.reqs], dtype=torch.int, device=device
        ).view(-1, 1)
        self.frequency_penalties = torch.tensor(
            [r.sampling_params.frequency_penalty for r in self.reqs],
            dtype=torch.float,
            device=device,
        )
        self.presence_penalties = torch.tensor(
            [r.sampling_params.presence_penalty for r in self.reqs],
            dtype=torch.float,
            device=device,
        )
        self.logit_bias = logit_bias

        self.tree_mask = (
            torch.tensor(tree_mask, dtype=torch.int32, device=device)
            if tree_mask is not None
            else None
        )
        self.tree_mask_start = (
            torch.tensor(tree_mask_start, dtype=torch.int32, device=device)
            if tree_mask_start is not None
            else None
        )
        self.tree_mask_idx = (
            torch.tensor(tree_mask_idx, dtype=torch.int32, device=device)
            if tree_mask_idx is not None
            else None
        )
        self.tree_depths = tree_depths

    def check_decode_mem(self):
        bs = len(self.reqs)
        if self.token_to_kv_pool.available_size() >= bs:
            return True

        if not self.tree_cache.disable:
            self.tree_cache.evict(bs, self.token_to_kv_pool.free)
        if self.token_to_kv_pool.available_size() >= bs:
            return True

        return False

    def retract_decode(self):
        sorted_indices = [i for i in range(len(self.reqs))]
        sorted_indices.sort(
            key=lambda i: (len(self.reqs[i].output_ids), -len(self.reqs[i].input_ids)),
            reverse=True,
        )

        retracted_reqs = []
        seq_lens_np = self.seq_lens.cpu().numpy()
        req_pool_indices_np = self.req_pool_indices.cpu().numpy()
        while self.token_to_kv_pool.available_size() < len(self.reqs):
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            self.tree_cache.dec_ref_counter(req.last_node)
            req.prefix_indices = None
            req.last_node = None
            req.extend_input_len = 0
            req.output_ids = []
            req.regex_fsm_state = 0

            # TODO: apply more fine-grained retraction

            token_indices = self.req_to_token_pool.req_to_token[
                req_pool_indices_np[idx]
            ][: seq_lens_np[idx]]
            self.token_to_kv_pool.free(token_indices)

        self.filter_batch(sorted_indices)

        return retracted_reqs

    def check_for_jump_forward(self, model_runner):
        jump_forward_reqs = []
        filter_indices = [i for i in range(len(self.reqs))]

        req_pool_indices_cpu = None

        for i, req in enumerate(self.reqs):
            if req.jump_forward_map is not None:
                res = req.jump_forward_map.jump_forward(req.regex_fsm_state)
                if res is not None:
                    jump_forward_str, next_state = res
                    if len(jump_forward_str) <= 1:
                        continue

                    # insert requests into tree_cache
                    token_ids_in_memory = tuple(req.input_ids + req.output_ids)[:-1]
                    if req_pool_indices_cpu is None:
                        req_pool_indices_cpu = self.req_pool_indices.cpu().tolist()
                    req_pool_idx = req_pool_indices_cpu[i]
                    indices = self.req_to_token_pool.req_to_token[
                        req_pool_idx, : len(token_ids_in_memory)
                    ]
                    prefix_len = self.tree_cache.insert(
                        token_ids_in_memory, indices.clone()
                    )
                    self.token_to_kv_pool.free(indices[:prefix_len])
                    self.req_to_token_pool.free(req_pool_idx)
                    self.tree_cache.dec_ref_counter(req.last_node)

                    # jump-forward
                    req.jump_forward_and_retokenize(jump_forward_str, next_state)

                    jump_forward_reqs.append(req)
                    filter_indices.remove(i)

        if len(filter_indices) < len(self.reqs):
            self.filter_batch(filter_indices)

        # re-applying image padding
        for req in jump_forward_reqs:
            if req.pixel_values is not None:
                (
                    req.input_ids,
                    req.image_offset,
                ) = model_runner.model.pad_input_ids(
                    req.input_ids,
                    req.pad_value,
                    req.pixel_values.shape,
                    req.image_size,
                )

        return jump_forward_reqs

    def check_for_speculative(self):
        spec_reqs = []
        filter_indices = [i for i in range(len(self.reqs))]

        req_pool_indices_cpu = None

        for i, req in enumerate(self.reqs):
            if req.speculate_engine is not None:
                prev_tokens = req.input_ids + req.output_ids
                req.speculate_tries = req.speculate_engine.search(prev_tokens)

                if not req.speculate_tries.is_empty():
                    # insert reqs into tree_cache
                    token_ids_in_memory = tuple(req.input_ids + req.output_ids)[:-1]
                    if req_pool_indices_cpu is None:
                        req_pool_indices_cpu = self.req_pool_indices.cpu().tolist()
                    req_pool_idx = req_pool_indices_cpu[i]
                    indices = self.req_to_token_pool.req_to_token[
                        req_pool_idx, : len(token_ids_in_memory)
                    ]
                    prefix_len = self.tree_cache.insert(
                        token_ids_in_memory, indices.clone()
                    )
                    self.token_to_kv_pool.free(indices[:prefix_len])
                    self.req_to_token_pool.free(req_pool_idx)
                    self.tree_cache.dec_ref_counter(req.last_node)

                    spec_reqs.append(req)
                    filter_indices.remove(i)

                    # append old output str to the output_and_jump_forward_str
                    old_output_ids = req.output_ids[:-1]
                    if len(old_output_ids) > 0:
                        old_output_str = req.tokenizer.decode(old_output_ids)
                        if req.tokenizer.convert_ids_to_tokens(
                            old_output_ids[0]
                        ).startswith("▁"):
                            old_output_str = " " + old_output_str
                        req.output_and_jump_forward_str = (
                            req.output_and_jump_forward_str + old_output_str
                        )

                    # debug print
                    # print("=" * 20, "Check For Speculative", "=" * 20)
                    # print(
                    #     f"Prev Tokens: {req.tokenizer.convert_ids_to_tokens(req.input_ids[-10:] + req.output_ids)}"
                    # )
                    # print(
                    #     f"Tree Tokens: #{len(req.speculate_tries.tree_ids)} {req.tokenizer.convert_ids_to_tokens(req.speculate_tries.tree_ids)}"
                    # )
                    # print("=" * 75)

                    # NOTE: spec tokens include the last output_ids by default
                    prev_new_tokens = len(req.output_ids) - 1
                    req.input_ids = (
                        req.input_ids
                        + req.output_ids[:-1]
                        + req.speculate_tries.tree_ids
                    )
                    req.output_ids = []
                    req.sampling_params.max_new_tokens = max(
                        req.sampling_params.max_new_tokens - prev_new_tokens, 0
                    )

        if len(filter_indices) < len(self.reqs):
            self.filter_batch(filter_indices)

        return spec_reqs

    def speculative_sample(self, all_logits, last_logits):
        all_probs = torch.softmax(all_logits, dim=-1)
        extend_starts = np.array([0] + [req.extend_input_len for req in self.reqs])
        extend_starts = np.cumsum(extend_starts)
        req_pool_indices_cpu = self.req_pool_indices.cpu().numpy()
        for i, req in enumerate(self.reqs):
            if req.speculate_tries is not None and not req.speculate_tries.is_empty():
                assert len(req.output_ids) == 0, "Ouput ids should be empty"
                cur_extend_start = extend_starts[i]
                cur_extend_len = req.extend_input_len
                cur_spec_len = len(req.speculate_tries.tree_ids)
                cur_spec_start_in_extend = cur_extend_len - cur_spec_len
                cur_spec_start_in_seq = len(req.input_ids) - cur_spec_len

                # cur_prefix_len = len(req.prefix_indices)

                # speculative sample in tries
                # NOTE: first id is always verified (last output)
                probs_pos = (
                    cur_extend_start
                    + cur_spec_start_in_extend
                    + np.array(req.speculate_tries.parent_indices[1:])
                )
                probs_ids = req.input_ids[-cur_spec_len + 1 :]
                spec_probs = all_probs[probs_pos, probs_ids].cpu().tolist()
                req.speculate_tries.fill_probs([1.0] + spec_probs)
                verified_indices, verified_ids = req.speculate_tries.sample_tries()

                # update the kv cache
                req_pool_idx = req_pool_indices_cpu[i]
                cur_req_to_token = self.req_to_token_pool.req_to_token[req_pool_idx]
                all_spec_kv_indices = cur_req_to_token[
                    cur_spec_start_in_seq : cur_spec_start_in_seq + cur_spec_len
                ]
                verified_kv_indices = cur_req_to_token[
                    cur_spec_start_in_seq + np.array(verified_indices)
                ]
                self.token_to_kv_pool.add_refs(verified_kv_indices)
                self.token_to_kv_pool.decrease_refs(all_spec_kv_indices)
                cur_req_to_token[
                    cur_spec_start_in_seq : cur_spec_start_in_seq
                    + len(verified_kv_indices)
                ] = verified_kv_indices

                # debug print
                # prefix_len = len(req.prefix_indices)
                # extend_len = len(req.input_ids) - prefix_len
                # spec_len = len(req.speculate_tries.tree_ids)
                # print("*" * 20, "Speculative Sample", "*" * 20)
                # print(
                #     f"prefix tokens: #{prefix_len} {req.tokenizer.convert_ids_to_tokens((req.input_ids)[max(0, prefix_len - 10) : prefix_len])}"
                # )
                # print(
                #     f"extend tokens: #{extend_len} {req.tokenizer.convert_ids_to_tokens((req.input_ids)[prefix_len:])}"
                # )
                # print(
                #     f"spec tokens: #{spec_len} {req.tokenizer.convert_ids_to_tokens(req.speculate_tries.tree_ids)}"
                # )
                # print(
                #     f"verified tokens: #{len(verified_ids)} {req.tokenizer.convert_ids_to_tokens(verified_ids)}"
                # )
                # for token, probs in zip(
                #     req.tokenizer.convert_ids_to_tokens(probs_ids), spec_probs
                # ):
                #     print(f"{token}: {probs:.4f}")
                # print("*" * 75)

                # update other meta data
                req.input_ids[-cur_spec_len:] = verified_ids
                spec_forward_str = req.tokenizer.decode(verified_ids)
                if req.tokenizer.convert_ids_to_tokens(verified_ids[0]).startswith("▁"):
                    spec_forward_str = " " + spec_forward_str
                req.output_and_jump_forward_str = (
                    req.output_and_jump_forward_str + spec_forward_str
                )
                last_logits[i] = all_logits[
                    cur_extend_start + cur_spec_start_in_extend + verified_indices[-1]
                ]
                self.seq_lens[i] -= len(all_spec_kv_indices) - len(verified_kv_indices)

    def prepare_for_decode(self, input_ids=None):
        if input_ids is None:
            input_ids = [
                r.output_ids[-1] if r.output_ids else r.input_ids[-1] for r in self.reqs
            ]
        self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        self.seq_lens.add_(1)
        self.prefix_lens = None

        # Alloc mem
        bs = len(self.reqs)
        alloc_res = self.token_to_kv_pool.alloc_contiguous(bs)
        if alloc_res is None:
            self.out_cache_loc = self.token_to_kv_pool.alloc(bs)

            if self.out_cache_loc is None:
                print("Decode out of memory. This should nerver happen.")
                self.tree_cache.pretty_print()
                exit()

            self.out_cache_cont_start = None
            self.out_cache_cont_end = None
        else:
            self.out_cache_loc = alloc_res[0]
            self.out_cache_cont_start = alloc_res[1]
            self.out_cache_cont_end = alloc_res[2]

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
        self.out_cache_loc = self.out_cache_cont_start = self.out_cache_cont_end = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "frequency_penalties",
            "presence_penalties",
            "logit_bias",
        ]:
            setattr(self, item, getattr(self, item)[new_indices])

    def merge(self, other):
        self.reqs.extend(other.reqs)

        self.req_pool_indices = torch.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.concat([self.seq_lens, other.seq_lens])
        self.prefix_lens = None
        self.position_ids_offsets = torch.concat(
            [self.position_ids_offsets, other.position_ids_offsets]
        )
        self.out_cache_loc = self.out_cache_cont_start = self.out_cache_cont_end = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "frequency_penalties",
            "presence_penalties",
            "logit_bias",
        ]:
            setattr(
                self, item, torch.concat([getattr(self, item), getattr(other, item)])
            )

    def sample(self, logits: torch.Tensor):
        # Post process logits
        logits = logits.contiguous()
        logits.div_(self.temperatures)
        logits.add_(self.logit_bias)

        has_regex = any(req.regex_fsm is not None for req in self.reqs)
        if has_regex:
            allowed_mask = torch.empty_like(logits[0], dtype=torch.bool)
            for i, req in enumerate(self.reqs):
                if req.regex_fsm is not None:
                    allowed_mask.zero_()
                    allowed_mask[
                        req.regex_fsm.allowed_token_ids(req.regex_fsm_state)
                    ] = 1
                    logits[i].masked_fill_(~allowed_mask, float("-inf"))

        # TODO(lmzheng): apply penalty
        probs = torch.softmax(logits, dim=-1)
        probs_sort, probs_idx = _top_p_top_k(probs, self.top_ps, self.top_ks)
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(
            -1
        )
        batch_next_token_probs = torch.gather(
            probs_sort, dim=1, index=sampled_index
        ).view(-1)

        if has_regex:
            batch_next_token_ids_cpu = batch_next_token_ids.cpu().numpy()
            for i, req in enumerate(self.reqs):
                if req.regex_fsm is not None:
                    req.regex_fsm_state = req.regex_fsm.next_state(
                        req.regex_fsm_state, batch_next_token_ids_cpu[i]
                    )

        return batch_next_token_ids, batch_next_token_probs


def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps] = 0.0
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1) >= top_ks
    ] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    return probs_sort, probs_idx
