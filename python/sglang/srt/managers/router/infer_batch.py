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
    VERIFY_WITH_DECODE = auto()
    VERIFY_WITH_EXTEND = auto()


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
        # The number of decoded tokens for token usage report.
        # Note that this does not include any forward tokens.
        self.decode_tokens = 0
        # include fast-forward part (jump-forward, speculate)
        # and the non-empty output_ids when doing fast-forward
        self.gap_str = ""
        # For checking the stop str
        self.last_turn_decode_tokens = 0

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

        # For speculative decoding
        # TODO(lsyin): global SpeculateEngine to share ref materials
        self.speculate_engine: SpeculateEngine = None
        self.speculate_tries: SpeculateTries = None

    def spec_len(self):
        return (
            self.speculate_tries.spec_len() if self.speculate_tries is not None else 0
        )

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
            self.input_text + self.gap_str + old_output_str + jump_forward_str
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
        self.gap_str = self.gap_str + old_output_str + jump_forward_str

        # print(f"Gap Str:\n{self.gap_str}")
        # print("*" * 100)

    def check_finished(self):
        if self.finished:
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished = True
            self.finish_reason = FinishReason.LENGTH
            return

        # gap_str will never include <eos>
        if (
            self.output_ids[-1] == self.tokenizer.eos_token_id
            and self.sampling_params.ignore_eos == False
        ):
            self.finished = True
            self.finish_reason = FinishReason.EOS_TOKEN
            return

        if len(self.sampling_params.stop_strs) > 0:
            max_possible_len = (
                self.last_turn_decode_tokens + self.sampling_params.stop_str_max_len + 1
            )
            tail_ids = self.output_ids[-max_possible_len:]
            tail_str = self.tokenizer.decode(tail_ids)
            if self.tokenizer.convert_ids_to_tokens(tail_ids[0]).startswith("▁"):
                tail_str = " " + tail_str

            # When len(self.output_ids) == 1, the gap_str is just refreshed
            if len(self.output_ids) == 1:
                tail_str = self.gap_str + tail_str
            else:
                max_stop_str_len = max(len(s) for s in self.sampling_params.stop_strs)
                tail_str = self.gap_str[-max_stop_str_len + 1 :] + tail_str

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
    b_qo_lens: torch.Tensor = None
    tree_mask_flatten: torch.Tensor = None
    tree_mask_start_loc: torch.Tensor = None
    tree_mask_lens: torch.Tensor = None
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

        # init tree mask args
        tree_mask_flatten = np.zeros((0,), dtype=np.int32)
        tree_mask_start_loc, tree_mask_lens, tree_depths = [], [], []

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
            if r.spec_len() == 0:
                tree_mask_start_loc.append(tree_mask_flatten.shape[0])
                tree_mask_lens.append(0)
                tree_depths.append([])
            else:
                tree_mask_start_loc.append(tree_mask_flatten.shape[0])
                tree_mask_lens.append(r.spec_len())
                tree_mask_flatten = np.concatenate(
                    (tree_mask_flatten, r.speculate_tries.tree_mask.flatten())
                )
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

        # assign tree mask fields
        self.tree_mask_flatten = torch.tensor(
            tree_mask_flatten, dtype=torch.int32, device=device
        )
        self.tree_mask_start_loc = torch.tensor(
            tree_mask_start_loc, dtype=torch.int32, device=device
        )
        self.tree_mask_lens = torch.tensor(
            tree_mask_lens, dtype=torch.int32, device=device
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
        jump_forward_indices = []
        jump_forward_res = []
        filter_indices = []

        for i, req in enumerate(self.reqs):
            res = (
                req.jump_forward_map.jump_forward(req.regex_fsm_state)
                if req.jump_forward_map is not None
                else None
            )
            if res is not None and len(res[0]) > 1:
                jump_forward_indices.append(i)
                jump_forward_res.append(res)
            else:
                filter_indices.append(i)

        # insert requests into tree_cache
        self.cache_batch(jump_forward_indices)

        for idx, res in zip(jump_forward_indices, jump_forward_res):
            req = self.reqs[idx]
            req.jump_forward_and_retokenize(*res)

            # re-applying image padding
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

        jump_forward_reqs = [self.reqs[i] for i in jump_forward_indices]
        if len(filter_indices) < len(self.reqs):
            self.filter_batch(filter_indices)

        return jump_forward_reqs

    def check_for_speculative(self, filter_out: bool = False):
        spec_indices = []
        filter_indices = []

        for i, req in enumerate(self.reqs):
            if req.speculate_engine is not None:
                prev_ids = req.input_ids + req.output_ids
                req.speculate_engine.set_prev_ids(prev_ids)
                req.speculate_tries = req.speculate_engine.search(prev_ids)

                if not req.speculate_tries.is_empty():
                    spec_indices.append(i)
                    continue

            filter_indices.append(i)

        if not filter_out:
            return []

        # insert requests into tree_cache
        self.cache_batch(spec_indices)

        for idx in spec_indices:
            req = self.reqs[idx]
            # append old output str to the gap_str
            old_output_ids = req.output_ids[:-1]
            if len(old_output_ids) > 0:
                old_output_str = req.tokenizer.decode(old_output_ids)
                if req.tokenizer.convert_ids_to_tokens(old_output_ids[0]).startswith(
                    "▁"
                ):
                    old_output_str = " " + old_output_str
                req.gap_str = req.gap_str + old_output_str

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
                req.input_ids + req.output_ids[:-1] + req.speculate_tries.tree_ids
            )
            req.output_ids = []
            req.sampling_params.max_new_tokens = max(
                req.sampling_params.max_new_tokens - prev_new_tokens, 0
            )

        spec_reqs = [self.reqs[i] for i in spec_indices]
        if len(filter_indices) < len(self.reqs):
            self.filter_batch(filter_indices)

        return spec_reqs

    def speculative_sample_extend(self, all_logits, last_logits):
        extend_starts = np.array([0] + [req.extend_input_len for req in self.reqs])
        extend_starts = np.cumsum(extend_starts)
        req_pool_indices_cpu = self.req_pool_indices.cpu().numpy()
        for i, req in enumerate(self.reqs):
            if req.spec_len() > 0:
                assert len(req.output_ids) == 0, "Ouput ids should be empty"
                cur_extend_start = extend_starts[i]
                cur_extend_len = req.extend_input_len
                cur_spec_len = req.spec_len()
                cur_spec_start_in_extend = cur_extend_len - cur_spec_len
                cur_spec_start_in_seq = len(req.input_ids) - cur_spec_len

                # apply sampling params
                logits = all_logits[
                    cur_extend_start
                    + cur_spec_start_in_extend : cur_extend_start
                    + cur_extend_len
                ].clone()
                logits.div_(req.sampling_params.temperature)
                logits.add_(self.logit_bias[i])
                probs = torch.softmax(logits, dim=-1)
                top_ps = torch.full(
                    (probs.shape[0], 1),
                    req.sampling_params.top_p,
                    dtype=torch.float,
                    device="cuda",
                )
                top_ks = torch.full(
                    (probs.shape[0], 1),
                    req.sampling_params.top_k,
                    dtype=torch.int32,
                    device="cuda",
                )
                probs_sort, probs_idx = _top_p_top_k(probs, top_ps, top_ks)
                probs = probs_sort.gather(1, probs_idx.argsort(1))

                # speculative sample in tries
                # NOTE: first id is always verified (last turn output)
                parent_indices = np.array(req.speculate_tries.parent_indices[1:])
                ids_indices = req.input_ids[-cur_spec_len + 1 :]
                probs_sums = probs.sum(dim=-1).cpu().tolist()
                spec_probs = [probs_sums[0]] + probs[
                    parent_indices, ids_indices
                ].cpu().tolist()
                req.speculate_tries.fill_probs(zip(spec_probs, probs_sums))
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
                #     req.tokenizer.convert_ids_to_tokens(verified_ids), spec_probs
                # ):
                #     print(f"{token}: {probs:.4f}")
                # print("*" * 75)

                # update other meta data
                req.input_ids[-cur_spec_len:] = verified_ids
                spec_forward_str = req.tokenizer.decode(verified_ids)
                if req.tokenizer.convert_ids_to_tokens(verified_ids[0]).startswith("▁"):
                    spec_forward_str = " " + spec_forward_str
                req.gap_str = req.gap_str + spec_forward_str
                last_logits[i] = all_logits[
                    cur_extend_start + cur_spec_start_in_extend + verified_indices[-1]
                ]
                self.seq_lens[i] -= len(all_spec_kv_indices) - len(verified_kv_indices)
                req.sampling_params.max_new_tokens = max(
                    req.sampling_params.max_new_tokens - len(verified_ids), 0
                )
                # print(f"Speculative Sample: \x1b[31m{spec_forward_str}\x1b[0m")

    def speculative_sample_decode(self, all_logits, last_logits):
        qo_lens_cpu = self.b_qo_lens.cpu().numpy()
        qo_lens_start_cpu = np.cumsum(np.concatenate(([0], qo_lens_cpu[:-1])))
        req_pool_indices_cpu = self.req_pool_indices.cpu().numpy()

        for i, req in enumerate(self.reqs):
            if req.spec_len() > 0:
                cur_qo_start = qo_lens_start_cpu[i]
                cur_qo_len = qo_lens_cpu[i]
                cur_spec_len = req.spec_len()
                cur_seq_len = len(req.input_ids) + len(req.output_ids)
                assert cur_spec_len == cur_qo_len

                logits = all_logits[cur_qo_start : cur_qo_start + cur_qo_len].clone()
                logits.div_(req.sampling_params.temperature)
                logits.add_(self.logit_bias[i])
                probs = torch.softmax(logits, dim=-1)
                top_ps = torch.full(
                    (probs.shape[0], 1),
                    req.sampling_params.top_p,
                    dtype=torch.float,
                    device="cuda",
                )
                top_ks = torch.full(
                    (probs.shape[0], 1),
                    req.sampling_params.top_k,
                    dtype=torch.int32,
                    device="cuda",
                )
                probs_sort, probs_idx = _top_p_top_k(probs, top_ps, top_ks)
                probs = probs_sort.gather(1, probs_idx.argsort(1))

                # speculative sample in tries
                # NOTE: first id is always verified (last turn output)
                parent_indices = np.array(req.speculate_tries.parent_indices[1:])
                ids_indices = req.speculate_tries.tree_ids[1:]
                probs_sums = probs.sum(dim=-1).cpu().tolist()
                spec_probs = [probs_sums[0]] + probs[
                    parent_indices, ids_indices
                ].cpu().tolist()
                req.speculate_tries.fill_probs(zip(spec_probs, probs_sums))
                verified_indices, verified_ids = req.speculate_tries.sample_tries()

                # debug print
                # all_ids = req.input_ids + req.output_ids
                # prev_ids = all_ids[:-cur_spec_len]
                # print("*" * 20, "Speculative Sample", "*" * 20)
                # print(
                #     f"previous tokens: {req.tokenizer.convert_ids_to_tokens(prev_ids[-10:])}"
                # )
                # print(
                #     f"spec tokens: {req.tokenizer.convert_ids_to_tokens(req.speculate_tries.tree_ids)}"
                # )
                # print(
                #     f"verified tokens: {req.tokenizer.convert_ids_to_tokens(verified_ids)}"
                # )
                # for token, probs in zip(
                #     req.tokenizer.convert_ids_to_tokens(req.speculate_tries.tree_ids),
                #     spec_probs,
                # ):
                #     print(f"{token}: {probs:.4f}")
                # print("*" * 75)

                # update the kv cache
                req_pool_idx = req_pool_indices_cpu[i]
                cur_req_to_token = self.req_to_token_pool.req_to_token[req_pool_idx]
                cur_spec_start_in_seq = cur_seq_len - cur_spec_len
                all_spec_kv_indices = cur_req_to_token[
                    cur_spec_start_in_seq:cur_seq_len
                ]
                verified_kv_indice = cur_req_to_token[
                    cur_spec_start_in_seq + np.array(verified_indices)
                ]
                self.token_to_kv_pool.add_refs(verified_kv_indice)
                self.token_to_kv_pool.decrease_refs(all_spec_kv_indices)
                cur_req_to_token[
                    cur_spec_start_in_seq : cur_spec_start_in_seq
                    + len(verified_kv_indice)
                ] = verified_kv_indice

                # update other meta data
                req.output_ids[-cur_spec_len:] = verified_ids
                last_logits[i] = all_logits[cur_qo_start + verified_indices[-1]]
                self.seq_lens[i] -= len(all_spec_kv_indices) - len(verified_kv_indice)
                req.last_turn_decode_tokens = len(verified_ids) - 1
                # print(
                #     f"Speculative Sample: \x1b[31m{req.tokenizer.decode(verified_ids)}\x1b[0m"
                # )

    def prepare_for_verify_with_decode(self):
        tree_mask_flatten = np.zeros((0,), dtype=np.int32)
        tree_mask_start_loc, tree_mask_lens, tree_mask_depths = [], [], []
        input_ids, b_qo_lens = [], []

        for r in self.reqs:
            if r.spec_len() > 0:
                tree_mask_start_loc.append(tree_mask_flatten.shape[0])
                tree_mask_lens.append(r.spec_len())
                tree_mask_depths.append(r.speculate_tries.tree_depths)
                tree_mask_flatten = np.concatenate(
                    (tree_mask_flatten, r.speculate_tries.tree_mask.flatten())
                )

                input_ids.extend(r.speculate_tries.tree_ids)
                b_qo_lens.append(r.spec_len())
                r.output_ids.extend(r.speculate_tries.tree_ids[1:])
            else:
                tree_mask_start_loc.append(tree_mask_flatten.shape[0])
                tree_mask_lens.append(0)
                tree_mask_depths.append([])
                input_ids.append(r.output_ids[-1])
                b_qo_lens.append(1)

        qo_num_tokens = sum(b_qo_lens)
        self.out_cache_loc = self.token_to_kv_pool.alloc(qo_num_tokens)
        self.out_cache_cont_start = None
        self.out_cache_cont_end = None

        if self.out_cache_loc is None:
            if not self.tree_cache.disable:
                self.tree_cache.evict(qo_num_tokens, self.token_to_kv_pool.free)
                self.out_cache_loc = self.token_to_kv_pool.alloc(qo_num_tokens)

            if self.out_cache_loc is None:
                print("Decode verification out of memory. This should nerver happen.")
                self.tree_cache.pretty_print()
                exit()

        pt = 0
        req_pool_indices_cpu = self.req_pool_indices.cpu().numpy()
        seq_lens_cpu = self.seq_lens.cpu().numpy()
        for i in range(len(self.reqs)):
            self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
                seq_lens_cpu[i] : seq_lens_cpu[i] + b_qo_lens[i]
            ] = self.out_cache_loc[pt : pt + b_qo_lens[i]]
            pt += b_qo_lens[i]

        self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        self.b_qo_lens = torch.tensor(b_qo_lens, dtype=torch.int32, device="cuda")
        self.seq_lens = self.seq_lens + self.b_qo_lens
        self.tree_mask_flatten = torch.tensor(
            tree_mask_flatten, dtype=torch.int32, device="cuda"
        )
        self.tree_mask_start_loc = torch.tensor(
            tree_mask_start_loc, dtype=torch.int32, device="cuda"
        )
        self.tree_mask_lens = torch.tensor(
            tree_mask_lens, dtype=torch.int32, device="cuda"
        )
        self.tree_depths = tree_mask_depths

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

    def cache_batch(self, cache_indices: List[int]):
        if cache_indices is None or len(cache_indices) == 0:
            return

        # insert requests into tree_cache
        req_pool_indices_cpu = self.req_pool_indices.cpu().tolist()
        for idx in cache_indices:
            req = self.reqs[idx]
            token_ids_in_memory = tuple(req.input_ids + req.output_ids)[:-1]
            req_pool_idx = req_pool_indices_cpu[idx]
            indices = self.req_to_token_pool.req_to_token[
                req_pool_idx, : len(token_ids_in_memory)
            ]
            prefix_len = self.tree_cache.insert(token_ids_in_memory, indices.clone())
            self.token_to_kv_pool.free(indices[:prefix_len])
            self.req_to_token_pool.free(req_pool_idx)
            self.tree_cache.dec_ref_counter(req.last_node)

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
