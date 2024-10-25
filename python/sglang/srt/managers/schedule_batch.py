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

"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""

import dataclasses
import logging
from typing import List, Optional, Tuple, Union

import torch

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained import RegexGuide
from sglang.srt.constrained.jump_forward import JumpForwardMap
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# Put some global args for easy access
global_server_args_dict = {
    "attention_backend": ServerArgs.attention_backend,
    "sampling_backend": ServerArgs.sampling_backend,
    "triton_attention_reduce_in_fp32": ServerArgs.triton_attention_reduce_in_fp32,
    "disable_mla": ServerArgs.disable_mla,
    "torchao_config": ServerArgs.torchao_config,
    "disable_nan_detection": ServerArgs.disable_nan_detection,
}


logger = logging.getLogger(__name__)


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self):
        super().__init__(is_error=True)

    def to_json(self):
        return {
            "type": "abort",
        }


@dataclasses.dataclass
class ImageInputs:
    """The image related inputs."""

    pixel_values: torch.Tensor
    image_hashes: Optional[list] = None
    image_sizes: Optional[list] = None
    image_offsets: Optional[list] = None
    pad_values: Optional[list] = None
    modalities: Optional[list] = None
    num_image_tokens: Optional[int] = None

    image_embeds: Optional[List[torch.Tensor]] = None
    aspect_ratio_ids: Optional[List[torch.Tensor]] = None
    aspect_ratio_mask: Optional[List[torch.Tensor]] = None
    # QWen2-VL related
    image_grid_thws: List[Tuple[int, int, int]] = None

    @staticmethod
    def from_dict(obj, vocab_size):
        # Use image hash as fake token_ids, which is then used for prefix matching
        ret = ImageInputs(
            pixel_values=obj["pixel_values"],
            image_hashes=hash(tuple(obj["image_hashes"])),
        )
        image_hash = ret.image_hashes
        ret.pad_values = [
            (image_hash) % vocab_size,
            (image_hash >> 16) % vocab_size,
            (image_hash >> 32) % vocab_size,
            (image_hash >> 64) % vocab_size,
        ]

        optional_args = [
            "image_sizes",
            "modalities",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "image_grid_thws",
        ]
        for arg in optional_args:
            if arg in obj:
                setattr(ret, arg, obj[arg])

        return ret


class Req:
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: Tuple[int],
        sampling_params: SamplingParams,
        lora_path: Optional[str] = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = origin_input_ids  # Before image padding
        self.origin_input_ids = origin_input_ids
        self.output_ids = []  # Each decode stage's output ids
        self.fill_ids = None  # fill_ids = origin_input_ids + output_ids

        self.sampling_params = sampling_params
        self.lora_path = lora_path

        # Memory info
        self.req_pool_idx = None

        # Check finish
        self.tokenizer = None
        self.finished_reason = None
        self.stream = False

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

        # The number of cached tokens, that were already cached in the KV store
        self.cached_tokens = 0

        # For vision inputs
        self.image_inputs: Optional[ImageInputs] = None

        # Prefix info
        self.prefix_indices = []
        self.extend_input_len = 0
        self.last_node = None
        self.is_inflight_req = 0

        # Logprobs (arguments)
        self.return_logprob = False
        self.logprob_start_len = 0
        self.top_logprobs_num = 0

        # Logprobs (return value)
        self.normalized_prompt_logprob = None
        self.input_token_logprobs = None
        self.input_top_logprobs = None
        self.output_token_logprobs = []
        self.output_top_logprobs = []

        # Logprobs (internal values)
        # The tokens is prefilled but need to be considered as decode tokens
        # and should be updated for the decode logprobs
        self.last_update_decode_tokens = 0
        # The relative logprob_start_len in an extend batch
        self.extend_logprob_start_len = 0

        # Embedding
        self.embedding = None

        # Constrained decoding
        self.regex_fsm: RegexGuide = None
        self.regex_fsm_state: int = 0
        self.jump_forward_map: JumpForwardMap = None

        # For Qwen2-VL
        self.mrope_position_delta = []  # use mutable object

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

        matched_eos = False

        # Check stop token ids
        if self.sampling_params.stop_token_ids:
            matched_eos = last_token_id in self.sampling_params.stop_token_ids
        if self.tokenizer is not None:
            matched_eos |= last_token_id == self.tokenizer.eos_token_id
            if self.tokenizer.additional_stop_token_ids:
                matched_eos |= last_token_id in self.tokenizer.additional_stop_token_ids
        if matched_eos and not self.sampling_params.ignore_eos:
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
            return

        # Check stop strings
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


bid = 0


@dataclasses.dataclass
class ScheduleBatch:
    """Store all inforamtion of a batch."""

    # Request, memory pool, and cache
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    tree_cache: BasePrefixCache = None

    # For utility
    model_config: ModelConfig = None

    forward_mode: ForwardMode = None
    sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: torch.Tensor = None
    req_pool_indices: torch.Tensor = None
    seq_lens: torch.Tensor = None
    # The output locations of the KV cache
    out_cache_loc: torch.Tensor = None
    output_ids: torch.Tensor = None

    # The sum of all sequence lengths
    seq_lens_sum: int = None

    # For processing logprobs
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # For extend and mixed chunekd prefill
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_num_tokens: int = None
    decoding_reqs: List[Req] = None

    # For encoder-decoder
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # Stream
    has_stream: bool = False

    # Has regex
    has_regex: bool = False

    # device
    device: str = "cuda"

    @classmethod
    def init_new(
        cls,
        reqs,
        req_to_token_pool,
        token_to_kv_pool,
        tree_cache,
        model_config,
    ):
        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            tree_cache=tree_cache,
            model_config=model_config,
            return_logprob=any(req.return_logprob for req in reqs),
            has_stream=any(req.stream for req in reqs),
            has_regex=any(req.regex_fsm for req in reqs),
            device=req_to_token_pool.device,
        )

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

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
                phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
                logger.error(
                    f"{phase_str} out of memory. Try to lower your batch size.\n"
                    f"Try to allocate {num_tokens} tokens.\n"
                    f"Avaliable tokens: {self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()}\n"
                )
                if self.tree_cache is not None:
                    self.tree_cache.pretty_print()
                exit(1)

        return out_cache_loc

    def prepare_encoder_info_extend(self, input_ids: List[int], seq_lens: List[int]):
        self.encoder_lens_cpu = []
        self.encoder_cached = []

        for req in self.reqs:
            im = req.image_inputs
            if im is None or im.num_image_tokens is None:
                # No image input
                self.encoder_lens_cpu.append(0)
                self.encoder_cached.append(True)
            else:
                self.encoder_lens_cpu.append(im.num_image_tokens)
                self.encoder_cached.append(
                    self.forward_mode.is_decode()
                    or len(req.prefix_indices) >= im.num_image_tokens
                )

        self.encoder_lens = torch.tensor(self.encoder_lens_cpu, dtype=torch.int32).to(
            self.device, non_blocking=True
        )

        # Strip encoder infos
        pt = 0
        decoder_out_cache_loc = []
        encoder_out_cache_loc = []
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len

            if len(req.prefix_indices) < encoder_len:
                # NOTE: the encoder part should considered as a whole
                assert len(req.prefix_indices) == 0
                input_ids[i] = input_ids[i][encoder_len:]
                encoder_out_cache_loc.append(self.out_cache_loc[pt : pt + encoder_len])
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_input_len]
                )
                self.extend_lens[i] -= encoder_len
                self.extend_num_tokens -= encoder_len
            else:
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_input_len]
                )
                self.prefix_lens[i] -= encoder_len

            pt += req.extend_input_len

        # Reassign
        self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )

        if not decoder_out_cache_loc:
            self.out_cache_loc = torch.empty(0, dtype=torch.int32).to(
                self.device, non_blocking=True
            )
        else:
            self.out_cache_loc = torch.cat(decoder_out_cache_loc)

        if not encoder_out_cache_loc:
            self.encoder_out_cache_loc = torch.empty(0, dtype=torch.int32).to(
                self.device, non_blocking=True
            )
        else:
            self.encoder_out_cache_loc = torch.cat(encoder_out_cache_loc)

        assert len(self.out_cache_loc) == self.extend_num_tokens

    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        bs = len(self.reqs)
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []

        # Allocate memory
        req_pool_indices = self.alloc_req_slots(bs)
        out_cache_loc = self.alloc_token_slots(extend_num_tokens)

        pt = 0
        for i, req in enumerate(reqs):
            already_computed = (
                req.extend_logprob_start_len + 1 + req.cached_tokens
                if req.extend_logprob_start_len > 0
                else 0
            )
            req.cached_tokens += len(req.prefix_indices) - already_computed

            req.req_pool_idx = req_pool_indices[i]
            pre_len, seq_len = len(req.prefix_indices), len(req.fill_ids)
            seq_lens.append(seq_len)
            assert seq_len - pre_len == req.extend_input_len

            if pre_len > 0:
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
                )
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(pre_len, seq_len)),
                out_cache_loc[pt : pt + req.extend_input_len],
            )

            # Compute the relative logprob_start_len in an extend batch
            if req.logprob_start_len >= pre_len:
                extend_logprob_start_len = min(
                    req.logprob_start_len - pre_len, req.extend_input_len - 1
                )
            else:
                extend_logprob_start_len = req.extend_input_len - 1

            req.extend_logprob_start_len = extend_logprob_start_len
            pt += req.extend_input_len

        # Set fields
        self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )

        self.out_cache_loc = out_cache_loc

        self.seq_lens_sum = sum(seq_lens)
        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_input_len for r in reqs]
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
            global_server_args_dict["disable_penalizer"],
        )

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        for req in running_batch.reqs:
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.extend_input_len = 1

        input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc
        self.extend_num_tokens += running_bs

        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        self.prefix_lens.extend(
            [
                len(r.origin_input_ids) + len(r.output_ids) - 1
                for r in running_batch.reqs
            ]
        )
        self.extend_lens.extend([1] * running_bs)
        self.extend_logprob_start_lens.extend([0] * running_bs)

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
        first_iter = True
        while (
            self.token_to_kv_pool.available_size()
            < len(sorted_indices) * global_config.retract_decode_steps
            or first_iter
        ):
            if len(sorted_indices) == 1:
                # Corner case: only one request left
                assert (
                    self.token_to_kv_pool.available_size() > 0
                ), "No space left for only one request"
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            if isinstance(self.tree_cache, ChunkCache):
                # ChunkCache does not have eviction
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : seq_lens_cpu[idx]
                ]
                self.token_to_kv_pool.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)
                del self.tree_cache.entries[req.rid]
            else:
                # TODO: apply more fine-grained retraction
                last_uncached_pos = len(req.prefix_indices)
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, last_uncached_pos : seq_lens_cpu[idx]
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

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens + global_config.retract_decode_steps * len(self.reqs)
        ) / total_max_new_tokens
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio

    def check_for_jump_forward(self, pad_input_ids_func):
        jump_forward_reqs = []
        keep_indices = set(i for i in range(len(self.reqs)))

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
                    if req.image_inputs is not None:
                        req.origin_input_ids = pad_input_ids_func(
                            req.origin_input_ids_unpadded, req.image_inputs
                        )

                    jump_forward_reqs.append(req)
                    keep_indices.remove(i)

        self.filter_batch(keep_indices=list(keep_indices))

        return jump_forward_reqs

    def prepare_encoder_info_decode(self):
        # Reset the encoder cached status
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_decode(self, enable_overlap: bool = False):
        self.forward_mode = ForwardMode.DECODE

        self.input_ids = self.output_ids
        self.output_ids = None
        if self.sampling_info.penalizer_orchestrator:
            self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                self.input_ids
            )

        # Alloc mem
        bs = len(self.reqs)
        self.out_cache_loc = self.alloc_token_slots(bs)

        if self.model_config.is_encoder_decoder:
            locs = self.encoder_lens + self.seq_lens
            self.prepare_encoder_info_decode()
        else:
            locs = self.seq_lens

        if enable_overlap:
            # Do not use in-place operations in the overlap mode
            self.req_to_token_pool.write(
                (self.req_pool_indices, locs), self.out_cache_loc
            )
            self.seq_lens = self.seq_lens + 1
        else:
            # A faster in-place version
            self.req_to_token_pool.write(
                (self.req_pool_indices, locs), self.out_cache_loc
            )
            self.seq_lens.add_(1)
        self.seq_lens_sum += bs

    def filter_batch(
        self,
        current_inflight_req: Optional[Req] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] is not current_inflight_req
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        new_indices = torch.tensor(keep_indices, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.req_pool_indices = self.req_pool_indices[new_indices]
        self.seq_lens = self.seq_lens[new_indices]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.output_ids = self.output_ids[new_indices]
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None

        self.has_stream = any(req.stream for req in self.reqs)
        self.has_regex = any(req.regex_fsm for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, new_indices)

    def merge_batch(self, other: "ScheduleBatch"):
        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info)

        # Encoder-decoder infos
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu.extend(other.encoder_lens_cpu)

        self.req_pool_indices = torch.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.concat([self.seq_lens, other.seq_lens])
        self.out_cache_loc = None
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = torch.concat([self.output_ids, other.output_ids])
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
        elif self.return_logprob:
            self.top_logprobs_nums.extend([0] * len(other.reqs))
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
        self.reqs.extend(other.reqs)

        self.return_logprob = self.return_logprob or other.return_logprob
        self.has_stream = self.has_stream or other.has_stream
        self.has_regex = self.has_regex or other.has_regex

    def get_model_worker_batch(self):
        if self.forward_mode.is_decode():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            extend_logprob_start_lens = self.extend_logprob_start_lens

        if self.has_regex:
            self.sampling_info.regex_fsms = [req.regex_fsm for req in self.reqs]
            self.sampling_info.regex_fsm_states = [
                req.regex_fsm_state for req in self.reqs
            ]
        else:
            self.sampling_info.regex_fsms = None

        global bid
        bid += 1

        mrope_positions_delta = [req.mrope_position_delta for req in self.reqs]

        return ModelWorkerBatch(
            bid=bid,
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            out_cache_loc=self.out_cache_loc,
            seq_lens_sum=self.seq_lens_sum,
            req_to_token_pool_records=self.req_to_token_pool.get_write_records(),
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            image_inputs=[r.image_inputs for r in self.reqs],
            encoder_cached=self.encoder_cached,
            encoder_lens=self.encoder_lens,
            encoder_lens_cpu=self.encoder_lens_cpu,
            encoder_out_cache_loc=self.encoder_out_cache_loc,
            lora_paths=[req.lora_path for req in self.reqs],
            sampling_info=self.sampling_info,
            mrope_positions_delta=mrope_positions_delta,
        )

    def copy(self):
        # Only contain fields that will be used by process_batch_result
        return ScheduleBatch(
            reqs=self.reqs,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
        )

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name}, "
            f"#req={(len(self.reqs))})"
        )


@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # The memory pool operation records
    req_to_token_pool_records: Optional[List[Tuple[Tuple, torch.Tensor]]]

    # For logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]

    # For extend
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
    extend_logprob_start_lens: Optional[List[int]]

    # For multimodal
    image_inputs: Optional[List[ImageInputs]]

    # For encoder-decoder
    encoder_cached: Optional[List[bool]]
    encoder_lens: Optional[torch.Tensor]
    encoder_lens_cpu: Optional[List[int]]
    encoder_out_cache_loc: Optional[torch.Tensor]

    # For LoRA
    lora_paths: Optional[List[str]]

    # Sampling info
    sampling_info: SamplingBatchInfo

    # For Qwen2-VL
    mrope_positions_delta: List[List[int]]

    def copy(self):
        return dataclasses.replace(self, sampling_info=self.sampling_info.copy())

    def to(self, device: str):
        self.input_ids = self.input_ids.to(device, non_blocking=True)
        self.req_pool_indices = self.req_pool_indices.to(device, non_blocking=True)
        self.seq_lens = self.seq_lens.to(device, non_blocking=True)
        self.out_cache_loc = self.out_cache_loc.to(device, non_blocking=True)
        self.req_to_token_pool_records = [
            (x, y.to(device, non_blocking=True))
            for x, y in self.req_to_token_pool_records
        ]
        self.sampling_info.to(device)
