from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSummableTensorPairFn,
    ScatterMode,
)
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_tbo_token_distribution_threshold,
    is_tbo_enabled,
)
from sglang.srt.layers.moe.token_dispatcher import (
    DeepEPDispatcher,
    MooncakeEPDispatcher,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.operations import execute_operations, execute_overlapped_operations
from sglang.srt.operations_strategy import OperationsStrategy
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import BumpAllocator, empty_context, get_bool_env_var, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutput
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

_is_hip = is_hip()

_tbo_debug = get_bool_env_var("SGLANG_TBO_DEBUG")

logger = logging.getLogger(__name__)


# -------------------------------- Compute Basic Info ---------------------------------------


def get_token_num_per_seq(
    forward_mode: ForwardMode,
    spec_info: Optional[SpecInput] = None,
):
    if forward_mode.is_target_verify():
        return spec_info.draft_token_num
    elif forward_mode.is_decode():
        return 1
    elif forward_mode.is_idle():
        return 0
    else:
        # For extend, we should not use `token_num_per_seq`.
        return None


# TODO: may smartly disable TBO when batch size is too small b/c it will slow down
def compute_split_seq_index(
    forward_mode: ForwardMode,
    num_tokens: int,
    extend_lens: Optional[Sequence[int]],
    token_num_per_seq: Optional[int],
) -> Optional[int]:
    if forward_mode == ForwardMode.EXTEND:
        assert extend_lens is not None
        return _split_extend_seqs(extend_lens)
    elif forward_mode.is_target_verify() or forward_mode.is_decode():
        assert token_num_per_seq is not None
        return (num_tokens // token_num_per_seq) // 2
    elif forward_mode.is_idle() or forward_mode.is_prebuilt_extend():
        assert num_tokens == 0
        return 0
    else:
        raise NotImplementedError()


def _is_two_chunk_split_enabled(extend_lens: Sequence[int]) -> bool:
    if extend_lens is None:
        return False

    vanilla_split_seq_index = _split_array_by_balanced_sum(extend_lens)
    left_sum = sum(extend_lens[:vanilla_split_seq_index])
    overall_sum = sum(extend_lens)
    threshold = get_tbo_token_distribution_threshold()
    assert threshold <= 0.5, f"{threshold=}"
    return left_sum < overall_sum * threshold or left_sum > overall_sum * (
        1 - threshold
    )


def _split_extend_seqs(arr: Sequence[int]) -> int:
    if _is_two_chunk_split_enabled(arr):
        return _split_array_by_cum_less_than_half(arr)

    return _split_array_by_balanced_sum(arr)


def _split_array_by_cum_less_than_half(arr: Sequence[int]) -> int:
    left_sum = 0
    overall_sum = sum(arr)
    half_sum = overall_sum // 2
    chosen_index = 0

    for i in range(len(arr)):
        left_sum += arr[i]
        if left_sum > half_sum:
            chosen_index = i
            break

    return chosen_index


def _split_array_by_balanced_sum(arr: Sequence[int]) -> int:
    overall_sum = sum(arr)
    left_sum = 0
    min_diff = float("inf")
    best_index = 0

    for i in range(1, len(arr)):
        left_sum += arr[i - 1]
        right_sum = overall_sum - left_sum
        diff = abs(left_sum - right_sum)
        if diff <= min_diff:
            min_diff = diff
            best_index = i
        else:
            break

    return best_index


def _update_device_and_sum_field_from_cpu_field(
    batch: ForwardBatch, cpu_field: str, device_field: str, sum_field: str = None
):
    cpu_value = getattr(batch, cpu_field, None)
    old_device_value = getattr(batch, device_field, None)
    if (
        cpu_value is None
        or old_device_value is None
        or not (isinstance(cpu_value, torch.Tensor) or isinstance(cpu_value, list))
    ):
        return

    new_device_value = (
        cpu_value
        if isinstance(cpu_value, torch.Tensor)
        else torch.tensor(cpu_value, dtype=old_device_value.dtype)
    ).to(device=get_global_server_args().device, non_blocking=True)
    setattr(batch, device_field, new_device_value)

    if sum_field is not None:
        sum_value = (
            cpu_value.sum().item()
            if isinstance(cpu_value, torch.Tensor)
            else sum(cpu_value)
        )
        setattr(batch, sum_field, sum_value)


def _compute_mask_offset(seq_index: int, spec_info: Optional[EagleVerifyInput]) -> int:
    if seq_index == 0:
        return 0

    offset = 0
    max_seq_len = min(seq_index, spec_info.seq_lens_cpu.shape[0])
    for i in range(max_seq_len):
        offset += (
            spec_info.seq_lens_cpu[i] + spec_info.draft_token_num
        ) * spec_info.draft_token_num
    return offset


def split_spec_info(
    spec_info: Optional[EagleVerifyInput],
    start_seq_index: int,
    end_seq_index: int,
    start_token_index: int,
    end_token_index: int,
):
    if spec_info is None:
        return None
    if spec_info.draft_token is not None:
        draft_token = spec_info.draft_token[start_token_index:end_token_index]
    else:
        draft_token = None
    if spec_info.custom_mask is not None and spec_info.draft_token is not None:
        custom_mask_start = _compute_mask_offset(start_seq_index, spec_info)
        if end_seq_index == spec_info.seq_lens_cpu.shape[0]:
            custom_mask_end = spec_info.custom_mask.shape[0]
        else:
            custom_mask_end = _compute_mask_offset(end_seq_index, spec_info)

        if custom_mask_end > custom_mask_start:
            custom_mask = spec_info.custom_mask[custom_mask_start:custom_mask_end]
        else:
            custom_mask = spec_info.custom_mask
    else:
        custom_mask = spec_info.custom_mask
    if spec_info.positions is not None:
        positions = spec_info.positions[start_token_index:end_token_index]
    else:
        positions = None
    if spec_info.retrive_index is not None:
        retrive_index = spec_info.retrive_index[start_seq_index:end_seq_index]
    else:
        retrive_index = None
    if spec_info.retrive_next_token is not None:
        retrive_next_token = spec_info.retrive_next_token[start_seq_index:end_seq_index]
    else:
        retrive_next_token = None
    if spec_info.retrive_next_sibling is not None:
        retrive_next_sibling = spec_info.retrive_next_sibling[
            start_seq_index:end_seq_index
        ]
    else:
        retrive_next_sibling = None
    if spec_info.retrive_cum_len is not None:
        retrive_cum_len = spec_info.retrive_cum_len[start_seq_index:end_seq_index]
    else:
        retrive_cum_len = None

    if spec_info.seq_lens_cpu is not None:
        seq_lens_cpu = spec_info.seq_lens_cpu[start_seq_index:end_seq_index]
    else:
        seq_lens_cpu = None
    if seq_lens_cpu is not None:
        seq_lens_sum = seq_lens_cpu.sum()
    else:
        seq_lens_sum = None
    output_spec_info = replace(
        spec_info,
        custom_mask=custom_mask,
        draft_token=draft_token,
        positions=positions,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        retrive_cum_len=retrive_cum_len,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_sum=seq_lens_sum,
    )
    return output_spec_info


def compute_split_token_index(
    split_seq_index: int,
    forward_mode: "ForwardMode",
    extend_seq_lens: Optional[Sequence[int]],
    token_num_per_seq: Optional[int],
) -> int:
    if forward_mode == ForwardMode.EXTEND:
        assert extend_seq_lens is not None
        if _is_two_chunk_split_enabled(extend_seq_lens):
            return sum(extend_seq_lens) // 2
        return sum(extend_seq_lens[:split_seq_index])
    elif forward_mode.is_target_verify() or forward_mode.is_decode():
        assert token_num_per_seq is not None
        return split_seq_index * token_num_per_seq
    elif forward_mode.is_idle():
        assert split_seq_index == 0
        return 0
    else:
        raise NotImplementedError


def compute_split_indices_for_cuda_graph_replay(
    forward_mode: ForwardMode,
    cuda_graph_num_tokens: int,
    spec_info: Optional[SpecInput],
):
    forward_mode_for_tbo_split = (
        forward_mode if forward_mode != ForwardMode.IDLE else ForwardMode.DECODE
    )
    token_num_per_seq = get_token_num_per_seq(
        forward_mode=forward_mode, spec_info=spec_info
    )
    tbo_split_seq_index = compute_split_seq_index(
        forward_mode=forward_mode_for_tbo_split,
        num_tokens=cuda_graph_num_tokens,
        extend_lens=None,
        token_num_per_seq=token_num_per_seq,
    )
    tbo_split_token_index = compute_split_token_index(
        split_seq_index=tbo_split_seq_index,
        forward_mode=forward_mode_for_tbo_split,
        extend_seq_lens=None,
        token_num_per_seq=token_num_per_seq,
    )
    return tbo_split_seq_index, tbo_split_token_index


# -------------------------------- Preparation ---------------------------------------


class TboCudaGraphRunnerPlugin:
    def __init__(self):
        self._tbo_children_num_token_non_padded = torch.zeros((2,), dtype=torch.int32)

    def capture_one_batch_size(self, batch: ForwardBatch, num_tokens: int):
        if not is_tbo_enabled():
            return
        token_num_per_seq = get_token_num_per_seq(
            forward_mode=batch.forward_mode, spec_info=batch.spec_info
        )

        batch.tbo_split_seq_index = compute_split_seq_index(
            forward_mode=batch.forward_mode,
            num_tokens=num_tokens,
            extend_lens=None,
            token_num_per_seq=token_num_per_seq,
        )
        # For simplicity, when two_batch_overlap is enabled, we only capture CUDA Graph for tbo=true
        assert batch.tbo_split_seq_index is not None, f"{num_tokens=}"

        self._tbo_children_num_token_non_padded[...] = (
            TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded(batch)
        )

        TboForwardBatchPreparer.prepare_raw(
            batch,
            tbo_children_num_token_non_padded=self._tbo_children_num_token_non_padded,
        )

    def replay_prepare(
        self,
        forward_mode: ForwardMode,
        bs: int,
        num_token_non_padded: int,
        spec_info: Optional[SpecInput],
    ):
        token_num_per_seq = get_token_num_per_seq(
            forward_mode=forward_mode, spec_info=spec_info
        )
        tbo_split_seq_index, tbo_split_token_index = (
            compute_split_indices_for_cuda_graph_replay(
                forward_mode=forward_mode,
                cuda_graph_num_tokens=bs * token_num_per_seq,
                spec_info=spec_info,
            )
        )

        self._tbo_children_num_token_non_padded[...] = (
            TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded_raw(
                tbo_split_token_index=tbo_split_token_index,
                num_token_non_padded=num_token_non_padded,
            )
        )


class TboDPAttentionPreparer:
    def prepare_all_gather(
        self,
        local_batch: ScheduleBatch,
    ):

        deepep_mode = get_deepep_mode()
        enable_a2a_moe = not get_moe_a2a_backend().is_none()
        enable_two_batch_overlap = is_tbo_enabled()

        self.enable_two_batch_overlap = enable_two_batch_overlap

        if local_batch is not None:
            token_num_per_seq = get_token_num_per_seq(
                forward_mode=local_batch.forward_mode, spec_info=local_batch.spec_info
            )

            if (
                local_batch.forward_mode.is_target_verify()
                or local_batch.forward_mode.is_decode()
            ):
                num_tokens = local_batch.batch_size() * token_num_per_seq
            elif local_batch.forward_mode.is_prebuilt_extend():
                num_tokens = 0
            else:
                num_tokens = local_batch.extend_num_tokens
            self.local_tbo_split_seq_index = compute_split_seq_index(
                forward_mode=local_batch.forward_mode,
                num_tokens=num_tokens,
                extend_lens=local_batch.extend_lens,
                token_num_per_seq=token_num_per_seq,
            )
            resolved_deepep_mode = deepep_mode.resolve(local_batch.is_extend_in_batch)
            local_can_run_tbo = (self.local_tbo_split_seq_index is not None) and not (
                (
                    local_batch.forward_mode.is_extend()
                    and not local_batch.forward_mode.is_target_verify()
                )
                and enable_a2a_moe
                and (resolved_deepep_mode.is_low_latency())
            )
        else:
            self.local_tbo_split_seq_index = 0
            local_can_run_tbo = True

        local_forward_mode = self._compute_local_forward_mode(local_batch)

        return local_can_run_tbo, local_forward_mode

    def compute_output(self, partial_global_info):
        local_can_run_tbo_aggregated = min(partial_global_info[:, 0].tolist())
        forward_modes = partial_global_info[:, 1].tolist()

        global_forward_mode, forward_mode_agree = self._compute_global_forward_mode(
            forward_modes
        )

        can_run_tbo = (
            self.enable_two_batch_overlap
            and local_can_run_tbo_aggregated
            and forward_mode_agree
        )

        tbo_split_seq_index = self.local_tbo_split_seq_index if can_run_tbo else None
        global_forward_mode = global_forward_mode if can_run_tbo else None
        return tbo_split_seq_index, global_forward_mode

    @staticmethod
    def _compute_local_forward_mode(local_batch):
        return (
            local_batch.forward_mode if local_batch is not None else ForwardMode.IDLE
        ).value

    @staticmethod
    def _compute_global_forward_mode(forward_modes):
        forward_modes_excluding_idle = [
            x for x in forward_modes if x != ForwardMode.IDLE.value
        ]

        if not forward_modes_excluding_idle:
            return ForwardMode.IDLE, False

        forward_mode_agree = TboDPAttentionPreparer._is_all_same(
            forward_modes_excluding_idle
        )
        global_forward_mode = (
            ForwardMode(forward_modes_excluding_idle[0]) if forward_mode_agree else None
        )
        return global_forward_mode, forward_mode_agree

    @staticmethod
    def _is_all_same(x):
        return all(value == x[0] for value in x)


class TboForwardBatchPreparer:
    @classmethod
    def prepare(cls, batch: ForwardBatch, is_draft_worker: bool = False):
        if batch.tbo_split_seq_index is None or is_draft_worker:
            return

        tbo_children_num_token_non_padded = (
            cls.compute_tbo_children_num_token_non_padded(batch)
        )
        cls.prepare_raw(
            batch, tbo_children_num_token_non_padded=tbo_children_num_token_non_padded
        )

    @classmethod
    def prepare_raw(
        cls, batch: ForwardBatch, tbo_children_num_token_non_padded: torch.Tensor
    ):
        from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

        tbo_split_token_index = cls._compute_split_token_index(batch)

        is_enable_two_chunk = (
            batch.forward_mode == ForwardMode.EXTEND
            and _is_two_chunk_split_enabled(batch.extend_seq_lens_cpu)
        )

        if _tbo_debug:
            logger.info(
                f"TboForwardBatchPreparer.prepare "
                f"is_enable_two_chunk={is_enable_two_chunk} "
                f"tbo_split_seq_index={batch.tbo_split_seq_index} "
                f"tbo_split_token_index={tbo_split_token_index} "
                f"extend_seq_lens={batch.extend_seq_lens_cpu} "
                f"bs={batch.batch_size} "
                f"forward_mode={batch.forward_mode}"
            )

        assert isinstance(batch.attn_backend, TboAttnBackend)
        attn_backend_child_a, attn_backend_child_b = batch.attn_backend.children

        [out_num_token_non_padded_a, out_num_token_non_padded_b] = (
            tbo_children_num_token_non_padded
        )

        child_a = cls.filter_batch(
            batch,
            start_token_index=0,
            end_token_index=tbo_split_token_index,
            start_seq_index=0,
            end_seq_index=(
                batch.tbo_split_seq_index + 1
                if is_enable_two_chunk
                else batch.tbo_split_seq_index
            ),
            output_attn_backend=attn_backend_child_a,
            out_num_token_non_padded=out_num_token_non_padded_a,
        )
        child_b = cls.filter_batch(
            batch,
            start_token_index=tbo_split_token_index,
            end_token_index=batch.input_ids.shape[0],
            start_seq_index=batch.tbo_split_seq_index,
            end_seq_index=batch.batch_size,
            output_attn_backend=attn_backend_child_b,
            out_num_token_non_padded=out_num_token_non_padded_b,
        )

        if is_enable_two_chunk:
            cls.derive_fields_related_to_seq_len_for_two_chunk(
                batch,
                child_a=child_a,
                child_b=child_b,
                tbo_split_seq_index=batch.tbo_split_seq_index,
            )

        assert batch.tbo_children is None
        batch.tbo_children = [child_a, child_b]

    @classmethod
    def derive_fields_related_to_seq_len_for_two_chunk(
        cls,
        batch: ForwardBatch,
        *,
        child_a: ForwardBatch,
        child_b: ForwardBatch,
        tbo_split_seq_index: int,
    ):
        extend_seq_lens_cpu = batch.extend_seq_lens_cpu
        overall_seq_lens_sum = sum(extend_seq_lens_cpu)
        half_seq_lens_sum = overall_seq_lens_sum // 2
        left_last_seq_token_num = half_seq_lens_sum - sum(
            extend_seq_lens_cpu[:tbo_split_seq_index]
        )
        right_first_seq_token_num = (
            extend_seq_lens_cpu[tbo_split_seq_index] - left_last_seq_token_num
        )

        # making deepcopy to be extra safe
        child_a.extend_seq_lens_cpu = copy.deepcopy(child_a.extend_seq_lens_cpu)
        child_a.extend_seq_lens_cpu[-1] = left_last_seq_token_num
        child_b.extend_seq_lens_cpu = copy.deepcopy(child_b.extend_seq_lens_cpu)
        child_b.extend_seq_lens_cpu[0] = right_first_seq_token_num
        for child in [child_a, child_b]:
            _update_device_and_sum_field_from_cpu_field(
                batch=child,
                cpu_field="extend_seq_lens_cpu",
                device_field="extend_seq_lens",
                sum_field="extend_num_tokens",
            )

        assert (
            child_a.extend_num_tokens == half_seq_lens_sum
        ), f"{child_a.extend_num_tokens=}, {half_seq_lens_sum=}"

        child_a.seq_lens_cpu = copy.deepcopy(child_a.seq_lens_cpu)
        child_a.seq_lens_cpu[-1] = (
            child_a.extend_seq_lens_cpu[-1] + child_a.extend_prefix_lens_cpu[-1]
        )
        _update_device_and_sum_field_from_cpu_field(
            batch=child_a,
            cpu_field="seq_lens_cpu",
            device_field="seq_lens",
            sum_field="seq_lens_sum",
        )

        child_b.extend_prefix_lens_cpu = copy.deepcopy(child_b.extend_prefix_lens_cpu)
        child_b.extend_prefix_lens_cpu[0] += left_last_seq_token_num
        _update_device_and_sum_field_from_cpu_field(
            batch=child_b,
            cpu_field="extend_prefix_lens_cpu",
            device_field="extend_prefix_lens",
            sum_field=None,
        )
        _, child_b.extend_start_loc = compute_position(
            get_global_server_args().attention_backend,
            child_b.extend_prefix_lens,
            child_b.extend_seq_lens,
            child_b.extend_num_tokens,
        )

    @classmethod
    def filter_batch(
        cls,
        batch: ForwardBatch,
        *,
        start_token_index: int,
        end_token_index: int,
        start_seq_index: int,
        end_seq_index: int,
        output_attn_backend: AttentionBackend,
        out_num_token_non_padded: torch.Tensor,
    ):
        assert (
            end_token_index >= start_token_index
        ), f"{end_token_index=}, {start_token_index=}, batch={batch}"
        num_tokens = batch.input_ids.shape[0]
        num_seqs = batch.batch_size

        output_dict = dict()

        for key in [
            "input_ids",
            "positions",
            "out_cache_loc",
        ]:
            old_value = getattr(batch, key)
            assert (
                old_value.shape[0] == num_tokens
            ), f"{key=} {old_value=} {num_tokens=} {batch=}"
            output_dict[key] = old_value[start_token_index:end_token_index]

        for key in [
            "req_pool_indices",
            "seq_lens",
            "seq_lens_cpu",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
            "extend_prefix_lens_cpu",
            "extend_seq_lens_cpu",
            "extend_logprob_start_lens_cpu",
            "lora_ids",
        ]:
            old_value = getattr(batch, key)
            if old_value is None:
                continue
            elif batch.forward_mode.is_target_verify() and (
                key == "extend_seq_lens"
                or key == "extend_prefix_lens"
                or key == "extend_start_loc"
                or key == "extend_prefix_lens_cpu"
                or key == "extend_seq_lens_cpu"
                or key == "extend_logprob_start_lens_cpu"
            ):
                output_dict[key] = None
                continue
            assert (
                len(old_value) == num_seqs
            ), f"{key=} {old_value=} {num_seqs=} {batch=}"
            output_dict[key] = old_value[start_seq_index:end_seq_index]

        spec_info = getattr(batch, "spec_info")
        output_spec_info = split_spec_info(
            spec_info=spec_info,
            start_token_index=start_token_index,
            end_token_index=end_token_index,
            start_seq_index=start_seq_index,
            end_seq_index=end_seq_index,
        )
        output_dict["spec_info"] = output_spec_info
        for key in [
            "forward_mode",
            "is_extend_in_batch",
            "return_logprob",
            "req_to_token_pool",
            "token_to_kv_pool",
            "can_run_dp_cuda_graph",
            "dp_padding_mode",
            "global_forward_mode",
            "is_prefill_only",
            "spec_algorithm",
            "capture_hidden_mode",
            "padded_static_len",
            "mrope_positions",  # only used by qwen2-vl, thus not care
            "split_index",  # for split prefill
            "orig_seq_lens",  # only used by qwen-1m, thus not care
        ]:
            output_dict[key] = getattr(batch, key)
        if not batch.forward_mode.is_target_verify():
            assert (
                _compute_extend_num_tokens(batch.input_ids, batch.forward_mode)
                == batch.extend_num_tokens
            ), f"{batch=}"
        extend_num_tokens = _compute_extend_num_tokens(
            output_dict["input_ids"], output_dict["forward_mode"]
        )

        # TODO improve, e.g. unify w/ `init_raw`
        if (
            get_global_server_args().moe_dense_tp_size == 1
            and batch.global_dp_buffer_len is not None
        ):
            sum_len = end_token_index - start_token_index
            global_dp_buffer_len = sum_len
        else:
            global_dp_buffer_len = None

        output_dict.update(
            dict(
                batch_size=end_seq_index - start_seq_index,
                seq_lens_sum=(
                    output_dict["seq_lens_cpu"].sum()
                    if "seq_lens_cpu" in output_dict
                    else None
                ),
                extend_num_tokens=extend_num_tokens,
                attn_backend=output_attn_backend,
                num_token_non_padded=out_num_token_non_padded,
                # TODO: handle it when we need TBO + DeepSeek V3.2
                num_token_non_padded_cpu=None,
                tbo_split_seq_index=None,
                tbo_parent_token_range=(start_token_index, end_token_index),
                tbo_children=None,
                global_num_tokens_gpu=None,
                global_num_tokens_cpu=None,
                global_dp_buffer_len=global_dp_buffer_len,
                global_num_tokens_for_logprob_gpu=None,
                global_num_tokens_for_logprob_cpu=None,
                sampling_info=None,
                # For logits and logprobs post processing, thus we do not care
                temp_scaled_logprobs=False,
                temperature=None,
                top_p_normalized_logprobs=False,
                top_p=None,
                mm_inputs=None,
                top_logprobs_nums=None,
                token_ids_logprobs=None,
                next_token_logits_buffer=None,
            )
        )

        errors = []
        for field in dataclasses.fields(ForwardBatch):
            if getattr(batch, field.name) is not None and field.name not in output_dict:
                errors.append(
                    f"Field {field.name} has value, but is not yet supported (value={getattr(batch, field.name)} batch={batch})"
                )
        if len(errors) > 0:
            raise Exception(f"{len(errors)} errors happen:\n" + "\n\n".join(errors))

        return ForwardBatch(**output_dict)

    @classmethod
    def compute_tbo_children_num_token_non_padded(cls, batch: ForwardBatch):
        return cls.compute_tbo_children_num_token_non_padded_raw(
            tbo_split_token_index=cls._compute_split_token_index(batch),
            num_token_non_padded=len(batch.input_ids),
        )

    @classmethod
    def compute_tbo_children_num_token_non_padded_raw(
        cls, tbo_split_token_index: int, num_token_non_padded: int
    ):
        # TODO we may make padding on both sub-batches to make it slightly more balanced
        value_a = min(tbo_split_token_index, num_token_non_padded)
        value_b = max(0, num_token_non_padded - tbo_split_token_index)
        return torch.tensor([value_a, value_b], dtype=torch.int32).to(
            device=get_global_server_args().device, non_blocking=True
        )

    @classmethod
    def _compute_split_token_index(cls, batch: ForwardBatch):
        token_num_per_seq = get_token_num_per_seq(
            forward_mode=batch.forward_mode, spec_info=batch.spec_info
        )
        return compute_split_token_index(
            split_seq_index=batch.tbo_split_seq_index,
            forward_mode=batch.forward_mode,
            extend_seq_lens=batch.extend_seq_lens_cpu,
            token_num_per_seq=token_num_per_seq,
        )


def _compute_extend_num_tokens(input_ids, forward_mode: ForwardMode):
    if (
        forward_mode.is_decode()
        or forward_mode.is_idle()
        or forward_mode.is_target_verify()
    ):
        return None
    elif forward_mode.is_extend():
        return input_ids.shape[0]
    raise NotImplementedError


# -------------------------------- Execution ---------------------------------------


def model_forward_maybe_tbo(
    layers,
    enable_tbo: bool,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    input_data_scatter_mode: ScatterMode,
    residual: Optional[torch.Tensor],
    zero_allocator: Optional[BumpAllocator] = None,
):
    inputs = dict(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        residual=residual,
        zero_allocator=zero_allocator,
    )
    layer_input_scatter_mode = layers[0].layer_scatter_modes.layer_input_mode
    operations_strategy = OperationsStrategy.init_new_tbo(
        layers, forward_batch.global_forward_mode
    )
    if enable_tbo:
        return _model_forward_tbo(
            inputs=inputs,
            operations_strategy=operations_strategy,
            input_data_scatter_mode=input_data_scatter_mode,
            layer_input_scatter_mode=layer_input_scatter_mode,
        )
    else:
        return _model_forward_non_tbo(inputs, operations_strategy)


def _model_forward_tbo(
    inputs,
    operations_strategy: OperationsStrategy,
    input_data_scatter_mode: ScatterMode,
    layer_input_scatter_mode: ScatterMode,
):
    inputs_arr = _model_forward_tbo_split_inputs(
        **inputs,
        input_data_scatter_mode=input_data_scatter_mode,
        layer_input_scatter_mode=layer_input_scatter_mode,
    )
    del inputs

    context = (
        empty_context()
        if _is_hip
        else deep_gemm_wrapper.configure_deep_gemm_num_sms(
            operations_strategy.deep_gemm_num_sms
        )
    )

    with context:
        outputs_arr = execute_overlapped_operations(
            inputs_arr=inputs_arr,
            operations_arr=[operations_strategy.operations] * 2,
            delta_stages=[0, operations_strategy.tbo_delta_stages],
        )

    return _model_forward_tbo_merge_outputs(*outputs_arr)


def _model_forward_non_tbo(inputs, operations_strategy: OperationsStrategy):
    outputs = execute_operations(inputs, operations_strategy.operations)
    return outputs["hidden_states"], outputs["residual"]


def _model_forward_tbo_split_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    zero_allocator: Optional[BumpAllocator],
    input_data_scatter_mode: ScatterMode,
    layer_input_scatter_mode: ScatterMode,
) -> List[Dict]:
    tbo_splitter_scatter_mode = ScatterMode.TP_ATTN_FULL
    context = CommunicateContext.init_new()

    hidden_states, residual = CommunicateSummableTensorPairFn.execute(
        hidden_states_input_mode=input_data_scatter_mode,
        residual_input_mode=input_data_scatter_mode,
        output_mode=tbo_splitter_scatter_mode,
        hidden_states=hidden_states,
        residual=residual,
        forward_batch=forward_batch,
        context=context,
    )

    inputs_arr = _model_forward_tbo_split_inputs_raw(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        zero_allocator=zero_allocator,
    )

    def _post_transform(hidden_states, residual, forward_batch, **kwargs):
        hidden_states, residual = CommunicateSummableTensorPairFn.execute(
            hidden_states_input_mode=tbo_splitter_scatter_mode,
            residual_input_mode=tbo_splitter_scatter_mode,
            output_mode=layer_input_scatter_mode,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=context,
        )
        return dict(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            **kwargs,
        )

    return [_post_transform(**inputs) for inputs in inputs_arr]


def _model_forward_tbo_split_inputs_raw(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    zero_allocator: Optional[BumpAllocator],
) -> List[Dict]:
    return [
        dict(
            **_model_forward_filter_inputs(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                output_forward_batch=output_forward_batch,
                tbo_subbatch_index=tbo_subbatch_index,
            ),
            **(
                dict(zero_allocator=zero_allocator)
                if zero_allocator is not None
                else {}
            ),
        )
        for tbo_subbatch_index, output_forward_batch in enumerate(
            forward_batch.tbo_children
        )
    ]


def _model_forward_filter_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    output_forward_batch: ForwardBatch,
    tbo_subbatch_index: int,
) -> Dict:
    token_slice = slice(*output_forward_batch.tbo_parent_token_range)
    return dict(
        hidden_states=hidden_states[token_slice],
        residual=None if residual is None else residual[token_slice],
        positions=positions[token_slice],
        forward_batch=output_forward_batch,
        tbo_subbatch_index=tbo_subbatch_index,
    )


def _model_forward_tbo_merge_outputs(output_a, output_b):
    def _handle_key(name):
        value_a = output_a[name]
        value_b = output_b[name]
        assert (value_a is None) == (value_b is None)
        if value_a is None:
            return None
        return torch.concat([value_a, value_b], dim=0)

    return _handle_key("hidden_states"), _handle_key("residual")


# -------------------------------- Utilities and wrappers ---------------------------------------


class MaybeTboDeepEPDispatcher:
    def __init__(self, **kwargs):
        num_inner_dispatchers = 2 if is_tbo_enabled() else 1
        if get_moe_a2a_backend().is_deepep():
            self._inners = [
                DeepEPDispatcher(**kwargs) for _ in range(num_inner_dispatchers)
            ]
        elif get_moe_a2a_backend().is_mooncake():
            self._inners = [
                MooncakeEPDispatcher(**kwargs) for _ in range(num_inner_dispatchers)
            ]

    def _execute(self, name, tbo_subbatch_index: Optional[int] = None, **kwargs):
        return getattr(self._inners[tbo_subbatch_index or 0], name)(**kwargs)

    def dispatch(self, **kwargs) -> DispatchOutput:
        return self._execute("dispatch", **kwargs)

    def dispatch_a(self, **kwargs):
        return self._execute("dispatch_a", **kwargs)

    def dispatch_b(self, **kwargs):
        return self._execute("dispatch_b", **kwargs)

    def combine(self, **kwargs) -> torch.Tensor:
        return self._execute("combine", **kwargs)

    def combine_a(self, **kwargs):
        return self._execute("combine_a", **kwargs)

    def combine_b(self, **kwargs):
        return self._execute("combine_b", **kwargs)

    def set_quant_config(self, quant_config: dict):
        for inner in self._inners:
            inner.set_quant_config(quant_config)
