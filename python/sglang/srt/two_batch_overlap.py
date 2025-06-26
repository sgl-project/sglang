import dataclasses
import logging
from typing import Dict, List, Optional, Sequence

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSummableTensorPairFn,
    ScatterMode,
)
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.operations import execute_operations, execute_overlapped_operations
from sglang.srt.operations_strategy import OperationsStrategy
from sglang.srt.utils import BumpAllocator, DeepEPMode, get_bool_env_var

_tbo_debug = get_bool_env_var("SGLANG_TBO_DEBUG")

logger = logging.getLogger(__name__)


# -------------------------------- Compute Basic Info ---------------------------------------


# TODO: may smartly disable TBO when batch size is too small b/c it will slow down
def compute_split_seq_index(
    forward_mode: "ForwardMode",
    num_tokens: int,
    extend_lens: Optional[Sequence[int]],
) -> Optional[int]:
    if forward_mode.is_extend():
        assert extend_lens is not None
        return _split_array_by_half_sum(extend_lens)
    elif forward_mode.is_decode():
        return num_tokens // 2
    elif forward_mode.is_idle():
        assert num_tokens == 0
        return 0
    else:
        raise NotImplementedError()


def _split_array_by_half_sum(arr: Sequence[int]) -> int:
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


def compute_split_token_index(
    split_seq_index: int,
    forward_mode: "ForwardMode",
    extend_seq_lens: Optional[Sequence[int]],
) -> int:
    if forward_mode.is_extend():
        assert extend_seq_lens is not None
        return sum(extend_seq_lens[:split_seq_index])
    elif forward_mode.is_decode():
        return split_seq_index
    elif forward_mode.is_idle():
        assert split_seq_index == 0
        return 0
    else:
        raise NotImplementedError


def compute_split_indices_for_cuda_graph_replay(
    forward_mode: ForwardMode,
    cuda_graph_num_tokens: int,
):
    forward_mode_for_tbo_split = (
        forward_mode if forward_mode != ForwardMode.IDLE else ForwardMode.DECODE
    )
    tbo_split_seq_index = compute_split_seq_index(
        forward_mode=forward_mode_for_tbo_split,
        num_tokens=cuda_graph_num_tokens,
        extend_lens=None,
    )
    tbo_split_token_index = compute_split_token_index(
        split_seq_index=tbo_split_seq_index,
        forward_mode=forward_mode_for_tbo_split,
        extend_seq_lens=None,
    )
    return tbo_split_seq_index, tbo_split_token_index


# -------------------------------- Preparation ---------------------------------------


class TboCudaGraphRunnerPlugin:
    def __init__(self):
        self._tbo_children_num_token_non_padded = torch.zeros((2,), dtype=torch.int32)

    def capture_one_batch_size(self, batch: ForwardBatch, num_tokens: int):
        if not global_server_args_dict["enable_two_batch_overlap"]:
            return

        batch.tbo_split_seq_index = compute_split_seq_index(
            forward_mode=batch.forward_mode,
            num_tokens=num_tokens,
            extend_lens=None,
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
        self, forward_mode: ForwardMode, bs: int, num_token_non_padded: int
    ):
        tbo_split_seq_index, tbo_split_token_index = (
            compute_split_indices_for_cuda_graph_replay(
                forward_mode=forward_mode,
                # TODO support bs!=num_tokens
                cuda_graph_num_tokens=bs,
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
        self, local_batch, deepep_mode, enable_deepep_moe, enable_two_batch_overlap
    ):
        self.enable_two_batch_overlap = enable_two_batch_overlap

        if local_batch is not None:
            self.local_tbo_split_seq_index = compute_split_seq_index(
                forward_mode=local_batch.forward_mode,
                num_tokens=local_batch.input_ids.shape[0],
                extend_lens=local_batch.extend_lens,
            )
            resolved_deepep_mode = deepep_mode.resolve(local_batch.forward_mode)
            local_can_run_tbo = (self.local_tbo_split_seq_index is not None) and not (
                local_batch.forward_mode.is_extend()
                and enable_deepep_moe
                and (resolved_deepep_mode == DeepEPMode.low_latency)
            )
        else:
            self.local_tbo_split_seq_index = 0
            local_can_run_tbo = True

        local_forward_mode = self._compute_local_forward_mode(local_batch)

        return local_can_run_tbo, local_forward_mode

    def compute_output(self, partial_global_info):
        local_can_run_tbo_aggregated = min(partial_global_info[:, 0, 0].tolist())
        forward_modes = partial_global_info[:, 0, 1].tolist()

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
        converted_forward_modes = [
            ForwardMode.DECODE.value if x == ForwardMode.IDLE.value else x
            for x in forward_modes
        ]
        forward_mode_agree = TboDPAttentionPreparer._is_all_same(
            converted_forward_modes
        )
        global_forward_mode = (
            ForwardMode(converted_forward_modes[0]) if forward_mode_agree else None
        )
        return global_forward_mode, forward_mode_agree

    @staticmethod
    def _is_all_same(x):
        return all(value == x[0] for value in x)


class TboForwardBatchPreparer:
    @classmethod
    def prepare(cls, batch: ForwardBatch):
        if batch.tbo_split_seq_index is None:
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

        if _tbo_debug:
            logger.info(
                f"TboForwardBatchPreparer.prepare "
                f"tbo_split_seq_index={batch.tbo_split_seq_index} "
                f"tbo_split_token_index={tbo_split_token_index} "
                f"extend_seq_lens={batch.extend_seq_lens_cpu}"
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
            end_seq_index=batch.tbo_split_seq_index,
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

        assert batch.tbo_children is None
        batch.tbo_children = [child_a, child_b]

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
            "lora_paths",
        ]:
            old_value = getattr(batch, key)
            if old_value is None:
                continue
            assert (
                len(old_value) == num_seqs
            ), f"{key=} {old_value=} {num_seqs=} {batch=}"
            output_dict[key] = old_value[start_seq_index:end_seq_index]

        for key in [
            "forward_mode",
            "return_logprob",
            "req_to_token_pool",
            "token_to_kv_pool",
            "can_run_dp_cuda_graph",
            "global_forward_mode",
            "spec_info",
            "spec_algorithm",
            "capture_hidden_mode",
            "padded_static_len",
            "mrope_positions",  # only used by qwen2-vl, thus not care
        ]:
            output_dict[key] = getattr(batch, key)

        assert (
            _compute_extend_num_tokens(batch.input_ids, batch.forward_mode)
            == batch.extend_num_tokens
        ), f"{batch=}"
        extend_num_tokens = _compute_extend_num_tokens(
            output_dict["input_ids"], output_dict["forward_mode"]
        )

        # TODO improve, e.g. unify w/ `init_raw`
        if (
            global_server_args_dict["moe_dense_tp_size"] == 1
            and batch.gathered_buffer is not None
        ):
            sum_len = end_token_index - start_token_index
            gathered_buffer = torch.zeros(
                (sum_len, batch.gathered_buffer.shape[1]),
                dtype=batch.gathered_buffer.dtype,
                device=batch.gathered_buffer.device,
            )
        else:
            gathered_buffer = None

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
                tbo_split_seq_index=None,
                tbo_parent_token_range=(start_token_index, end_token_index),
                tbo_children=None,
                global_num_tokens_gpu=None,
                global_num_tokens_cpu=None,
                gathered_buffer=gathered_buffer,
                global_num_tokens_for_logprob_gpu=None,
                global_num_tokens_for_logprob_cpu=None,
                sampling_info=None,
                # For logits and logprobs post processing, thus we do not care
                temp_scaled_logprobs=False,
                temperature=None,
                top_p_normalized_logprobs=False,
                top_p=None,
                mm_inputs=None,
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
            device=global_server_args_dict["device"], non_blocking=True
        )

    @classmethod
    def _compute_split_token_index(cls, batch: ForwardBatch):
        return compute_split_token_index(
            split_seq_index=batch.tbo_split_seq_index,
            forward_mode=batch.forward_mode,
            extend_seq_lens=batch.extend_seq_lens_cpu,
        )


def _compute_extend_num_tokens(input_ids, forward_mode: ForwardMode):
    if forward_mode.is_extend():
        return input_ids.shape[0]
    elif forward_mode.is_decode() or forward_mode.is_idle():
        return None
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

    with deep_gemm_wrapper.configure_deep_gemm_num_sms(
        operations_strategy.deep_gemm_num_sms
    ):
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
        num_inner_dispatchers = (
            2 if global_server_args_dict["enable_two_batch_overlap"] else 1
        )
        self._inners = [
            DeepEPDispatcher(**kwargs) for _ in range(num_inner_dispatchers)
        ]

    def _execute(self, name, tbo_subbatch_index: Optional[int] = None, **kwargs):
        return getattr(self._inners[tbo_subbatch_index or 0], name)(**kwargs)

    def dispatch(self, **kwargs):
        return self._execute("dispatch", **kwargs)

    def dispatch_a(self, **kwargs):
        return self._execute("dispatch_a", **kwargs)

    def dispatch_b(self, **kwargs):
        return self._execute("dispatch_b", **kwargs)

    def combine(self, **kwargs):
        return self._execute("combine", **kwargs)

    def combine_a(self, **kwargs):
        return self._execute("combine_a", **kwargs)

    def combine_b(self, **kwargs):
        return self._execute("combine_b", **kwargs)
