import dataclasses
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.quantization.deep_gemm import configure_deep_gemm_num_sms
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.operations import execute_operations, execute_overlapped_operations
from sglang.srt.operations_strategy import OperationsStrategy
from sglang.srt.utils import BumpAllocator, DeepEPMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner


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
        raise NotImplementedError


def _split_array_by_half_sum(arr: Sequence[int]) -> int:
    overall_sum = sum(arr)
    accumulator, split_index = 0, 0
    for value in arr[:-1]:
        accumulator += value
        split_index += 1
        if accumulator >= overall_sum // 2:
            break
    return split_index


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


# -------------------------------- Preparation ---------------------------------------


class TboCudaGraphRunnerUtils:
    @staticmethod
    def compute_tbo_split_seq_index(that: "CudaGraphRunner", num_tokens: int):
        if that.model_runner.server_args.enable_two_batch_overlap:
            tbo_split_seq_index = compute_split_seq_index(
                forward_mode=that.capture_forward_mode,
                num_tokens=num_tokens,
                extend_lens=None,
            )
            # For simplicity, when two_batch_overlap is enabled, we only capture CUDA Graph for tbo=true
            assert (
                tbo_split_seq_index is not None
            ), f"{that.capture_forward_mode=} {num_tokens=}"
        else:
            tbo_split_seq_index = None
        return tbo_split_seq_index


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
        from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

        if batch.tbo_split_seq_index is None:
            return

        tbo_split_token_index = compute_split_token_index(
            split_seq_index=batch.tbo_split_seq_index,
            forward_mode=batch.forward_mode,
            extend_seq_lens=batch.extend_seq_lens_cpu,
        )

        assert isinstance(batch.attn_backend, TboAttnBackend)
        attn_backend_child_a, attn_backend_child_b = batch.attn_backend.children

        child_a = cls.filter_batch(
            batch,
            start_token_index=0,
            end_token_index=tbo_split_token_index,
            start_seq_index=0,
            end_seq_index=batch.tbo_split_seq_index,
            output_attn_backend=attn_backend_child_a,
        )
        child_b = cls.filter_batch(
            batch,
            start_token_index=tbo_split_token_index,
            end_token_index=batch.input_ids.shape[0],
            start_seq_index=batch.tbo_split_seq_index,
            end_seq_index=batch.batch_size,
            output_attn_backend=attn_backend_child_b,
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
    ):
        from sglang.srt.managers.schedule_batch import global_server_args_dict

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
        if global_server_args_dict["moe_dense_tp_size"] == 1:
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
                num_token_non_padded=None,
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
    residual: Optional[torch.Tensor],
    zero_allocator: BumpAllocator,
):
    inputs = dict(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        residual=residual,
        zero_allocator=zero_allocator,
    )
    operations_strategy = OperationsStrategy.init_new_tbo(
        layers, forward_batch.global_forward_mode
    )
    if enable_tbo:
        return _model_forward_tbo(inputs, operations_strategy)
    else:
        return _model_forward_non_tbo(inputs, operations_strategy)


def _model_forward_tbo(inputs, operations_strategy: OperationsStrategy):
    # The attn_tp_size!=1 case is not yet extracted to master
    assert get_attention_tp_size() == 1

    inputs_arr = _model_forward_tbo_split_inputs(**inputs)
    del inputs

    with configure_deep_gemm_num_sms(operations_strategy.deep_gemm_num_sms):
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
    zero_allocator: BumpAllocator,
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
            zero_allocator=zero_allocator,
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
