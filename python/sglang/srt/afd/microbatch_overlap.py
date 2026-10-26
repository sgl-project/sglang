"""A-side execution for fixed-depth AFD decode microbatch overlap."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.afd.dispatcher import (
    get_afd_dispatcher,
    split_afd_batch_size,
)
from sglang.srt.batch_overlap.two_batch_overlap import (
    TboForwardBatchPreparer,
    get_token_num_per_seq,
    split_spec_info,
)
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.compute_sm import compute_sm_budget
from sglang.srt.layers.dp_attention import set_dp_buffer_len
from sglang.srt.model_executor.forward_context import (
    forward_context,
    get_forward_context,
)
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpecInput


_AFD_MICROBATCH_OVERLAP_SUPPORTED_MODEL_ARCHS = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


def assert_afd_microbatch_overlap_supported_model(model_architecture: str) -> None:
    if model_architecture not in _AFD_MICROBATCH_OVERLAP_SUPPORTED_MODEL_ARCHS:
        supported = ", ".join(sorted(_AFD_MICROBATCH_OVERLAP_SUPPORTED_MODEL_ARCHS))
        raise NotImplementedError(
            f"AFD decode microbatch overlap does not support {model_architecture}. "
            f"Supported architectures: {supported}"
        )


def build_afd_attention_backend(*, model_config, creator, num_microbatches):
    from sglang.srt.model_loader.utils import get_architecture_class_name

    assert_afd_microbatch_overlap_supported_model(
        get_architecture_class_name(model_config)
    )
    return AFDMicrobatchAttentionBackend.init_new(creator, num_microbatches)


class AFDMicrobatchAttentionBackend(AttentionBackend):
    """Use the primary backend normally and one backend per AFD child batch."""

    def __init__(
        self,
        primary: AttentionBackend,
        children: list[AttentionBackend],
    ):
        super().__init__()
        if len(children) not in (2, 3):
            raise ValueError(
                "AFD microbatch attention backend requires two or three children"
            )
        self.primary = primary
        self.children = children
        self.num_microbatches = len(children)
        self.token_to_kv_pool = primary.token_to_kv_pool
        self.req_to_token_pool = primary.req_to_token_pool
        self.kv_index_translator = primary.kv_index_translator
        self.extend_dummy_seqs_capped_by_req_pool = (
            primary.extend_dummy_seqs_capped_by_req_pool
        )
        self.data_type = primary.data_type
        self.kv_cache_dtype = primary.kv_cache_dtype
        self._graph_children_num_token_non_padded: Optional[torch.Tensor] = None
        # Owns each parent bucket's captured child global_num_tokens tensor.
        # CUDA graphs record raw device pointers without holding Python
        # references; if these tensors were dropped after capture, the
        # allocator could reuse their memory while captured graphs still
        # read it.
        self._graph_children_global_num_tokens: dict[int, torch.Tensor] = {}

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "primary"), name)

    @property
    def verify_mask(self):
        return self.primary.verify_mask

    @property
    def needs_eagle_custom_mask(self):
        verify_mask = self.verify_mask
        return self.primary.needs_eagle_custom_mask and (
            verify_mask is None or verify_mask.is_read
        )

    @property
    def needs_cpu_seq_lens(self):
        return self.primary.needs_cpu_seq_lens or self.needs_eagle_custom_mask

    @property
    def forward_metadata(self):
        return self.primary.forward_metadata

    @forward_metadata.setter
    def forward_metadata(self, value) -> None:
        self.primary.forward_metadata = value

    @classmethod
    def init_new(
        cls,
        creator: Callable[[], AttentionBackend],
        num_microbatches: int,
    ):
        primary = creator()
        return cls(
            primary=primary,
            children=[creator() for _ in range(num_microbatches)],
        )

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        if get_afd_dispatcher().current_num_children != self.num_microbatches:
            return
        prepare_afd_microbatch_forward_batch(
            forward_batch,
            self.num_microbatches,
            needs_custom_mask=self.needs_eagle_custom_mask,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if (
            forward_batch.afd_microbatch_children is None
            and forward_batch.forward_mode.is_target_verify()
        ):
            self.prepare_forward_batch(forward_batch)
        if forward_batch.afd_microbatch_children is None:
            self.primary.init_forward_metadata(forward_batch)
            return
        with compute_sm_budget(get_afd_dispatcher().compute_num_sms):
            for backend, child in zip(
                self.children, forward_batch.afd_microbatch_children, strict=True
            ):
                backend.init_forward_metadata(child)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        child_plan = get_afd_dispatcher().graph_capture_child_plan(
            forward_batch.batch_size
        )
        if not in_capture:
            self._check_parent_graph_child_plan(forward_batch.batch_size, child_plan)
        if len(child_plan) != self.num_microbatches:
            self.primary.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )
            return
        tokens_per_request = _tokens_per_request(forward_batch)
        raw_batch_size = (
            forward_batch.batch_size
            if in_capture
            else forward_batch.batch_size - forward_batch.num_padding
        )
        child_num_tokens = self._graph_child_num_tokens(
            forward_batch.batch_size,
            raw_batch_size * tokens_per_request,
            tokens_per_request,
            forward_batch.input_ids.device,
        )
        if in_capture:
            child_global_num_tokens_gpu = (
                forward_batch.global_num_tokens_gpu.new_tensor(
                    _split_global_num_tokens(
                        forward_batch.global_num_tokens_cpu,
                        self.num_microbatches,
                        tokens_per_request,
                    )
                )
            )
            self._graph_children_global_num_tokens[forward_batch.batch_size] = (
                child_global_num_tokens_gpu
            )
            prepare_afd_microbatch_forward_batch(
                forward_batch,
                self.num_microbatches,
                raw_num_tokens=raw_batch_size * tokens_per_request,
                child_num_tokens_gpu=child_num_tokens,
                child_global_num_tokens_gpu=child_global_num_tokens_gpu,
                needs_custom_mask=self.needs_eagle_custom_mask,
            )
            child_batches = forward_batch.afd_microbatch_children
        else:
            child_batches = _build_afd_microbatch_child_replay_views(
                forward_batch,
                child_num_tokens,
                self.num_microbatches,
                tokens_per_request,
                needs_custom_mask=self.needs_eagle_custom_mask,
            )
        with compute_sm_budget(get_afd_dispatcher().compute_num_sms):
            for backend, child in zip(self.children, child_batches, strict=True):
                if in_capture:
                    # Adjacent parent buckets can reuse a child batch size with
                    # overlapping slices. Cached metadata must own its storage.
                    child = replace(
                        child,
                        req_pool_indices=child.req_pool_indices.clone(),
                        seq_lens=child.seq_lens.clone(),
                    )
                backend.init_forward_metadata_out_graph(child, in_capture=in_capture)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        if forward_batch.afd_microbatch_children is None:
            self.primary.init_forward_metadata_in_graph(forward_batch)
            return
        with compute_sm_budget(get_afd_dispatcher().compute_num_sms):
            for backend, child in zip(
                self.children, forward_batch.afd_microbatch_children, strict=True
            ):
                backend.init_forward_metadata_in_graph(child)

    def set_cuda_graph_decode_backend_variant(self, variant_label, decode_impl):
        for backend in (self.primary, *self.children):
            setter = getattr(backend, "set_cuda_graph_decode_backend_variant", None)
            if setter is not None:
                setter(variant_label, decode_impl)

    def reset_cuda_graph_decode_backend_variant(self):
        for backend in (self.primary, *self.children):
            resetter = getattr(backend, "reset_cuda_graph_decode_backend_variant", None)
            if resetter is not None:
                resetter()

    def shared_read_ends(self, forward_mode):
        return SharedReadEnds.max_of(
            backend.shared_read_ends(forward_mode)
            for backend in (self.primary, *self.children)
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.primary.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)
        child_max_bs = (max_bs + self.num_microbatches - 1) // self.num_microbatches
        tokens_per_request = max_num_tokens // max_bs
        child_max_num_tokens = child_max_bs * tokens_per_request
        for child in self.children:
            child.init_cuda_graph_state(
                max_bs=child_max_bs,
                max_num_tokens=child_max_num_tokens,
            )

    def on_after_cuda_graph_warmup(self):
        self.primary.on_after_cuda_graph_warmup()
        for child in self.children:
            child.on_after_cuda_graph_warmup()

    def get_cuda_graph_seq_len_fill_value(self):
        fill_value = self.primary.get_cuda_graph_seq_len_fill_value()
        for child in self.children:
            assert fill_value == child.get_cuda_graph_seq_len_fill_value()
        return fill_value

    def _check_parent_graph_child_plan(
        self, batch_size: int, child_plan: tuple[int, ...]
    ) -> None:
        # EAGLE preloads graph metadata before the final AFD graph decision.
        # The runner bucket and the batch plan prepared by begin_batch must
        # still resolve to the same child tuple.
        dispatcher_plan = get_afd_dispatcher().prepared_parent_graph_child_plan()
        backend_plan = child_plan if len(child_plan) == self.num_microbatches else None
        if dispatcher_plan != backend_plan:
            raise RuntimeError(
                "AFD microbatch parent graph replay mismatch: runner bucket "
                f"batch_size={batch_size} resolves to child plan "
                f"{backend_plan}, but the dispatcher prepared child plan "
                f"{dispatcher_plan}"
            )

    def _graph_child_num_tokens(
        self,
        parent_batch_size: int,
        raw_num_tokens: int,
        tokens_per_request: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self._graph_children_num_token_non_padded is None:
            self._graph_children_num_token_non_padded = torch.empty(
                (self.num_microbatches,), dtype=torch.int32, device=device
            )
        child_token_sizes = tuple(
            child_size * tokens_per_request
            for child_size in split_afd_batch_size(
                parent_batch_size, self.num_microbatches
            )
        )
        child_num_tokens = _split_raw_num_tokens(child_token_sizes, raw_num_tokens)
        child_num_tokens_pinned = torch.tensor(
            child_num_tokens, dtype=torch.int32, pin_memory=True
        )
        self._graph_children_num_token_non_padded.copy_(
            child_num_tokens_pinned, non_blocking=True
        )
        return self._graph_children_num_token_non_padded

    def forward(self, *args, **kwargs):
        return self.primary.forward(*args, **kwargs)

    def forward_extend(self, *args, **kwargs):
        return self.primary.forward_extend(*args, **kwargs)

    def forward_decode(self, *args, **kwargs):
        return self.primary.forward_decode(*args, **kwargs)

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.primary.get_indexer_metadata(layer_id, forward_batch)


def _split_afd_spec_info(
    spec_info: Optional[SpecInput],
    start_seq_index: int,
    end_seq_index: int,
    start_token_index: int,
    end_token_index: int,
    mask_offset_cache: Optional[dict[int, int]] = None,
):
    if spec_info is None:
        return None
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    if not isinstance(spec_info, DFlashVerifyInput):
        return split_spec_info(
            spec_info,
            start_seq_index,
            end_seq_index,
            start_token_index,
            end_token_index,
            mask_offset_cache=mask_offset_cache,
        )
    if spec_info.custom_mask is not None or spec_info.ragged_verify_layout is not None:
        raise ValueError("AFD DSpark microbatches require static linear verify")
    return replace(
        spec_info,
        draft_token=(
            spec_info.draft_token[start_token_index:end_token_index]
            if spec_info.draft_token is not None
            else None
        ),
        positions=(
            spec_info.positions[start_token_index:end_token_index]
            if spec_info.positions is not None
            else None
        ),
        live_seq_lens_cpu=(
            spec_info.live_seq_lens_cpu[start_seq_index:end_seq_index]
            if spec_info.live_seq_lens_cpu is not None
            else None
        ),
        kv_loc_cols=(
            spec_info.kv_loc_plan.cols_slice(
                spec_info.kv_loc_cols, slice(start_token_index, end_token_index)
            )
            if spec_info.kv_loc_plan is not None
            else None
        ),
    )


def prepare_afd_microbatch_forward_batch(
    forward_batch: ForwardBatch,
    num_microbatches: int,
    *,
    raw_num_tokens: Optional[int] = None,
    child_num_tokens_gpu: Optional[torch.Tensor] = None,
    child_global_num_tokens_gpu: Optional[torch.Tensor] = None,
    needs_custom_mask: bool = True,
) -> None:
    if not (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_target_verify()
    ):
        raise RuntimeError("AFD microbatch overlap requires decode or target verify")

    tokens_per_request = _tokens_per_request(forward_batch)
    child_sizes = split_afd_batch_size(forward_batch.batch_size, num_microbatches)
    child_token_sizes = tuple(
        child_size * tokens_per_request for child_size in child_sizes
    )
    global_child_sizes = _split_global_num_tokens(
        forward_batch.global_num_tokens_cpu,
        num_microbatches,
        tokens_per_request,
    )
    if raw_num_tokens is None:
        raw_num_tokens = forward_batch.global_num_token_non_padded_cpu
    child_num_tokens = _split_raw_num_tokens(child_token_sizes, raw_num_tokens)
    if child_num_tokens_gpu is None:
        child_num_tokens_gpu = torch.tensor(
            child_num_tokens,
            dtype=torch.int32,
            device=forward_batch.input_ids.device,
        )
    if child_global_num_tokens_gpu is None:
        child_global_num_tokens_gpu = forward_batch.global_num_tokens_gpu.new_tensor(
            global_child_sizes
        )

    spec_info = forward_batch.spec_info
    if not needs_custom_mask and spec_info is not None:
        spec_info = replace(spec_info, custom_mask=None)
    # Reuse upstream field slicing; AFD owns speculative child metadata.
    batch_to_split = (
        replace(forward_batch, spec_info=None)
        if spec_info is not None
        else forward_batch
    )

    children = []
    start_seq = 0
    start_token = 0
    for child_index, (child_size, child_token_size) in enumerate(
        zip(child_sizes, child_token_sizes, strict=True)
    ):
        end_seq = start_seq + child_size
        end_token = start_token + child_token_size
        child = TboForwardBatchPreparer.filter_batch(
            batch_to_split,
            start_token_index=start_token,
            end_token_index=end_token,
            start_seq_index=start_seq,
            end_seq_index=end_seq,
            out_num_token_non_padded=child_num_tokens_gpu[child_index],
            out_num_token_non_padded_cpu=child_num_tokens[child_index],
        )
        child.spec_info = _split_afd_spec_info(
            spec_info,
            start_seq_index=start_seq,
            end_seq_index=end_seq,
            start_token_index=start_token,
            end_token_index=end_token,
        )
        child.global_num_token_non_padded = child_num_tokens_gpu[child_index]
        child.global_num_tokens_cpu = global_child_sizes[child_index]
        child.global_num_tokens_gpu = child_global_num_tokens_gpu[child_index]
        child.global_dp_buffer_len = sum(child.global_num_tokens_cpu)
        children.append(child)
        start_seq = end_seq
        start_token = end_token

    forward_batch.afd_microbatch_children = children


def _split_raw_num_tokens(
    child_sizes: tuple[int, ...], raw_num_tokens: int
) -> tuple[int, ...]:
    child_num_tokens = []
    start = 0
    for child_size in child_sizes:
        end = start + child_size
        child_num_tokens.append(max(0, min(end, raw_num_tokens) - start))
        start = end
    return tuple(child_num_tokens)


def _build_afd_microbatch_child_replay_views(
    forward_batch,
    child_num_tokens: torch.Tensor,
    num_microbatches: int,
    tokens_per_request: int,
    *,
    needs_custom_mask: bool = True,
) -> list[SimpleNamespace]:
    """Cheap per-step child views for out-of-graph replay metadata.

    Follows the runner's build_replay_fb_view pattern: each namespace only
    needs to cover the fields the child attention backends read in
    init_forward_metadata_out_graph, not the full ForwardBatch surface.
    A missing field fails loudly with AttributeError.
    """
    child_sizes = split_afd_batch_size(forward_batch.batch_size, num_microbatches)
    spec_info = forward_batch.spec_info
    if not needs_custom_mask and spec_info is not None:
        spec_info = replace(spec_info, custom_mask=None)
    children = []
    mask_offset_cache = {0: 0}
    start_seq = 0
    start_token = 0
    for child_index, child_size in enumerate(child_sizes):
        end_seq = start_seq + child_size
        end_token = start_token + child_size * tokens_per_request
        child_seq_lens_cpu = (
            forward_batch.seq_lens_cpu[start_seq:end_seq]
            if forward_batch.seq_lens_cpu is not None
            else None
        )
        children.append(
            SimpleNamespace(
                batch_size=child_size,
                forward_mode=forward_batch.forward_mode,
                actual_forward_mode=getattr(
                    forward_batch,
                    "actual_forward_mode",
                    forward_batch.forward_mode,
                ),
                input_ids=forward_batch.input_ids[start_token:end_token],
                req_pool_indices=forward_batch.req_pool_indices[start_seq:end_seq],
                seq_lens=forward_batch.seq_lens[start_seq:end_seq],
                seq_lens_sum=(
                    int(child_seq_lens_cpu.sum())
                    if child_seq_lens_cpu is not None
                    else None
                ),
                seq_lens_cpu=child_seq_lens_cpu,
                num_padding=max(
                    0,
                    end_seq
                    - max(
                        start_seq,
                        forward_batch.batch_size - forward_batch.num_padding,
                    ),
                ),
                mamba_track_indices=(
                    forward_batch.mamba_track_indices[start_seq:end_seq]
                    if forward_batch.mamba_track_indices is not None
                    else None
                ),
                encoder_lens=(
                    forward_batch.encoder_lens[start_seq:end_seq]
                    if forward_batch.encoder_lens is not None
                    else None
                ),
                out_cache_loc=(
                    forward_batch.out_cache_loc[start_token:end_token]
                    if forward_batch.out_cache_loc is not None
                    else None
                ),
                out_cache_loc_virtual=(
                    forward_batch.out_cache_loc_virtual[start_token:end_token]
                    if forward_batch.out_cache_loc_virtual is not None
                    else None
                ),
                out_cache_loc_is_physical=forward_batch.out_cache_loc_is_physical,
                kv_loc_plan=forward_batch.kv_loc_plan,
                kv_loc_cols=(
                    forward_batch.kv_loc_plan.cols_slice(
                        forward_batch.kv_loc_cols, slice(start_token, end_token)
                    )
                    if forward_batch.kv_loc_plan is not None
                    else None
                ),
                spec_info=_split_afd_spec_info(
                    spec_info,
                    start_seq_index=start_seq,
                    end_seq_index=end_seq,
                    start_token_index=start_token,
                    end_token_index=end_token,
                    mask_offset_cache=mask_offset_cache,
                ),
                num_token_non_padded=child_num_tokens[child_index],
            )
        )
        start_seq = end_seq
        start_token = end_token
    return children


def _split_global_num_tokens(
    global_num_tokens: list[int],
    num_microbatches: int,
    tokens_per_request: int,
) -> list[list[int]]:
    per_rank = [
        tuple(
            child_size * tokens_per_request
            for child_size in split_afd_batch_size(
                num_tokens // tokens_per_request, num_microbatches
            )
        )
        for num_tokens in global_num_tokens
    ]
    return [
        [sizes[child_id] for sizes in per_rank] for child_id in range(num_microbatches)
    ]


def _tokens_per_request(forward_batch: ForwardBatch) -> int:
    if forward_batch.forward_mode.is_target_verify():
        return get_token_num_per_seq(
            forward_batch.forward_mode, forward_batch.spec_info
        )
    return 1


def run_afd_microbatch_serial_layers(
    *,
    layers,
    start_layer: int,
    end_layer: int,
    forward_batch: ForwardBatch,
    child_states: list,
    forward_child_layer: Callable,
):
    children = forward_batch.afd_microbatch_children
    parent_context = get_forward_context()
    child_contexts = [
        replace(parent_context, attn_backend=child_backend)
        for child_backend in parent_context.attn_backend.children
    ]

    for layer_id in range(start_layer, end_layer):
        layer = layers[layer_id]
        with (
            compute_sm_budget(get_afd_dispatcher().compute_num_sms),
            get_global_expert_distribution_recorder().with_current_layer(layer_id),
        ):
            for child_index, (child, child_context) in enumerate(
                zip(children, child_contexts, strict=True)
            ):
                set_dp_buffer_len(
                    child.global_dp_buffer_len,
                    child.tbo_padded_len,
                    child.dp_padding_mode.is_max_len(),
                    child.global_num_tokens_cpu,
                )
                with forward_context(child_context):
                    child_states[child_index] = forward_child_layer(
                        layer,
                        child,
                        child_states[child_index],
                        child_index,
                    )

    set_dp_buffer_len(
        forward_batch.global_dp_buffer_len,
        forward_batch.input_ids.shape[0],
        forward_batch.dp_padding_mode.is_max_len(),
        forward_batch.global_num_tokens_cpu,
    )
    return child_states


def run_afd_microbatch_wavefront(
    *,
    layers,
    start_layer: int,
    end_layer: int,
    forward_batch: ForwardBatch,
    child_states: list,
    launch_child_layer: Callable,
    finish_child_layer: Callable,
    after_launch_layer: Optional[Callable] = None,
):
    children = forward_batch.afd_microbatch_children
    parent_context = get_forward_context()
    child_contexts = [
        replace(parent_context, attn_backend=child_backend)
        for child_backend in parent_context.attn_backend.children
    ]
    pending = [None] * len(children)
    dispatcher = get_afd_dispatcher()
    capture_context = (
        dispatcher.capture_parent_graph(forward_batch.batch_size)
        if get_is_capture_mode()
        else nullcontext()
    )
    with (
        compute_sm_budget(dispatcher.compute_num_sms),
        capture_context,
    ):
        # Mirror the parent reset in decode_cuda_graph_runner.run_once: force
        # get_dp_local_info to recompute inside the graph. A warmup-cached
        # cumsum would bake pointers to tensors that are freed after capture,
        # so replay's dp gather/scatter would read garbage offset/size.
        for child in children:
            child.dp_local_start_pos = None
            child.dp_local_num_tokens = None
        for child_index, (child, child_context) in enumerate(
            zip(children, child_contexts, strict=True)
        ):
            _set_child_dp_buffer(child)
            with (
                forward_context(child_context),
                get_global_expert_distribution_recorder().with_current_layer(
                    start_layer
                ),
            ):
                pending[child_index] = launch_child_layer(
                    layers[start_layer],
                    child,
                    child_states[child_index],
                    child_index,
                )

        if after_launch_layer is not None:
            pending = after_launch_layer(layers[start_layer], pending)

        for layer_id in range(start_layer, end_layer):
            next_layer_id = layer_id + 1
            for child_index, (child, child_context) in enumerate(
                zip(children, child_contexts, strict=True)
            ):
                _set_child_dp_buffer(child)
                with forward_context(child_context):
                    child_states[child_index] = finish_child_layer(
                        layers[layer_id],
                        child,
                        pending[child_index],
                        child_index,
                    )
                    if next_layer_id < end_layer:
                        recorder = get_global_expert_distribution_recorder()
                        with recorder.with_current_layer(next_layer_id):
                            pending[child_index] = launch_child_layer(
                                layers[next_layer_id],
                                child,
                                child_states[child_index],
                                child_index,
                            )

            if next_layer_id < end_layer and after_launch_layer is not None:
                pending = after_launch_layer(layers[next_layer_id], pending)

    set_dp_buffer_len(
        forward_batch.global_dp_buffer_len,
        forward_batch.input_ids.shape[0],
        forward_batch.dp_padding_mode.is_max_len(),
        forward_batch.global_num_tokens_cpu,
    )
    return child_states


def split_afd_residual_stream(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> list[torch.Tensor]:
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

    stream = residual_batch.stream_of(forward_batch)
    update = stream.pending.update if stream.pending is not None else None
    hidden_states, residual = stream.export(hidden_states)
    child_hidden_states = []
    for child in forward_batch.afd_microbatch_children:
        rows = slice(*child.tbo_parent_token_range)
        hidden, child.residual_stream = ResidualStream.from_handoff(
            hidden_states[rows],
            residual[rows] if residual is not None else None,
            update,
        )
        child_hidden_states.append(hidden)
    forward_batch.residual_stream = None
    return child_hidden_states


def merge_afd_residual_streams(
    child_hidden_states: list[torch.Tensor], forward_batch: ForwardBatch
) -> torch.Tensor:
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

    children = forward_batch.afd_microbatch_children
    outputs = []
    residuals = []
    first_stream = residual_batch.stream_of(children[0])
    update = first_stream.pending.update if first_stream.pending is not None else None
    for hidden, child in zip(child_hidden_states, children, strict=True):
        stream = residual_batch.stream_of(child)
        assert (stream.pending.update if stream.pending is not None else None) is update
        hidden, residual = stream.export(hidden)
        outputs.append(hidden)
        residuals.append(residual)
        child.residual_stream = None
    hidden_states, forward_batch.residual_stream = ResidualStream.from_handoff(
        torch.cat(outputs),
        torch.cat(residuals) if residuals[0] is not None else None,
        update,
    )
    return hidden_states


def model_forward_afd_microbatch(
    *,
    layers,
    start_layer: int,
    end_layer: int,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
):
    child_states = split_afd_residual_stream(hidden_states, forward_batch)

    def launch_child_layer(layer, child, hidden, _child_index):
        return layer.forward_afd_microbatch_launch(child.positions, hidden, child)

    def finish_child_layer(layer, child, pending, _child_index):
        return layer.forward_afd_microbatch_finish(pending, child)

    child_states = run_afd_microbatch_wavefront(
        layers=layers,
        start_layer=start_layer,
        end_layer=end_layer,
        forward_batch=forward_batch,
        child_states=child_states,
        launch_child_layer=launch_child_layer,
        finish_child_layer=finish_child_layer,
    )
    return merge_afd_residual_streams(child_states, forward_batch)


def _set_child_dp_buffer(child: ForwardBatch) -> None:
    set_dp_buffer_len(
        child.global_dp_buffer_len,
        child.tbo_padded_len,
        child.dp_padding_mode.is_max_len(),
        child.global_num_tokens_cpu,
    )
