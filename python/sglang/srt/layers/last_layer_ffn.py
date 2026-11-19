"""Shared last-layer FFN row selection for supported dense decoders.

The optimization is automatic for the measured dense architectures. There is
no environment variable. Quantized, parallel, LoRA, TBO, and full-hidden-state
requests keep the ordinary path, as do input shapes without a measured gain.
"""

import torch

from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode

# Smallest newly extended token count whose paired 95% savings interval stayed
# positive on RTX 4090, BF16, breakable prefill graphs. Shorter totals keep the
# ordinary path. The key is the architecture, not one checkpoint. Each cutoff
# was measured on one model and then used for that architecture.
# qwen2 (Qwen2 and Qwen2.5): Qwen2.5-7B, 128 crossed zero; 512-token totals did not.
# qwen3: Qwen3-4B, 128 crossed zero; 512x1 and larger measured totals did not.
# llama: LlamaForCausalLM with model_type "llama", including Llama 2 and Llama 3.
# The 2048-token cutoff was measured on Llama-3.2-1B.
MIN_EXTEND_TOKENS = {
    "qwen2": 512,
    "qwen3": 512,
    "llama": 2048,
}
# Pruned graphs exist only at buckets replayed with a positive interval.
# The default capture ladder is otherwise left on the ordinary graph.
# Qwen2 adds 16384 because 4096x4 was measured; Qwen3 and Llama were not.
FFN_GRAPH_TOKEN_BUCKETS = {
    "qwen2": (512, 2048, 4096, 8192, 16384),
    "qwen3": (512, 2048, 4096, 8192),
    "llama": (2048, 8192),
}
# Measured batches are 1 and 4. A 16-row variant would cover unmeasured
# batches, so batches above 4 stay on the ordinary path.
FFN_GRAPH_ROWS = (1, 4)
FFN_GRAPH_VARIANT = "last_layer_ffn_"

if set(MIN_EXTEND_TOKENS) != set(FFN_GRAPH_TOKEN_BUCKETS):
    raise RuntimeError("last-layer FFN architecture tables diverged")
for _model_type, _buckets in FFN_GRAPH_TOKEN_BUCKETS.items():
    if any(bucket < MIN_EXTEND_TOKENS[_model_type] for bucket in _buckets):
        raise RuntimeError(f"{_model_type} graph bucket is below its measured minimum")


def eligible_batch(batch):
    """Fail closed unless every request finishes an ordinary prefill.

    CPU metadata is used for decisions; never synchronize a CUDA tensor here.
    Static model/parallel/graph restrictions are checked by the model caller.
    """
    # TBO splits/merges full token rows. Neither its parent nor a child may
    # compact the final FFN output until that layout contract is supported.
    if (
        getattr(batch, "tbo_split_seq_index", None) is not None
        or getattr(batch, "tbo_parent_token_range", None) is not None
        or getattr(batch, "tbo_children", None) is not None
    ):
        return False
    if batch.forward_mode != ForwardMode.EXTEND:
        return False
    if batch.return_logprob or batch.is_prefill_only or batch.spec_info is not None:
        return False
    if batch.capture_hidden_mode not in (None, CaptureHiddenMode.NULL):
        return False
    if batch.return_hidden_states_before_norm:
        return False
    if (
        batch.token_indices_to_pool is not None
        or batch.multi_item_delimiter_indices is not None
    ):
        return False
    if any(x is not None for x in (batch.lora_ids or ())):
        return False
    if any(x is not None for x in (batch.mm_inputs or ())):
        return False
    lens = batch.extend_seq_lens_cpu
    prefixes = batch.extend_prefix_lens_cpu
    final = batch.extend_is_final_cpu
    if lens is None or prefixes is None or final is None:
        return False
    if not (len(lens) == len(prefixes) == len(final) == batch.batch_size):
        return False
    # Cached tokens already have KV. Selection is local to the newly extended
    # rows, whether the prefix came from a cache hit or an earlier chunk.
    if not lens or min(lens) <= 0 or min(prefixes) < 0 or not all(final):
        return False
    if sum(lens) != batch.input_ids.shape[0] or sum(lens) <= len(lens):
        return False
    return batch.extend_seq_lens is not None


def select_attention_rows(hidden_states, batch, indices):
    """Select both operands without applying/reordering residual addition.

    Use the existing eager-break handoff, so the original fused add+RMSNorm
    still consumes the pair and retains its original rounding behavior.
    Attention and all KV writes have already finished; no batch/KV metadata
    is resized. This helper is restricted to the single-device plain update.
    """
    stream = residual_batch.stream_of(batch)
    if stream.pending is not None and not stream.pending.update.is_plain_add:
        raise RuntimeError("Last-layer FFN pruning requires a plain residual update")
    hidden_states = residual_batch.complete_output(hidden_states, batch)
    handoff = residual_batch.export_replay_handoff(hidden_states, batch)
    handoff = handoff._replace(
        value=handoff.value.index_select(0, indices),
        residual=(
            handoff.residual.index_select(0, indices)
            if handoff.residual is not None
            else None
        ),
    )
    return residual_batch.import_replay_handoff(handoff, batch)


def shape_allows_pruning(batch, min_extend_tokens):
    """Enable only shapes with a measured gain for this architecture."""
    if min_extend_tokens is None or batch.batch_size > FFN_GRAPH_ROWS[-1]:
        return False
    lens = batch.extend_seq_lens_cpu
    return bool(lens) and sum(lens) >= min_extend_tokens


def select_graph_token_buckets(model_type, capture_num_tokens):
    """Measured buckets that this server already captures."""
    minimum = MIN_EXTEND_TOKENS[model_type]
    captured = set(capture_num_tokens)
    return tuple(
        bucket
        for bucket in FFN_GRAPH_TOKEN_BUCKETS[model_type]
        if bucket >= minimum and bucket in captured
    )


def graph_rows_for_batch(batch, min_extend_tokens):
    if (
        not shape_allows_pruning(batch, min_extend_tokens)
        or not eligible_batch(batch)
        or batch.input_embeds is not None
        or getattr(batch, "return_pooled_hidden_states", False)
        or getattr(batch, "_original_forward_mode", None) == ForwardMode.MIXED
        or getattr(batch, "global_forward_mode", None) == ForwardMode.MIXED
    ):
        return 0
    return next((n for n in FFN_GRAPH_ROWS if batch.batch_size <= n), 0)


def stage_graph_indices(buffer, seq_lens, rows):
    """Update captured addresses on the replay stream, without a host sync."""
    bs = seq_lens.numel()
    assert 0 < bs <= rows <= buffer.numel()
    torch.cumsum(seq_lens, dim=0, out=buffer[:bs])
    buffer[:bs].sub_(1)
    if bs < rows:
        buffer[bs:rows].zero_()


def configure_last_layer_ffn(
    model, *, supported_architecture, quant_config, model_type
):
    """Enable measured architectures. Unsupported layouts stay on the original path."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_exec, get_parallel
    from sglang.srt.utils import is_cuda

    recognized = model_type in MIN_EXTEND_TOKENS
    model._ffn_pruning_supported = (
        supported_architecture
        and recognized
        and quant_config is None
        and is_cuda()
        and get_parallel().tp_size == 1
        and model.pp_group.world_size == 1
        and not get_parallel().enable_dp_attention
        and get_parallel().attn_cp_size == 1
        and get_exec().graph.cuda_graph_config.prefill.backend
        in (Backend.DISABLED, Backend.BREAKABLE)
    )
    model.enable_last_layer_ffn_pruning = model._ffn_pruning_supported
    if model._ffn_pruning_supported:
        model.last_layer_ffn_min_extend_tokens = MIN_EXTEND_TOKENS[model_type]
        model.last_layer_ffn_graph_buckets = FFN_GRAPH_TOKEN_BUCKETS[model_type]
    else:
        model.last_layer_ffn_min_extend_tokens = None
        model.last_layer_ffn_graph_buckets = ()


def prepare_last_layer_ffn(model, batch, *, get_embedding=False, input_embeds=None):
    """Plan selected rows once, before entering the shared layer boundaries."""
    from sglang.srt.model_executor.runner_utils.prefill_graph import (
        is_in_prefill_graph_scope,
    )

    graph_rows = batch.last_layer_ffn_graph_rows
    if graph_rows:
        batch.last_layer_ffn_pruned = True
    else:
        batch.last_layer_ffn_indices = None
        batch.last_layer_ffn_pruned = False
    if (
        not graph_rows
        and not is_in_prefill_graph_scope()
        and model.enable_last_layer_ffn_pruning
        and model._ffn_pruning_supported
        and shape_allows_pruning(batch, model.last_layer_ffn_min_extend_tokens)
        and not get_embedding
        and input_embeds is None
        and not model.capture_aux_hidden_states
        and not model.model.layers_to_capture
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
        and eligible_batch(batch)
    ):
        batch.last_layer_ffn_indices = (
            torch.cumsum(batch.extend_seq_lens, dim=0) - 1
        ).long()
    return graph_rows
