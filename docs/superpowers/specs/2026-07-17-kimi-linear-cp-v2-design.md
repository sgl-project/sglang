# Kimi-Linear CP-v2 Layer Transitions

## Context

PR #31619 moves MLA prefill context parallelism to the CP-v2 strategy API. At
the model boundary, CP-v2 splits embeddings and positions into the selected CP
layout, and after the model body it gathers hidden states before logits.

Kimi-Linear alternates two attention implementations with different token
layout requirements:

- Kimi Delta Attention (KDA) is tensor parallel and must receive the complete
  token batch on every TP rank.
- Multi-head Latent Attention (MLA) participates in prefill context parallelism
  and must receive the rank-local zigzag token shard.

The model carries both `hidden_states` and a deferred `residual` between decoder
layers. They must always use the same token layout.

## Goals

- Support Kimi-Linear prefill with `--tp 4 --attn-cp-size 4
  --enable-prefill-cp --cp-strategy zigzag`.
- Gather CP-sharded layer state before each KDA region.
- Split replicated layer state before each MLA region.
- Reuse the active CP strategy for ordering, padding, and collectives.
- Preserve the non-CP and decode paths.
- Verify correctness with focused unit tests and GSM8K on four GB300 GPUs.

## Non-goals

- Adding a new CP strategy or attention backend.
- Enabling the unfinished interleave strategy.
- Changing KDA kernels, MLA kernels, or KV-cache semantics.
- Supporting Kimi K2.5's multimodal wrapper in this change.
- Optimizing transition collectives beyond the minimum correct implementation.

## Design

### Communicator

Add `KimiLinearCPV2LayerCommunicator` under
`python/sglang/srt/layers/cp/kimi_linear.py`. The class owns only the transition
between the two token layouts; it does not replace the general TP/DP/MoE
`LayerCommunicator` in `layers/communicator.py`.

Each `KimiDecoderLayer` constructs the communicator with:

- Whether the current layer is KDA.
- Whether the preceding global layer is KDA. Layer zero has no preceding layer.

At the beginning of `KimiDecoderLayer.forward`, the communicator receives
`hidden_states`, `residual`, and `forward_batch`, and returns state in the layout
required by the current layer.

It is active only when `is_cp_v2_active(forward_batch)` is true. Otherwise it
returns its inputs without communication.

### Transition table

| Incoming state | Current layer | Operation |
| --- | --- | --- |
| Model-entry CP shard | First KDA | Gather to complete token order |
| Replicated KDA output | KDA | No-op |
| Replicated KDA output | MLA | Split with the active CP strategy |
| CP-sharded MLA output | MLA | No-op |
| CP-sharded MLA output | KDA | Gather to complete token order |

The first Kimi-Linear layer is KDA, so model-entry embeddings are gathered
before layer zero. The final Kimi-Linear layer is MLA, so it remains CP-sharded;
the CP-v2 eager runner performs the existing model-exit gather before logits.

### Hidden state and residual

When `residual` is present, both tensors undergo the same transition. A gather
uses `ContextParallelStrategy.gather_hidden_states` so zigzag rank ordering,
ragged batches, and padding are restored exactly as at the model boundary. A
split uses `ContextParallelStrategy.shard_hidden_states`.

The initial layer has `residual=None`; only `hidden_states` is gathered there.
No residual addition is moved across the transition.

### Positions

Position IDs remain CP-sharded after the model-entry split. KDA's forward path
does not consume `positions`, while MLA requires positions aligned with its
rank-local hidden-state shard. Consequently, position IDs do not need gather
and split transitions.

### CP-v2 activation

Add `KimiLinearForCausalLM` to `CP_V2_DEFAULT_MODEL_CLASSES`. Also expose the
model's input embedding layer through `get_input_embeddings`, which the CP-v2
eager runner uses to embed the complete token batch before its initial split.

The change remains restricted to CP-v2 context-parallel extend batches through
the existing `is_cp_v2_active` gate. Decode and legacy CP-v1 behavior are
unchanged.

## Error handling and invariants

- The communicator requires an initialized CP strategy whenever CP-v2 is
  active; this follows the existing CP-v2 invariant.
- Both tensors must have matching token dimensions before a joint transition.
- The strategy metadata prepared by the eager runner is the single source of
  truth for split and gather ordering.
- A transition is selected from static model layer types, not inferred from
  runtime tensor lengths.

## Testing

Focused CPU unit tests will use a recording strategy to verify:

- Model entry to first KDA gathers `hidden_states` and accepts `residual=None`.
- KDA-to-MLA splits both `hidden_states` and `residual`.
- MLA-to-KDA gathers both tensors.
- KDA-to-KDA, MLA-to-MLA, decode, and inactive CP-v2 paths are no-ops.
- Communicator integration uses the model's configured KDA/MLA layer sequence.

The existing zigzag strategy tests cover permutation and ragged-batch ordering;
an additional transition round-trip test will ensure the communicator uses
those strategy operations without changing order.

End-to-end verification will launch Kimi-Linear on `baizhou-dev-2` with four
GB300 GPUs and the requested flags, then run the repository's GSM8K evaluation.
The result will be compared with the same model's non-CP baseline or its known
expected accuracy, and server logs will be checked for collective, shape, and
KV-cache errors.

## Delivery

The implementation will be a draft stacked PR targeting
`sgl-project/sglang:cp-v2-mla-prefill`. After PR #31619 merges, the PR can be
retargeted or rebased onto `main`.
