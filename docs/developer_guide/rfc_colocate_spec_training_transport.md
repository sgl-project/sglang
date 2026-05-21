# RFC: Pluggable hidden-states transport for the online speculative-training callback

> Status: **Draft / placeholder.** This document accompanies the
> tracking issue for design discussion. Implementation lands in a
> follow-up PR once the API shape is agreed.

## Problem

SGLang's online speculative-training callback writes hidden states
(`hidden_states`, `aux_hidden_states`, `last_hidden_states`,
`target_logits`) to a Mooncake KV store keyed by UUID. The trainer
process reads them back through the same store. This is the right
shape for **disaggregated** layouts (trainer and engine on different
machines), and a related path for **offline** capture is being added
in #13868.

For **colocated** layouts — trainer and engine sharing the same GPUs
(MPS, single-node multi-role) — the Mooncake hop is wasted bandwidth
and serialization cost: each hidden-states tensor traverses engine GPU
→ Mooncake store → trainer GPU on the same device, while the two
processes could exchange them directly over `dist.batch_isend_irecv`
on a shared default process group.

Frameworks doing colocate RL / online speculative-training (TorchSpec
is one such caller; verl, slime, OpenRLHF have related needs per
#24119) currently maintain out-of-tree patches to bypass the Mooncake
writer. This RFC proposes a small, pluggable transport surface so that
work can move upstream.

## Proposal

Three patch surfaces, intended to be the minimum that unlocks
colocate without inventing a new top-level concept:

### 1. Distributed init: join an external default PG if one already exists

When the scheduler subprocess boots, it normally calls
`torch.distributed.init_process_group` to bring up its TP world. For
colocate, the framework has already brought up a union default PG of
size `2N` (N trainers + N engine TP workers) before launching the
scheduler. The scheduler should:

- detect the pre-existing init via `dist.is_initialized()` (and/or a
  sentinel env var the framework sets);
- skip its own `init_process_group`;
- derive its TP group as a contiguous slice of the existing default
  (`dist.new_group(ranks=range(N, 2N))`).

This is the pattern verl, slime, and TorchSpec all hand-roll today.
The exact env-var contract should be neutral (not
framework-specific).

### 2. Spec-training callback writer: abstract the transport

Today the callback path is hard-wired to `EagleMooncakeStore`. We
propose abstracting it behind a small interface:

```python
class SpecTrainingTransport(Protocol):
    def send(self, tensors: dict[str, torch.Tensor], request_id: str) -> None: ...
    def close(self) -> None: ...
```

with two in-tree implementations:

- `MooncakeTransport` (today's path, default; unchanged behaviour),
- `NoopTransport` (used when the callback is disabled).

Frameworks register their own transport via a registry (the same
shape `--model-config-parser` recently picked up in #25050) or by
passing an instance through `ServerArgs`. The colocate
NCCL-P2P-over-union-world transport lives in the framework, not in
sglang core.

### 3. Per-TP-rank participation

The callback runs on TP rank 0 only today — the rank that coordinates
the Mooncake write. For colocate P2P, every TP rank needs to
participate so each engine rank can send its local shard to its paired
trainer rank without a TP-0 all-gather + scatter.

Concretely: the dispatch hook should fire on every TP rank, with the
default `MooncakeTransport` continuing to early-return on rank != 0
(preserving today's behaviour). A transport opting into per-rank
participation declares it (e.g. `per_rank: bool = False` on the
transport class).

## Non-goals

- Adding a new transport implementation in sglang core (the colocate
  NCCL transport stays in the framework).
- Changing the disaggregated / Mooncake path's behaviour.
- Cross-cutting changes to weight transfer (`update_weights_from_ipc`
  is the symmetric ask in §2 of #24119 and is independent).

## Related

- #24119 *[Discussion] Tightening SGLang's RL inner loop API* — §2
  raises the colocate question for the **weight-transfer-in**
  direction. This RFC is the **hidden-states-out** symmetric ask.
- #13868 *[Feature] Add aux and last hidden state dumping for EAGLE*
  — same hidden states, **offline** transport (disk). A pluggable
  transport surface lets that PR's `HiddenStateDumper` register as a
  `DiskTransport` cleanly.
- #25050 *Add `--model-config-parser` registry for pluggable config
  formats* — precedent for a small registry indirection.

## What landed in this PR

| File | Status |
|---|---|
| `python/sglang/srt/distributed/torchspec_colocate.py` | **Added** (387 lines). Self-contained helper module: env-var reading, union-PG init, engine-rank computation, the `dist.new_group` "local-only" wrapper that defangs world-collective deadlocks against the trainer half, and the hidden-states writer factory. Module-level imports are stdlib-only; `torch` and `torchspec` imports are lazy (inside functions) so `import sglang.srt.distributed.torchspec_colocate` is safe with or without colocate active. |
| `python/sglang/srt/distributed/parallel_state.py` | **Not yet wired.** The `initialize_model_parallel` extension (`tp_world_ranks` parameter + colocate branches at each group-construction site) needs careful porting against current main's expanded attn/MoE group structure. Deferred to a follow-up PR once names are agreed. |
| `python/sglang/srt/managers/scheduler.py`, `scheduler_output_processor_mixin.py`, `model_executor/model_runner.py` | **Not yet wired.** The scheduler-side hooks reference symbols (`batch.spec_training_info`, `enable_spec_training_mooncake`, `_send_hidden_states_to_mooncake`) introduced by a separate base disagg patch that isn't in upstream. Will land after the transport-registry API shape is agreed in this issue. |

The new module uses TorchSpec-namespaced symbol names (`TORCHSPEC_COLOCATE_*`, `torchspec_colocate`) because that's what the fork-side code looks like today. **Renaming to neutral names is part of what this issue should settle** — the RFC's "Proposed shape" section above is the upstream target.

## Reference implementation (full fork-side patch, against v0.5.8.post1)

The complete fork-side patch is checked in alongside this RFC as
[`rfc_colocate_spec_training_transport.v0.5.8.post1.patch`](rfc_colocate_spec_training_transport.v0.5.8.post1.patch)
(836 lines, against `v0.5.8.post1`). It is provided as a *reference
artifact* so reviewers can see the exact shape of the change without
guessing — **it does not apply cleanly to current main** and is
not the form intended to land upstream.

What the patch touches:

| File | Change |
|---|---|
| `python/sglang/srt/distributed/parallel_state.py` | `initialize_model_parallel` accepts an explicit `tp_world_ranks` list and skips the `world_size != tp * pp` assertion when set (engine occupies `[N, 2N)` of a `2N`-rank union default PG). |
| `python/sglang/srt/distributed/torchspec_colocate.py` | New module: env-var-driven helper to compute the engine's union rank, build the TP rank list, and bootstrap the existing default PG. |
| `python/sglang/srt/managers/scheduler.py` | Route the spec-training callback to the colocate hidden-states sender when the colocate env vars are set. |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | Fire the callback on every TP rank for the colocate transport (default Mooncake path still TP-0-only). |
| `python/sglang/srt/model_executor/model_runner.py` | Pass `tp_world_ranks` through to `initialize_model_parallel` when colocate is active. |

Names in the patch are TorchSpec-namespaced (`torchspec_colocate.py`,
`TORCHSPEC_COLOCATE_*` env vars) because the patch was written for the
fork. Renaming them to a neutral shape is part of what the RFC
discussion is meant to settle — the proposed names above
(transport registry, framework-neutral env vars) are the upstream
target.

## Status

A working end-to-end run with the above patch on a 4×H100 box
(1 engine × TP=4 + 4 trainers × FSDP=4, GPUs shared via MPS) is
green: 6/6 colocate tests pass, training loss decreases monotonically,
peak-alloc flat over 200 steps. Once the API shape is agreed in the
linked issue, the patch will be re-shaped against the names above
and submitted as the implementation PR (against current main).
