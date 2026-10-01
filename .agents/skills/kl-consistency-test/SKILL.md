---
name: kl-consistency-test
description: Write, calibrate, and debug the prefill-vs-decode logprob (KL) consistency tests in sglang -- the two independent conditions a zero requires (every operator batch-invariant, and the two paths computing the same function), which helper separates them, how to pick a threshold once they hold, and how to localize a divergence to a single operator. Use when adding a KL test to a model, picking or defending a kl_div threshold, or investigating a KL number that is too high.
---

# KL Consistency Tests

## What the test is for

`kl_test_utils` scores the same token twice -- once as a prefill input logprob, once
as a decode output logprob -- and compares. The two paths run different kernels over
different shapes, so agreement is a statement about **state**, not about answer
quality: it catches a radix-cache prefix that does not reproduce a fresh prefill, a
stale conv/mamba checkpoint, a SWA pool that evicted something it still needed.

gsm8k passing says nothing about this. Accuracy is insensitive to a handful of
corrupted tokens; the KL check is not.

## Two independent conditions produce a zero

Reaching bit-identity needs both, and they fail for unrelated reasons. Knowing which
one a nonzero belongs to is most of the debugging.

1. **Every operator on the path is batch-invariant.** A token's result must not
   depend on how many tokens share its forward. Note this is a property *across* the
   two paths, not a property of each: a kernel can be perfectly reproducible at M=1
   and again at M=N while disagreeing between them, which is exactly what a
   tile-size switch or a message-size-dependent reduction does.

2. **The two paths compute the same function.** Decode's context and state at a
   position must equal what a fresh prefill computes there -- the same KV set, the
   same sliding window, the same conv/mamba state, a restored cache prefix that
   reproduces a recomputed one. This is logic, not arithmetic, and it survives any
   amount of numerical hygiene.

The conditions are independent, and one measurement separates them: with (1)
satisfied, `match` and `decode_cache_hit` read exactly 0 while `prefill_cache_hit`
stays nonzero when a prefix restore is wrong. Same server, same prompts -- float
noise cannot pick a code path, so a helper-specific divergence is (2).

Order the work accordingly. Settle (1) first: until it holds, its noise is orders of
magnitude above anything (2) produces and hides it completely.

## The three helpers differ in what touches the cache

`KLDivergenceMixin` runs the last two. Pick deliberately -- they are not
interchangeable, and only the cache-hit pair exercises prefix reuse.

| Helper | Cache involvement |
|---|---|
| `..._match_helper` | both sides flush; **no cache at all** |
| `..._match_prefill_cache_hit_helper` | prompt is prefilled once to warm the cache, then the generation prefill restores from it |
| `..._match_decode_cache_hit_helper` | decode side runs on a warmed cache |

A divergence confined to one helper is diagnostic. `match` clean but
`prefill_cache_hit` dirty means the restore path is wrong, not the arithmetic --
float noise does not pick a code path.

## Run it the way CI runs it

`KLDivergenceMixin` defaults: `max_samples=32`, `max_new_tokens=512`. Do not
characterize with fewer.

`avg_kl_div` is the k3 estimator, `exp(logr) - 1 - logr`, applied to the sampled
token's logprob. It is exponentially sensitive to the tail, so the mean is carried by
a handful of tokens. At 4 samples the same config measured 0.049 to 0.158 -- a 3x
spread that invalidates any A/B comparison drawn from it.

When characterizing rather than gating, report tail statistics -- the fraction of
tokens past a threshold, and the max -- rather than the mean.

Generate past the sliding window if the model has one, so decode carries the window
through the handover from prompt tokens to generated ones.

## Condition 1: determinism is not batch-invariance

This distinction decides whether a threshold means anything.

- **Deterministic**: same input, same shape, same result on every run.
- **Batch-invariant**: a token's result does not depend on how many other tokens
  share its batch.

The KL check compares a prefill of thousands of tokens against decode steps of one,
so it measures the second. `--enable-deterministic-inference` buys both -- it swaps
the aten kernels for fixed-reduction versions and pins the NCCL algorithm and channel
count -- but only for kernels it covers. Custom kernels that never reach an aten op
are outside `batch_invariant_ops` and stay shape-dependent.

The consequence: a nonzero KL under deterministic inference that appears in every
helper alike means some kernel on the path is still batch-dependent. Localize it
(below) rather than widening the threshold.

Background, and the source of the fixed-reduction approach the aten overrides take:
[Defeating nondeterminism in LLM inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

### How much batch-invariance an operator needs

For a **token-wise** operator -- GEMM, norm, activation, the router's linear -- a token's
output depends only on that token's row, so pinning the reduction order is the whole
requirement. Once its result is independent of how many rows share the launch, it is done.

Two kinds need more than that, and they are where the remaining nonzero usually lives:

- **Operators that reduce across tokens** -- attention over a KV range, and any collective.
  Fixing the arithmetic order is not enough if the *extent* still varies: an all-reduce whose
  tree shape follows the message size, or an attention split whose block boundary follows the
  query count, gives a token a different reduction depending on its batch. Pin the shape, not
  just the order.
- **Operators that carry state across calls** -- conv windows, SSM checkpoints. These are
  batch-invariant per call and still diverge, because what they store is reused by a later
  request. That is condition 2, and no amount of reduction-order work reaches it.

So "make everything batch-invariant" closes condition 1 for the token-wise majority, and the
residual after that is concentrated in these two classes.

### MoE amplifies this to a degree dense models do not

Top-k routing is a discrete decision over near-tied scores. A 1e-8 difference in gate
weights flips which experts a token is routed to, the outputs diverge completely, and
42 layers compound it. Measured on one MoE checkpoint: a gate GEMM that switched
tiling between M=8 and M=16 produced a 1.6e-5 logits difference, which became 20-37
nat on individual high-confidence tokens and a KL of 0.177.

A dense model of comparable size shows the same root cause as ~1e-4. So a KL in the
hundredths is not evidence of a worse bug on a MoE model -- it is the same class of
numerical difference, amplified. Do not calibrate a MoE threshold by analogy to a
dense one.

## Condition 2: the two paths must compute the same function

Once condition 1 holds, whatever remains is a state bug, and the helper it appears in
names the path. A restore that does not reproduce a recomputed prefix shows up in
`prefill_cache_hit` alone; the other two stay at exactly 0.

What the signature looks like, and how to read it:

- **Which sequences.** Divergence concentrated in a couple of requests out of a
  batch, with the rest bit-identical, is a condition triggered by those requests --
  not a systematic offset. Compare their prompt lengths, `cached_tokens`, and page
  and checkpoint-interval remainders against the ones that pass.
- **Where in the generation.** Contiguous from the first generated token means the
  state was already wrong when generation began, so the fault is in the prefix
  restore rather than in decode. Divergence starting mid-generation points instead at
  something that happens during decode -- a window handover, a checkpoint rotation.
- **Whether it is a race.** Re-run under different configurations that should not
  matter (page size, TP degree, buffer strategy). Bit-identical numbers across them
  mean a deterministic logic fault, which is far cheaper to chase than a race.

Generate past the sliding window if the model has one: the handover from prompt
tokens to generated ones inside the window is where eviction and checkpoint rotation
actually run.

## Choosing a threshold

Once every kernel on the path is batch-invariant, prefill and decode agree **bit for
bit** and the honest assertion is a stray-ulp floor, not a tolerance:

```python
KL_DIV_THRESHOLD = 1e-9   # measured 0; anything a state bug produces is orders above
```

A loose threshold tolerates float noise and small logic errors alike, which is how a
state-reuse bug hides. Prefer running the KL case on its own deterministic server and
asserting near-zero, and keep the accuracy case on the production numerics -- one
server cannot serve both.

Thresholds are per `(model, tp)`. A value calibrated at tp=1 does not transfer: tp=1
has no all-reduce, so it never exercises the source that dominates at tp>1.

## Localizing a divergence

Ablations answer "does it change" but never "where". The forward-hook dumper points
at the operator directly, and has done so reliably: run it once and read off the
first layer whose output differs while its inputs are bit-identical.

```bash
DUMPER_ENABLE=0 DUMPER_SERVER_PORT=reuse DUMPER_NON_INTRUSIVE_MODE=all \
DUMPER_DIR=/path/to/dumps python3 -m sglang.launch_server ... \
  --disable-cuda-graph --disable-prefill-cuda-graph
curl -X POST localhost:PORT/dumper/configure -d '{"enable": true, "exp_name": "dec"}'
```

Five settings that are each required, and each fails silently if wrong:

- `DUMPER_ENABLE=0` **plus** `DUMPER_SERVER_PORT=reuse`. The port sentinel makes
  `may_enable` true so the hooks register, while `enable=0` keeps warmup from
  dumping. Enabling at boot dumps every warmup prefill -- that is how a run wrote
  1.8T and filled a shared disk. Add a watchdog that kills the run below a free-space
  floor.
- `DUMPER_NON_INTRUSIVE_MODE=all`. The default `core` writes only `positions`,
  `seq_lens`, `req_pool_indices`, `input_ids`, `rids` -- no module tensors, and no
  error to tell you.
- `DUMPER_SERVER_PORT=reuse` is a literal sentinel, not a port number; the
  `/dumper/{method}` route only registers for that exact value.
- `--disable-prefill-cuda-graph` on top of `--disable-cuda-graph`. Some models
  default prefill onto a CUDA graph, and Python forward hooks do not run inside a
  replay -- the prefill pass then dumps the embedding and nothing else.
- Prefer `dumper.py` over `--debug-tensor-dump-*`: the latter asserts on a top-level
  module named `model`, which multimodal wrappers do not have.

**Prove the alignment before reading any diff.** Decode pass `k` and prefill row
`plen + k` consume the same token, so the embedding output must be bit-identical. If
it is not, the rows are misaligned and every downstream number is meaningless.
Getting this wrong once produced a confident, entirely wrong root cause.

Read the result as: the first layer where a module's **inputs are bit-identical and
its output is not** is the operator. Everything after it inherits.

## When the divergence needs a CUDA graph

A divergence that only appears with a captured graph defeats both usual probes, and the
failure is silent in each case:

- The **dumper's hooks do not run during replay** — the graph replays kernels, not Python.
  Disabling the graph to collect a dump also removes the divergence, so a clean layer-by-layer
  diff means nothing. Confirm the bug still reproduces under the exact flags you dump with.
- **Anything that syncs to host dies during capture** (`.item()`, `float()`, `.tolist()`).
  Guard probes with `torch.cuda.is_current_stream_capturing()` or the server will not boot.
- **The Python wrapper around a captured kernel is not called at replay.** Instrumenting it
  logs only the phases that stayed eager. Read that as evidence, not as a broken probe: it
  means the kernel runs with the arguments bound at capture, so any tensor handed in fresh
  per replay is invisible to it — a bug shape in its own right.

What works instead is to probe **the state that gets reused**, outside the graph: at the
moment a request donates its checkpoint, log the slot id, the length it claims to have
checkpointed at, and `abs().max()` over the stored state. Run it twice with the graph on and
off and diff per slot. A handful of slots whose content differs, with claimed lengths matching
the prefixes of the requests that go wrong, localizes the write in one round — where a dozen
ablations only bound the trigger.

**Make the probe prove it fired.** A probe on a code path that is not taken prints nothing,
which is indistinguishable from "measured, no difference". Assert a minimum hit count, or log
unconditionally at entry. Instrument the single choke point every caller reaches rather than
one call site.

## Confirm the mechanism, do not infer it

Two failure modes cost the most time, both avoidable:

- **A flag that changes nothing.** Bit-identical results before and after a toggle
  mean the flag did not take effect -- a dispatch guarded on a hidden condition, a
  path never taken for that config. Check the guard before concluding the component
  is innocent.
- **A harness that measures something else.** Capture through the helper's own
  functions rather than reconstructing its inputs. Reconstructing them once appended
  a generation twice and produced a plausible, wrong conclusion; another time a
  different `num_samples` silently selected a different prompt set through the
  `get_input_ids` cache key.

The logprob arrays are indexed by absolute position: with `logprob_start_len=0`,
`input_token_logprobs` carries one entry per input token, the first is `None`, and
entry `k` scores `input_ids[k]`. The helpers slice the tail, which lands on the
generated span; analysis that indexes absolutely has to agree with that. An
off-by-one here reads a neighbouring token, whose logprob is usually close enough to
look like a real signal.

For an isolated claim, reduce to a standalone repro. A ten-line script calling the
suspect op at M=1 and M=288 settles batch-invariance in seconds, and belongs in the
PR ahead of any end-to-end number.

Reading code to find a suspect is the slowest of these. One investigation refuted eight
successive code-derived hypotheses, each internally consistent, before a direct measurement of
the reused state found the defect in a single round. Prefer, in order: a single-variable A/B
that isolates the trigger, asking what the wrong output is the *correct* answer to, probing the
reused state itself, and only then reading for a mechanism to explain what was measured.
