# Experimental pi0.5 completion-boundary batching

This opt-in path preserves each request's original Euler grid and noise while
admitting compatible heterogeneous denoising budgets. At a dispatch boundary,
`L = min(remaining_steps)` steps execute; completed requests return actions and
unfinished requests retain their latent/prefix state and original FCFS age.
There is no dynamic bucketing or workload controller in this change.

## Run a hardware comparison

Use a Linux environment with the dependencies and checkpoint required by the
existing SGLang pi0.5 pipeline. Full-checkpoint CUDA/NPU and real TP validation
of this new integration are pending; CPU component tests are not a substitute.

Start a native server and a segmented server with the same checkpoint and
precision on separate device allocations. For a single-device configuration:

```bash
# Native server (allocate its device separately, e.g. CUDA_VISIBLE_DEVICES).
sglang serve --model-path MODEL_PATH --port 30000 --num-gpus 1 --sp-degree 1 \
  --batching-max-size 4 --batching-delay-ms 50

# Segmented server (a separate device/allocation).
sglang serve --model-path MODEL_PATH --port 30001 --num-gpus 1 --sp-degree 1 \
  --batching-max-size 4 --batching-delay-ms 50 \
  --pipeline-config-path examples/usage/pi05_segmented_actions/segmented.json
```

Use the same valid `/v1/actions/generations` JSON request on both servers. It
must contain fixed `input.observation.noise`, matching the checkpoint's action
horizon/dimensions, and the same images, state and prompt. The checker sends
native requests sequentially and segmented requests concurrently with budgets
3, 5, 4 and 8. Supply numerical tolerances appropriate to the backend/precision:

```bash
python examples/usage/pi05_segmented_actions/compare.py \
  --native-url http://localhost:30000 --segmented-url http://localhost:30001 \
  --request observation.json --atol ATOL --rtol RTOL
```

The checker rejects a run with no observed continuation. It checks complete
actions, original step counts, and action errors; it is not a benchmark. A
separate sustained-load test is needed to exercise repeated late-arrival refill.
For TP validation, repeat on the target topology with SP=1 and compare outputs
and terminal responses on every request, including failures.

## Scope and limits

- Disabled by default. Ordinary sampling signatures, warmup and whole-trajectory
  requests retain their existing path when disabled.
- Single observation/output per request, monolithic VLA stages, SP=1, no CFG
  parallelism or per-request action-expert offload. Stateful realtime sessions
  and grouped multi-output requests use the ordinary path.
- TP rank zero selects requests; workers verify queue identities and budgets.
  True multi-process/device behavior has not yet been validated.
- Prefix and action CUDA graphs are disabled for this opt-in path: graph-owned
  prefix outputs can be overwritten by later replays while a continuation still
  needs them. Fresh prefixes execute before denoising; there is no overlap.
- Prefix K/V collation uses padded, masked device copies; it is not zero-copy.
  Only tensor-compatible groups execute together.
- Logical requests remain in-flight in server metrics across continuation
  boundaries. Reported per-worker forward timing still describes individual
  dispatches; action payload timings include accumulated action denoising time.
- Defaults and hardware-specific thresholds from the research campaign are not
  installed into the upstream server. No performance claim is made here.

Related design discussion: https://github.com/sgl-project/sglang/issues/43522
