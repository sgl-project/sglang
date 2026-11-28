# Author-run validation notes

The runtime and registered tests were introduced in commit `3fbfeed`, against
upstream base `440086bdd8484df273ddab6134b5586ec3993ec1`. Subsequent commits
only update documentation. These notes retain experiment conditions and
provenance separately from the PR's implementation and result summary.

## Models and reproduction

All runs use NVIDIA H20, CUDA Triton GDN, BF16 recurrent state, `extra_buffer`,
eager execution and unified serving. See [README](README.md) for launch commands,
registered tests and supported configurations.

The public `Qwen/Qwen3.5-4B` model is pinned at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Both weight files match the
SHA256 values in the official model metadata:

| Weight file | Bytes | SHA256 |
| --- | --- | --- |
| `model.safetensors-00001-of-00002.safetensors` | 5,329,398,688 | `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` |
| `model.safetensors-00002-of-00002.safetensors` | 3,990,429,408 | `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |

The registered TP1/TP2 model-continuation test passed on this public model and
on Qwen3.6-27B. Its actual capacities are 32,768 Full KV tokens and 64 state
slots, with 4,096-token chunks, 64-token pages, Level 2 checkpoints and path
cap 1. The 6,049-token producer executes as 4,096 + 1,953 tokens; its continuation
adds a fixed 64-token suffix. The branch and continuation probes rebuild their
initial trees independently. Complete-state layer checks use independent prefix
forwards; model-level suffix logprob comparisons use native-restored, cold and
repeated-cold controls.

## Additional serving lifecycle checks

The final runtime was also exercised on Qwen3.6-27B TP2 with 65,536 actual KV
tokens, 32,768 context, and 64- or 8-slot state pools. Strict busy/idle pool,
tree and request checks were enabled. These supplemental serving runs are
distinct from the registered allocator and ownership unit tests.

There are 486 request results across the runs: 40 lifecycle and 9 generated-history
requests in each 64-slot arm, plus 97 extended requests in each native/enabled
64/8-slot arm. They include 12 cancellations before the first output, 6 Decode
cancellations and 80 independent checkpoint-producing pressure requests per
extended arm. A pre-output abort may omit ordinary token counters; absent
reported counters are left absent rather than treated as computed tokens.

Both candidates in the mixed-row check enter one actual prefill batch:

| State slots / mode | Continuation cached tokens, rows 0 / 1 | Optional branch snapshots |
| --- | --- | --- |
| 64 / native | 960 / 960 | 0 |
| 64 / enabled | 1,024 / 960 | 1 |
| 8 / native | 960 / 960 | 0 |
| 8 / enabled | 960 / 960 | 0 |

Inputs and non-cancelled greedy outputs agree between paired arms. No strict
invariant error was recorded, and running-request/token/state usage gauges are
zero after flush. Post-cancellation and post-pressure recovery requests pass.
With actual generated assistant history, the 32-token continuation resumes at
5,952 native versus 6,016 enabled; 128- and 256-token continuations both resume
at 6,144 in either arm.

## GSM8K non-regression control

The full 1,319-question split uses the repository's `sgl-eval==0.1.3` driver,
official data preparation transform and grader. The downloaded official test
JSONL has SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`;
the prepared JSONL has SHA256
`67e9046d913470477dfb321814d4e3240240a397cfc19c6c610af02dd2cb6456`.
Only data loading is redirected to the fixed local copy.

Both Qwen3.6-27B TP2 arms use 65,536 actual KV tokens, 128 state slots, eight
client threads, greedy generation, seed 427, disabled thinking and a 2,048-token
output cap. Smoke requests are flushed before evaluating the full split.

Each arm scores 1,286 / 1,319 (97.4981%), with five output-cap truncations,
four unextracted answers and zero request errors. All paired generated texts,
answer labels, token counts and finish reasons agree. Neither arm records an
optional capture, so this result is a non-regression control. Activated-state
restoration is tested separately by the registered boundary and state tests.

## Paired public trace measurements

The pinned AgentX corpus and deterministic hash-block reconstruction are
described in [README](README.md) and implemented in [replay.py](replay.py).
These Qwen3.6-27B TP2 arms use 524,288 actual KV tokens, 128 state slots,
65,536 context, 4,096-token chunks and 64-token pages. Each measured request
generates one token. Startup and warmup are excluded; cache contents are
isolated as specified by the replay script.

Each directed arm measures four setup/producer/continuation triples and four
cold controls. Native cached tokens are 153,856 and enabled cached tokens
155,840 in every round; the four continuations save 128, 960, 256 and 640
prefill tokens respectively. The percentages below compare sums of client
TTFT within each request group, enabled relative to native.

| Round and launch order | Producers | Continuations | Producer + continuation | Complete triples | Cold controls |
| --- | --- | --- | --- | --- | --- |
| 1: native, enabled | -0.0424% | -23.7718% | -1.7661% | -0.8717% | -0.0066% |
| 2: enabled, native | -0.0128% | -23.5290% | -1.7160% | -0.8726% | +0.0063% |
| 3: native, enabled | -0.0893% | -23.8702% | -1.8167% | -0.9982% | -0.0423% |

The fixed unselected pair has 96 requests / 3,671,488 prompt tokens in each arm.
Both reuse 1,434,112 tokens (39.0608%) and have zero optional captures; every
paired cached length and output ID agrees. TTFT sums are 733.602 / 733.399 s
and P50 values 1.6834 / 1.6823 s, native/enabled. This is a no-trigger control;
it does not measure the cost or benefit of active capture under concurrency.

Across the six directed arms and two unselected arms, all 288 measured requests
complete. Input fingerprints, request order and greedy output IDs match within
each pair. The replay is sequential and compressed, rather than timestamp-paced
production traffic. Representative candidate rate, capture rate and additional
tail reuse remain open measurements.

## Environment and review scope

The installed OpenAI SDK logs a nonfatal import warning that disables the
optional `/v1/responses` endpoint. Timed replays use `/generate`; GSM8K uses
`/v1/chat/completions`. Those requests complete without errors. Compatibility
of `/v1/responses` was not exercised.

The runtime change spans eight files, with 168 added and six removed lines.
Registered tests contribute 675 lines and the replay script 185 lines;
documentation is separate. KDA prototype provenance and its kernel-only
validation boundaries remain in [backend-evidence.md](backend-evidence.md).
