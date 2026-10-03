# Replicated KV Head Capture

This lane uses the real Qwen2.5-1.5B-Instruct target, with 12 query heads,
two KV heads and 128 dimensions per head. At TP4, ranks 0/1 contain logical
KV head 0 and ranks 2/3 contain head 1. The snapshot must contain two global
heads, with one payload owner per replica group. Four actual CUDA devices
are required; the test does not change the target's attention geometry.

## Ownership And Identity

The expected Store owners are `dp0-pp0-tp0` and `dp0-pp0-tp2`. Rank 0 also
owns token IDs, positions, response/loss masks and raw top-128 teacher data.
For every snapshot, the tests check each object's owner, head range and shape,
then read the complete tensors through the real Mooncake client. Existing
manifest validation rejects missing, overlapping or extra head coverage.

Test-only forward observations record each rank's actual capture partition and
allocated Host/device bytes. Ranks 1 and 3 must participate in execution and
capture coordination while allocating no payload buffers or exporting KV.
Ranks 0 and 2 must export exactly one head in every selected layer. This checks
the inactive-partition path in real serving, beyond a synthetic layout fixture.
The P/D observer also records the completed handoff: a one-token response can
finish without any D forward, and still requires every rank's ownership check.

The shared runtime fixture now uses its declared `model_id` for target loading,
capture configuration and manifest validation. Qwen2.5 must retain its actual
weight/tokenizer fingerprints, two-head geometry, unnormalized target K and
RoPE theta of 1,000,000. It is not labeled as the earlier Qwen3 target.

## Execution Matrix

The colocated suite runs AR and static target-KV DSpark with Full, Breakable
and torch.compile piecewise prefill graphs, plus an eager AR baseline. It
exercises chunked prefill, cached prefixes, padding, buffer reuse on every
rank, batched requests and speculative accept/reject paths. TP4 graph execution
uses overlap. Piecewise uses eager compile debug mode; Full prefill remains
experimental under the server's existing capability contract.

Each colocated configuration runs the same complete request schedule first
without `--training-capture-config`, then with capture enabled. The control
checks `/server_info` for a null capture configuration and no new Catalog
publications. Every generated token must match between the two runs. Draft
projection observations from the control are excluded from the capture run's
assertions. Cross-backend output differences remain visible in the reports.

The P/D suite runs matching TP4/PP1 endpoints with AR and static target-KV
DSpark, each in eager and decode graph modes. It also runs AR from TP2 prefill
to TP4 decode, where the source has split KV heads and the destination has
replicas. Both endpoints share the same four GPUs. Missing/stale teacher
handoffs and request cancellation must fail capture without partial READY
publication, while serving continues. P/D prefill graphs are disabled here.

Independent test observations read raw vocabulary logits before serving
processors and selected KV directly from live pools before source reuse.
The P/D observer also reads the source slots used for transfer. Exact Store
KV/top-128 values, token alignment and masks are checked; LSE uses the existing
numerical tolerance. Final Store reads run after producer process exit. These
checks establish selection, ownership and transfer correctness; they do not
independently prove the attention backend's writes into its own KV pool.

## Synthetic Draft

The DSpark fixture copies two target backbone layers and retains Qwen2 QKV
biases. Its draft architecture requires Q/K normalization, so the synthetic
checkpoint explicitly adds unit norm weights and zero output-projection bias.
The target's own architecture and KV values remain unchanged. Random encoder
and Markov weights, including the forced proposal used to exercise acceptance,
are test fixtures and do not establish trained draft quality.

The independent draft projection observer now includes the actual local QKV
bias in its mathematical reference. Strict production checkpoint loading
still validates every parameter and shard; this fixture does not weaken that
validation or change the production export format.
On every rank, another observer compares the encoder's selected global KV
heads with that rank's actual source pool rows. Thus both copies of a logical
head must agree with the selected input; shape checks alone cannot accept
duplicating head 0 in place of head 1. These observations run only in tests.

## Reproduction

Use the environment in `h100-runtime-lock.json`, four H100s, a local immutable
Qwen2.5-1.5B-Instruct model and `mooncake_master` on the runtime `PATH`:

```bash
export TRAINING_CAPTURE_TEST_MODEL=/models/Qwen2.5-1.5B-Instruct
export PYTHONPATH=python
export OMP_NUM_THREADS=1
python test/registered/storage/test_training_capture_replicated_prefill.py -v -f
python test/registered/storage/test_training_capture_pd_replicated.py -v -f
```

The Store and P/D transports use TCP; Catalog is the HTTP test implementation.
This lane does not certify cross-node RDMA, production retention, trained
draft acceptance, saturated traffic or serving SLOs. Broader model identities
and topology combinations require their own deployment validation.

## P/D Results

The frozen v3 fixture passes all six P/D methods in 815.856 seconds. Matching
TP4 AR/DSpark eager/graph and TP2-to-TP4 AR eager/graph validate 32 post-exit
snapshots, 928 tensor objects and 6,419,524 tensor bytes. All snapshots share
the same Qwen2.5 weight, tokenizer and teacher fingerprints across these
topologies. Only ranks 0 and 2 publish payload objects; the observed Host/device
payload allocation stays zero on ranks 1 and 3.

An initial test failed because it required D forward observations even for a
one-token response. The corrected observer checks the actual first-teacher
handoff and source pool rows, retaining all rank-level assertions. The initial
failure is test instrumentation, not a dropped or partially published sample.

## Colocated Results

The final v5 fixture passes all six methods in 853.475 seconds. Including the
eager baseline, seven configurations validate 70 post-exit snapshots, 2,408
tensor objects and 24,393,956 tensor bytes. Their 70 capture-off requests have
exactly the same output token IDs as the corresponding capture-on requests.
Each of the six graph configurations observes 44 prefill replay rank-frames,
264 in total, with padding and buffer reuse checked on every rank.

AR and DSpark piecewise both retain an unbiased-request output difference
against the eager baseline. All per-configuration capture-on/off comparisons
pass; these results do not claim cross-backend greedy equivalence. Together
with P/D, 102 Qwen2.5 snapshots validate 3,336 tensor objects and 30,813,480
tensor bytes under one teacher/weight/tokenizer identity.

## Cross-Backend Diagnostic

The initial colocated suite passed eager, Breakable and Full AR, then failed
the piecewise-to-eager greedy comparison for the unbiased rejection request.
Eager returned `[78, 11, 220, 17, 15, 16]`; piecewise returned
`[78, 11, 220, 16, 24, 24]`. A separate plain-server experiment, with capture
entirely disabled, reproduces both sequences using the same request schedule.
This rules out capture as a necessary cause of that observed difference; it
does not identify the numerical operation responsible or prove cross-backend
greedy equivalence. `diagnose_replicated_prefill.py` preserves the reproducer.

An initial diagnostic attempt read a nonexistent nested `server_args` field
and failed before producing comparison evidence. Its script was edited after
the interpreter had loaded the old version. The early directories shared a
hard link, so the original failed script was not preserved; its traceback is
retained and the archived corrected file is not claimed as its executed source.
Only the separate v4 diagnostic, using the actual flat
`training_capture_config` field, supplies the successful comparison. The final
colocated suite uses the per-configuration capture-off gate described above;
all KV, logits, graph and ownership assertions remain in force.

## Regression And Evidence

The final source also passes the existing Qwen3 colocated DSpark Full/overlap
test in 109.934 seconds and P/D AR Full/overlap test in 215.220 seconds. They
validate another 19 and 20 post-exit snapshots. Across the four final suites,
14 test methods and 141 snapshots pass. Preliminary Qwen3 successes and all
three failed attempts are retained separately from final acceptance.

The [machine-readable report](replicated-kv-capture.json) records every job
command/result, model identity, per-configuration counts and source hashes.
Its archive contains 82 artifacts under
`/gpfs/user/fuxuanwei/mooncake-lab-archive/replicated-kv-20261003`, with a hashed
inventory. All 3,359 Python source/storage test files match the final v5
checkout. P/D v3 differs only in the colocated prefill helper and test it does
not import; the corrected diagnostic is archived separately. Test cleanup
removed temporary reference tensors and projection JSONL files, so they are
not claimed as retained artifacts; the executed assertions and logs remain.

After all jobs completed, the queue and running directory were empty and
`nvidia-smi` reported no compute processes. The supervisor exited with code 0;
the temporary four-H100 job, pod and job resources were deleted. The original
resident H100 worker and its existing idle workload were verified live.
