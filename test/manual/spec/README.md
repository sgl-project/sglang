# STANDALONE PD regressions

These manual tests exercise prefill and decode as separate processes on one
Hopper GPU. They compare greedy output token IDs with a monolithic target
reference, including a 700-token prompt spanning two prefill chunks.
They are not benchmarks or cross-host/RDMA coverage.

## Requirements

Use the normal SGLang CUDA test environment, a Hopper GPU with at least 70 GiB
of visible memory, FA3, NVML (`pynvml`), and Mooncake with CUDA transport support.
The GPU must be idle: the fixture requires every GPU compute process to leave
before starting the next group of serving processes. Run the commands serially.
Do not use a shared GPU with unrelated serving processes.

The default STANDALONE pair is `Qwen/Qwen3-4B` and `Qwen/Qwen3-0.6B`.
Target/draft tokenizer IDs and full vocabularies must match. To use downloaded
model directories, set `SGLANG_TEST_STANDALONE_PD_TARGET` and
`SGLANG_TEST_STANDALONE_PD_DRAFT`. Each `OUTPUT_DIR` below must not already exist.

From the repository root:

```bash
SGLANG_TEST_STANDALONE_PD_OUTPUT_DIR=/tmp/standalone-pd-normal \
  python test/manual/spec/test_spec_standalone_disaggregation.py -f

SGLANG_TEST_STANDALONE_PD_OUTPUT_DIR=/tmp/standalone-pd-recovery \
  python test/manual/spec/test_spec_standalone_pd_recovery.py -f

SGLANG_TEST_STANDALONE_PD_OUTPUT_DIR=/tmp/ordinary-pd \
  python test/manual/spec/test_spec_pd_shared_paths.py TestOrdinaryPD -f
```

The recovery test writes one invalid producer seed into the registered metadata
buffer. Transfer succeeds, decode rejects that request before model execution,
and two later good requests reuse the released capacity. It checks request,
metadata and KV allocation recovery, not arbitrary transport damage.

## EAGLE3 shared-path regression

`TestEagle3PD` needs an actual trained EAGLE3 checkpoint, not a vanilla small LLM.
Its default is `meta-llama/Llama-3.1-8B-Instruct` with
`lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`; arrange model access before running.
Alternatively, the tested ungated pair is `Qwen/Qwen3-8B` with
`AngelSlim/Qwen3-8B_eagle3` at revision
`9629dfce7a4a10564dd48d3e5485c3976095653c`. Set the two model variables to the
matching local target and pinned draft directories.

```bash
SGLANG_TEST_STANDALONE_PD_OUTPUT_DIR=/tmp/eagle3-pd \
  python test/manual/spec/test_spec_pd_shared_paths.py TestEagle3PD -f
```

The observer records real draft forwards and target verification, and checks
hidden-state and top-k hashes from producer metadata through decode admission.
Ordinary PD checks that speculative metadata remains empty and does not invoke
the speculative factory. All tests preserve HTTP results, observed process
identities, queue metrics and cleanup receipts. An inference POST is never
retried; inspect its terminal result before starting another run after failure.

## CPU contracts

The registered CPU suite discovers these files automatically. In its CPU test
environment, run:

```bash
python -m pytest -q \
  test/registered/unit/speculative/test_pd_standalone_draft_input.py \
  test/registered/unit/disaggregation/test_pd_standalone_metadata.py \
  test/registered/unit/disaggregation/test_prefill_abort_result_cleanup.py \
  test/registered/unit/disaggregation/test_pd_standalone_decode_cleanup.py \
  test/registered/unit/server_args/test_standalone_pd_scope.py
```

These cover seed serialization/dispatch, stale-buffer clearing, malformed-seed
aborts, cleanup and supported startup configurations. The overlap interface test
only records FutureMap publication; it does not validate GPU event scheduling
or enable overlap for STANDALONE PD.
