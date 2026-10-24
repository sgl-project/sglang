# Resume an active hybrid request in a fresh process

This opt-in example preserves an active request's Attention KV, Mamba conv and
temporal state, generated token IDs, selected-token logprobs, and stream progress.
The target imports the committed state into newly allocated native slots instead
of prefilling the saved history again.

## Supported scope

Use identical model weights, weight version, source build, TP size, dtype, page
size, and layer geometry. The GPU qualification used Qwen3.5-35B-A3B, TP2, on the
same two A100 80GB devices, sequentially. The target may have a different local
cache capacity; the request's state geometry must remain compatible and its
allocations must fit. All TP ranks must have committed payloads before release.

Snapshot files and the HTTP control endpoint must be trusted. The snapshot
path must be visible at the same absolute path to both engines. This is not
cross-host recovery, weight-switch preservation, crash-atomic storage, or an HA
service. Sampling parameters and token history are retained; exact sampled-RNG
trajectory equivalence is not established. The measured reference uses greedy
generation. Do not edit a snapshot's identity to bypass a rejection.

## Prepare a public input

From the patched checkout, install SGLang's declared dependencies. Set MODEL to
a local model path or an available public model identifier. This sample input
is public and is not the private prompt used for the recorded GPU qualification.

```bash
export MODEL=Qwen/Qwen3.5-35B-A3B
export RESULT="$PWD/token-kv-example-result"
export SHARED_SNAPSHOT_DIR="$RESULT/snapshots"
export PROMPT="$RESULT/prompt.txt"
export SOURCE_URL=http://127.0.0.1:30000
export TARGET_URL=http://127.0.0.1:30000
mkdir -p "$SHARED_SNAPSHOT_DIR"
cat > "$PROMPT" <<'PROMPT_EOF'
Explain how plants convert sunlight into chemical energy. Describe the main
stages in detail, including what happens in the light-dependent reactions and
the Calvin cycle. Write a connected explanation of approximately 300 words.
PROMPT_EOF
```

## Source: reference, committed export, then exit

Start the source in its own terminal. The GPU pair must be available to this
example. Preserve radix caching and overlap; do not enable speculative decoding,
PD, PP, CP, DP, ReplaySSM, or additional unsupported side-state pools.

```bash
SGLANG_ENABLE_TOKEN_KV_SNAPSHOT=1 CUDA_VISIBLE_DEVICES=0,1 \
  python -m sglang.launch_server --model-path "$MODEL" --tp-size 2 \
  --mem-fraction-static 0.60 --weight-version snapshot-example-v1 --port 30000
```

After the engine is ready, run the public clients in another terminal:

```bash
python examples/runtime/token_kv_restore_reference.py capture \
  --public-helper examples/runtime/token_kv_restore.py \
  --engine-url "$SOURCE_URL" --prompt-file "$PROMPT" --max-new-tokens 128 \
  --out "$RESULT/reference" --timeout 300
python examples/runtime/token_kv_restore.py export \
  --engine-url "$SOURCE_URL" --prompt-file "$PROMPT" \
  --storage-dir "$SHARED_SNAPSHOT_DIR" --interrupt-at 8 --max-new-tokens 128 \
  --temperature 0 --out "$RESULT/source" --timeout 300
```

Require all TP ranks COMMITTED, retained output history, and successful source
private-slot release receipts. The CLI leaves the source paused; it does not
start or stop engines. Stop only this owned source using its own terminal or
process supervisor, and verify its GPU processes have exited before starting
the target. Slot release alone is not proof that an entire GPU is free.

## Target: new process, resume, compare

Start a fresh target on the same devices and build. Keep every compatibility
setting unchanged except the local memory fraction:

```bash
SGLANG_ENABLE_TOKEN_KV_SNAPSHOT=1 CUDA_VISIBLE_DEVICES=0,1 \
  python -m sglang.launch_server --model-path "$MODEL" --tp-size 2 \
  --mem-fraction-static 0.65 --weight-version snapshot-example-v1 --port 30000
```

Once the new engine is ready:

```bash
python examples/runtime/token_kv_restore.py resume \
  --source-receipt "$RESULT/source/source-checkpoint.json" \
  --engine-url "$TARGET_URL" --out "$RESULT/target" --timeout 300
python examples/runtime/token_kv_restore_reference.py compare \
  --public-helper examples/runtime/token_kv_restore.py \
  --reference "$RESULT/reference/reference.json" \
  --source-receipt "$RESULT/source/source-checkpoint.json" \
  --restore-result "$RESULT/target/restore-result.json" --out "$RESULT/comparison"
```

Inspect `correctness.json`, the restore result, and both ranks' completed-forward
observations. Require exact token IDs and logprobs, complete TP coverage, actual
new allocations, matching state digests, and zero recomputation of committed
history. Missing observations are unknown, not zero. Compare each rank's actual
local capacity rather than inferring capacity from the requested flags. Do not
sum replicated TP history counts or claim the whole workload had no prefill.
Stop the owned target after checking results.

## CPU checks

The two registered test files exercise immutable storage, incomplete/corrupt
records, compatibility rejection, native request reconstruction, batch updates,
and completed-forward accounting without loading a model:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python   python test/registered/unit/mem_cache/test_token_kv_snapshot_store.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python   python test/registered/unit/managers/test_token_kv_snapshot.py
```

Use a prepared Linux SGLang environment for full module imports. A macOS
installation without Triton cannot run those complete import paths. Historical
GPU validation is tied to its recorded source revision; these parameterized
commands and the public input have not been independently GPU-qualified on the
refreshed PR base.
