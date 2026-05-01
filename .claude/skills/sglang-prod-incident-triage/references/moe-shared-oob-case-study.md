# Example: Upstream Top-K Corruption, Downstream Shared-Memory OOB

Use this case when you want a replay-first CUDA failure that:

- crashes in a downstream MoE align kernel
- looks like shared-memory OOB or illegal address
- actually starts in the previous routing kernel
- is much easier to understand through dump plus replay than through ad-hoc prompts

The value of this case is simple: the visible crash lands in the consumer
kernel, but the real bug was injected one kernel earlier.

## Target Path

Use a Qwen3 MoE model that goes through `fused_topk` rather than grouped top-k:

- model: `Qwen/Qwen3-30B-A3B`
- `num_experts=128`
- `num_experts_per_tok=8`
- `use_grouped_topk=False`

Important call chain:

1. `python/sglang/srt/models/qwen3_moe.py`
2. `python/sglang/srt/layers/moe/topk.py`
3. `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu`
4. `sgl-kernel/csrc/moe/moe_align_kernel.cu`

For this model shape, `topk_softmax` dispatches to `topkGatingSoftmax`.

## Why It Works

The visible crash shows up in `moe_align_block_size_kernel`:

```cpp
int expert_id = topk_ids[i] + 1;
atomicAdd(&shared_counts[expert_id], 1);
```

If one `topk_ids[i]` is corrupted to a large positive value, the `atomicAdd`
writes far past `shared_counts[num_experts]`. `cuda-gdb` then points at the
align kernel even though the bad value came from the previous routing kernel.

That is exactly the behavior this skill needs:

- the request shape matters
- the crash dump preserves that request shape
- replay makes the failure stable
- the coredump names the failing kernel
- code reading still has to walk one step upstream

## Injection

Patch the producer, not the consumer.

File:

```bash
<sglang-root>/sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu
```

Patch site inside `topkGatingSoftmax`:

```cpp
if (thread_group_idx == 0) {
  const bool node_uses_expert = expert >= start_expert && expert < end_expert;
  const bool should_process_row = row_is_active && node_uses_expert;

  const int idx = k * thread_row + k_idx;
  output[idx] = max_val;
  indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;

  if (should_process_row && NUM_EXPERTS == 128 && k == 8 && num_rows == 769 &&
      thread_row == 17 && k_idx == 0) {
    indices[idx] = NUM_EXPERTS + 4096;
  }

  row_sum_for_renormalize += max_val;
}
```

Rules:

- do not modify `moe_align_block_size_kernel`
- corrupt exactly one slot
- use a clearly invalid large positive index
- keep the guard request-shape dependent
- do not add `printf`, `assert`, or extra logging

## Trigger

One trigger prompt was:

```text
"hello " * 768
```

On `Qwen/Qwen3-30B-A3B`, that tokenizes to `769` prompt tokens.

With the injected corruption:

```text
topk_ids[17, 0] = 4224
```

That is outside the valid expert-id range `[0, 127]`, but the visible crash
still happens later in `moe_align_block_size_kernel`.

## Replay-First Flow

### 1. Rebuild the serving kernel package

```bash
cd <sglang-root>/sgl-kernel
make build
```

Install the rebuilt package into the same serving environment used for replay.

Optional sanity check:

```text
row17_col0 4224
max_idx 4224
```

### 2. Start the bad build and collect a crash dump

```bash
export CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1>
export PYTHONPATH=<sglang-root>/python
cd <sglang-root>/python
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B \
  --tp 2 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --crash-dump-folder <crash-dump-folder>
```

Let serving traffic hit the server until the crash dump is written.

### 3. Summarize the dump

In one run the dump contained:

- one short warmup prompt
- one long trigger prompt: `"hello " * 768`

```bash
python3 scripts/incident_artifact_tool.py summarize-dump \
  --input-file <crash-dump-file>
```

### 4. Replay the captured request mix

If stock replay is blocked by `safe_pickle_load`, use:

```bash
python3 scripts/replay_trusted_request_dump.py \
  --input-file <crash-dump-file> \
  --host 127.0.0.1 \
  --port 32000 \
  --parallel 1
```

The visible client-side failure can still be generic:

```text
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

### 5. Restart with CUDA coredumps and replay again

```bash
export CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1>
export PYTHONPATH=<sglang-root>/python
export SGLANG_CUDA_COREDUMP=1
export SGLANG_CUDA_COREDUMP_DIR=<coredump-folder>
cd <sglang-root>/python
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B \
  --tp 2 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --crash-dump-folder <crash-dump-folder>
```

Inspect the coredump:

```bash
cuda-gdb "$(which python3)" \
  -ex "set pagination off" \
  -ex "target cudacore <coredump-file>" \
  -ex "where" \
  -ex "info cuda kernels" \
  -ex "x/12i <faulting-pc>"
```

Typical files:

- `<crash-dump-folder>/<worker-id>/crash_dump_<timestamp>.pkl`
- `<coredump-folder>/cuda_coredump_<host>.<pid>.<ts>`

Then switch to `debug-cuda-crash` or your narrower CUDA workflow.

## Expected Result

In one run, `cuda-gdb` reported:

```text
CUDA Exception: Warp Out-of-range Address
The exception was triggered at PC 0x7f7fe1dfac70  void moe_align_block_size_kernel<int>(...)
#0  0x00007f7fe1dfac00 in void moe_align_block_size_kernel<int>(...)<<<(2,1,1),(1024,1,1)>>> ()
```

The SASS near the faulting PC included:

```text
*> 0x7f7fe1dfac70 <...+624>: ATOMS.POPC.INC.32 RZ[R10+URZ+0x4]
   0x7f7fe1dfac80 <...+640>: @!P0 BRA 0x7f7fe1dfabf0
   0x7f7fe1dfac90 <...+656>: BSYNC B0
   0x7f7fe1dfaca0 <...+672>: BAR.SYNC.DEFER_BLOCKING 0x0
```

Expected progression:

1. the process crashes only for one narrow request shape
2. the dump preserves the exact request mix
3. replay reproduces the crash
4. `cuda-gdb` points at `moe_align_block_size_kernel`
5. one step upstream, `topkGatingSoftmax` is where the bad `topk_ids` value came from

Bottom line:

- failing kernel: `moe_align_block_size_kernel`
- root-cause kernel: `topkGatingSoftmax`

## What Not To Do

- do not patch `moe_align_block_size_kernel` first and call it fixed
- do not assume the faulting PC names the root cause
- do not switch to a grouped-topk model path
- do not skip replay

After demonstrating the bug, remove the injected corruption and rerun the same
replay. The crash should disappear without changing `moe_align_block_size_kernel`.
