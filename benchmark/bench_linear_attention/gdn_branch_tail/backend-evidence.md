# Backend mechanisms and evidence boundaries

Hybrid prefix reuse requires both Full KV and a recoverable Linear state at the
selected token boundary. The Linear snapshot must include convolution history
and recurrent state at that same depth. An extend with a single tracking
position can select a branch instead of its aligned tail; an independent
snapshot lets the two reuse opportunities coexist without another forward.
This describes the constraint of the affected tracking path, rather than a
claim about every Linear backend or every checkpoint configuration.

## Mechanism comparison

| Question | GDN in this patch | Separate KDA prototype |
| --- | --- | --- |
| Intermediate recurrent state | Copy from native Triton `h` into a spare state slot; validated BF16 state | Sparse FP32 export from the recurrent kernel |
| Convolution history | Capture the pre-convolution input window at the branch depth | Gather the raw input window, respecting the persistent KDA layout |
| Branch and tail in one forward | Additional branch capture alongside native tail tracking | Sparse export alongside native tracking in the same kernel call |
| State validity | Complete branch and tail states checked against independent prefix forwards | Recurrent export and convolution history checked in separate real CUDA tests |
| Resume correctness | Actual cache-tree publication and model continuation, cold/native logprob controls, TP1 and TP2 | Kernel suffix outputs and final recurrent state compared with the uninterrupted reference; convolution suffix checked separately |
| Ownership and fallback | Real cache allocators, duplicate publication, invalid-forward cleanup and collective no-spare fallback | Prototype control-flow checks are separate from numerical evidence |
| Support commitment | The opt-in GDN configurations listed in [README](README.md) | Mechanism evidence only; not enabled by this patch |

## Existing KDA result index

The separate prototype was based on engine revision
`27ff1f3ff2807e4781df79f7a482c5e30bc474c3`. Its candidate kernels were tested
on NVIDIA H20 using PyTorch `2.13.0+cu130`, Triton `3.7.1` and CUDA `13.0` with
an available `0.5.19` image's dependencies. These were standalone candidate
kernel tests, not a full engine compatibility test for that image.

The prototype test module is
`test/registered/attention/test_mamba_branch_tail_multisnapshot.py`
(SHA256 `ec8aa4d632dda13c4ff75056f845656ab5c01e8b3831cdb16854e5fb4e6ec17c`).
The recorded CUDA gate passed all three tests with no failures, errors or skips;
two tests are KDA evidence and the third is the prototype GDN test. The separate
13 + 20 control-flow checks are not counted as state-restoration evidence.

| KDA test | Inputs and boundaries | Recorded acceptance criterion |
| --- | --- | --- |
| `KDAMultiSnapshotCUDATests.test_sparse_snapshots_preserve_forward_and_restore_suffix` | Packed lengths 260 and 192; head dimensions 64 and 128; two heads; BF16 activations, FP32 recurrent pool with noncontiguous row strides; first-row snapshots at 64 and 256, native tracking retained | Forward output and live final state are bitwise equal with export off/on. Exported states match independent prefix forwards (`rtol=2e-3`, `atol=2e-4`); suffix outputs and final states match the uninterrupted reference (`rtol=3e-3`, `atol=1e-3`). These are asserted tolerances, not reported maximum errors. |
| `KDAMultiSnapshotCUDATests.test_raw_conv_window_is_exact_prefix_state` | Length 260, 128 channels, convolution width 4; boundaries 64, 128 and 256 | Gathered convolution history equals the independent prefix state exactly; resumed suffix convolution output is bitwise equal. |

Recorded candidate SHA256 values identify the tested prototype rather than
whatever revision its working copy contains later:

| Candidate source | SHA256 |
| --- | --- |
| `python/sglang/kernels/ops/attention/fla/kda.py` | `08fcb090aa214d76f21ea8b2e06df486b4cdc8d7266ec8fc95453f988cf379be` |
| `python/sglang/kernels/ops/attention/fla/chunk.py` | `5315fb56b4320642b2c1454a9ce4dbb25075e281e19292c455a35033dab27398` |
| `python/sglang/kernels/ops/attention/fla/chunk_delta_h.py` | `ed37ab8d1ac9ac1be75a1348ef706b147f8f36c7612c54e95a6cde192626058c` |
| `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py` | `e5bf5095b98f3407b200fdb7e74ac511a7caa7b1f15b798250ea365dcb41c2a9` |
| `python/sglang/srt/mem_cache/mamba_branch_and_tail.py` | `470ab3ce96d42e04caf8acbbef5c5fa472af67f7f5549715901c0eed4445caf2` |

The KDA prototype and its tests are not part of this patch. This index records
existing author-run evidence; it is not an independently runnable KDA test
target in this branch. It does not establish KDA model-level cache restoration,
TP serving, PD transfer, cancellation safety or workload performance.
Fake-bootstrap tests that reject publication provide no evidence of numerical
state restoration or successful cache-tree reuse.

## Extension path

The capture-and-publication pattern can be adapted to other Linear backends.
Each adapter must expose a complete state at a supported boundary, preserve
native live state and tail tracking, and transfer its independently owned slot
only after the forward has completed successfully. Tile alignment, state dtype,
convolution layout and capture mechanism remain backend-specific.

The KDA prototype supports the feasibility of sparse intermediate-state capture
and suffix continuation. A future KDA integration still needs the model-level
cache, ownership, fallback and distributed tests exercised by the GDN path.
The current architecture and backend guards remain in place until those
contracts have been implemented and validated.
