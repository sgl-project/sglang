# DeepSelect top-k

`vendor/` is [deepseek-ai/DeepSelect](https://github.com/deepseek-ai/DeepSelect)
(commit `382d62a`, plus the SM90 cluster variant from upstream PR #14), stripped
to the kernel implementation. It also carries the padding, `-inf` tie, and
int64 offset correctness fixes from SGLang's former AOT implementation. MIT
licensed; see `vendor/LICENSE`.

`entry.cuh` is the SGLang side: tensor validation, the `TopkSelectArgs` block,
and the two host entry points the JIT exports. `python/sglang/kernels/ops/deep_select.py`
is the wrapper that compiles and calls them.

## What is vendored

| Path under `vendor/` | What it is |
| --- | --- |
| `structs.h` | `TopkSelectArgs` -- the launcher's plain-C argument block, plus the stride-alignment constants |
| `cuda_kernels/config.h` | `TopkSelectConfig<...>` -- every compile-time knob of a kernel instance |
| `cuda_kernels/{utils,bit_utils}.cuh` | small device helpers |
| `cuda_kernels/common_parts.cuh` | the shared kernel body (TMA pipeline, radix pass, epilogue) |
| `cuda_kernels/v3/topk_select.cuh` | bf16, one CTA per row |
| `cuda_kernels/v3_cluster/topk_select.cuh` | bf16, one cluster per row (8 CTAs on SM90, 16 on SM100/SM103) |
| `cuda_kernels/v3_fp32/topk_select.cuh` | fp32 |
| `3rdparty/kerutils/` | upstream's header-only helper library, subset |

Each `topk_select.cuh` defines one kernel plus its launcher,
`run_topk_select_kernel<Config>(const TopkSelectArgs&)`, in its own namespace
(`topk_select_bf16_normal`, `topk_select_bf16_cluster`, `topk_select_fp32`).

Dropped from the vendored tree, because a JIT build does not need them:

- `cuda_kernels/*/instantiations/**` -- 86 explicit instantiation `.cu` files.
  They exist so an AOT build can compile every configuration into one shared
  object. `entry.cuh` instantiates the one or two a module can reach.
- `api.cpp` -- the torch/pybind entry point and the runtime dispatch that
  picked a configuration from `(dtype, topk, vocab_size, batch_size, arch)`.
  That decision is now split between the Python module factory and `entry.cuh`.
- `dispatch_utils.h` -- `BOOL_SWITCH` / `INTEGER_TYPE_SWITCH`, which include
  `torch/extension.h` and exist only to expand that dispatch.
- `cuda_kernels/*/topk_select.h` -- forward declarations of
  `run_topk_select_kernel`, needed only to split the instantiations into
  separate translation units.

## Build inputs

Include paths: `vendor/` (the `#include "cuda_kernels/..."` lines resolve
against it) and `vendor/3rdparty/kerutils/include`. CUTLASS supplies
`cute/arch/copy_sm90_tma.hpp` and `cutlass/arch/barrier.h`, and is a registered
JIT dependency (`extra_dependencies=["cutlass"]`).

Beyond the JIT defaults the kernels need `--expt-extended-lambda` and
`--ftz=false`; the latter is load-bearing and must follow `--use_fast_math`,
because the radix pass simulates integer counters with fp32 addition and needs
subnormals.

## Re-syncing with upstream

Copy the files listed above into `vendor/`, remove the
`#include "topk_select.h"` line at the top of each `topk_select.cuh`, then
reapply the SGLang AOT correctness patch described above. `vendor/.clang-format`
disables formatting so a re-sync stays reviewable.
