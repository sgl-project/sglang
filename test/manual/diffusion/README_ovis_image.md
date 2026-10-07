# Ovis-Image manual validation

These commands compare native SGLang against an independent, pinned Diffusers
oracle. Run GPU commands only in a GPU allocation that already exists. For this
development task, the main session owns the `overflow` allocation and queues
every GPU case serially. The scripts do not call `sbatch`, `srun`, or request GPUs.

The default profile is **1024 × 1024, 50 steps, CFG 5, seed 42**, with an empty
negative prompt, text length 256, BF16, and a CPU random generator. `--quick`
selects 512 × 512 and four steps. A quick run is a smoke check; report it
separately from default-profile quality validation.

Ovis request dimensions must be integers of at least 16 and multiples of 16.
The `vae-tiled` and `vae-spatial` matrix cases always use **1024 × 1536**, including
under `--quick`; only their step count follows that profile. Native VAE tiling
starts when a dimension is greater than 1024. Enabling tiling at 512 × 512 or
1024 × 1024 alone does not exercise the tiled decode path. These larger cases
use a matching 1024 × 1536 reference and require more memory than a quick
512 × 512 run.

## Pin the source and environments

Use the full commit SHA of the Ovis implementation being reviewed, rather than
an advancing branch name. The original integration baseline is
`4ab720e655`; that baseline alone does not contain this implementation.

```bash
export NATIVE_REVISION='<full-commit-SHA-of-the-Ovis-PR>'
export VALIDATION_ROOT=/absolute/path/outside/the/checkout/ovis-validation
export SGLANG_SRC=/absolute/path/to/sglang
mkdir -p "$VALIDATION_ROOT"
# Fetch the reviewed PR/fork first so this repository contains NATIVE_REVISION.
git -C "$SGLANG_SRC" worktree add --detach "$VALIDATION_ROOT/native-src" "$NATIVE_REVISION"
export NATIVE_SRC="$VALIDATION_ROOT/native-src"
git -C "$NATIVE_SRC" rev-parse HEAD
git -C "$NATIVE_SRC" status --short
```

For reproduction from a published commit, use a clean checkout. The matrix
records the revision, source status, tracked diff, package freezes, and visible
GPU properties. A dirty checkout needs its untracked files archived separately;
`source.patch` contains tracked changes only.

The following is a fresh-install recipe, not a claim that installation into
empty virtual environments has been tested. The development environments reused
a provisioned Torch/CUDA stack through `--system-site-packages`; their complete
package metadata must accompany actual measurements. No successful full
`pip check` result is asserted here.

Create two virtual environments without inheriting site packages. Native uses
the checkout's pinned dependencies, including Diffusers **0.37.0**, Torch
**2.13.0**, and Transformers **5.17.0**. The full-model oracle uses Diffusers
commit **c6df88a511a98740646ee55577b590c9852650ce** (`0.41.0.dev0`), in its own
environment. Do not install SGLang into the reference environment.

```bash
python3.12 -m venv "$VALIDATION_ROOT/native-env"
python3.12 -m venv "$VALIDATION_ROOT/reference-env"
export NATIVE_PYTHON="$VALIDATION_ROOT/native-env/bin/python"
export REFERENCE_PYTHON="$VALIDATION_ROOT/reference-env/bin/python"

"$NATIVE_PYTHON" -m pip install 'pip==26.2.1' setuptools setuptools-scm build wheel
"$NATIVE_PYTHON" -m pip install 'torch==2.13.0' 'torchvision==0.28.0' \
  --index-url https://download.pytorch.org/whl/cu130
SGLANG_BUILD_RUST_EXTS=none "$NATIVE_PYTHON" -m pip install \
  --no-build-isolation -e "$NATIVE_SRC/python[diffusion]"
"$NATIVE_PYTHON" -m pip install 'pytest==9.1.1'

"$REFERENCE_PYTHON" -m pip install 'pip==26.2.1'
"$REFERENCE_PYTHON" -m pip install 'torch==2.13.0' 'torchvision==0.28.0' \
  --index-url https://download.pytorch.org/whl/cu130
"$REFERENCE_PYTHON" -m pip install \
  'transformers==5.17.0' 'accelerate==1.15.0' 'huggingface_hub==1.33.0' \
  'numpy==2.3.5' 'pillow==12.3.0' 'scikit-image==0.25.2' \
  'diffusers @ git+https://github.com/huggingface/diffusers.git@c6df88a511a98740646ee55577b590c9852650ce'

export PATH="$VALIDATION_ROOT/native-env/bin:$PATH"
export PYTHONPATH="$NATIVE_SRC/python"
"$NATIVE_PYTHON" -m pip freeze > "$VALIDATION_ROOT/native-freeze.txt"
"$REFERENCE_PYTHON" -m pip freeze > "$VALIDATION_ROOT/reference-freeze.txt"
```

Use a CUDA toolkit compatible with the selected Torch wheel; native JIT paths
need an executable `nvcc` under `CUDA_HOME/bin`. Ring cases need the native
FlashAttention backend and its supported GPU/kernel stack. They use
`--attention fa`; SDPA is used for the remaining full-model cases. A missing
kernel or unsupported GPU is a failed or unverified case, not numerical evidence.
The SDPA oracle excludes the CuDNN SDPA backend in a scoped Torch context,
matching the existing native CUDA platform policy. Both sides record their actual
SDPA flags. Selecting `sdpa` alone does not fix its underlying CUDA kernel:
the default independent environment can select CuDNN and produce different BF16
conditioning from native. This control changes the validation kernel policy;
the official model, checkpoint, and numerical tolerances are unchanged.
Use the native Python's `-m torch.distributed.run`, which avoids a `torchrun`
executable from a different environment. The native environment's `bin` directory
must also be on `PATH` so the HTTP fixture finds `sglang` and JIT builds find
`ninja`.

## Download only the Diffusers package

The public canonical model is `ATH-MaaS/Ovis-Image-7B`. Pin revision
**41be1c5821a92c970d63d7eb595a2fd3fe32b22e**. The allowlist excludes the standalone
single-file checkpoint and other repository content.

```bash
export MODEL_PATH="$VALIDATION_ROOT/models/Ovis-Image-7B-Diffusers"
"$REFERENCE_PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ATH-MaaS/Ovis-Image-7B",
    revision="41be1c5821a92c970d63d7eb595a2fd3fe32b22e",
    local_dir=os.environ["MODEL_PATH"],
    allow_patterns=[
        "model_index.json", "scheduler/**", "tokenizer/**",
        "text_encoder/**", "transformer/**", "vae/**",
    ],
)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Keep the model revision/download manifest with the results. The runners use
`local_files_only=True`; the matrix validates that all five component directories
and `model_index.json` exist. It does not prove that an arbitrary local directory
matches the pinned revision.

## Run one oracle comparison

From the native checkout, in the already assigned GPU environment:

```bash
cd "$NATIVE_SRC"
export RESULTS="$VALIDATION_ROOT/results"

"$REFERENCE_PYTHON" test/manual/diffusion/validate_ovis_image.py \
  --mode reference --model-path "$MODEL_PATH" \
  --output "$RESULTS/reference-full" --height 1024 --width 1024 --steps 50

"$NATIVE_PYTHON" -m torch.distributed.run --standalone --nproc_per_node 1 \
  test/manual/diffusion/validate_ovis_image.py \
  --mode native --model-path "$MODEL_PATH" \
  --output "$RESULTS/native-full" --height 1024 --width 1024 --steps 50 \
  --attention torch_sdpa

CUDA_VISIBLE_DEVICES= "$REFERENCE_PYTHON" test/manual/diffusion/validate_ovis_image.py \
  --mode compare --reference "$RESULTS/reference-full" --output "$RESULTS/native-full"
```

Reference runs are single-process. Native world size is
`tp × ulysses × ring × cfg`. The two runtimes receive the same CPU-generated
initial latents and save tensor records, output PNGs, configuration/provenance,
and timing/memory data. Comparison runs on CPU and writes `comparison.json`.
Keep raw tensors so conditioning, first-step predictions, scheduler trajectory,
and final-image errors can be inspected independently.
Comparison requires exact initial latents and effective timesteps, and checks
conditioning and first-step predictions with the component tolerances: FP32
`atol=rtol=1e-4`, BF16 `atol=0.05, rtol=0.02`. A numerical mismatch returns
nonzero after writing the report. Later trajectory and image errors, PSNR and
SSIM remain review evidence; they do not have an invented acceptance threshold.
The report also rejects differing sampling inputs and model revisions. Distributed
native captures require matching source commits and hashes on every rank; a
native baseline comparison requires the same source identity on both sides.
Keep the checkout unchanged while collecting a matrix with baseline comparisons.

To check internal multiple-prompt ordering, pass the same
`--second-prompt 'A blue sailboat on a calm sea, watercolor painting.' --outputs 2`
to both runners. This exercises two prompts with two samples each, in
`[A, A, B, B]` order. It is an internal `Req` batch oracle and does not claim
that the OpenAI image API accepts a list of prompts.

## Serial GPU matrix

`run_ovis_image_matrix.sh` creates a unique results subdirectory per invocation,
runs selected cases in order, reuses matching reference outputs only within that
invocation, and records `status.tsv`. It continues after a failed case and exits
nonzero if any executed case failed. With no explicit case list, unavailable
GPU counts are recorded as skipped. Selecting an unavailable case explicitly
also returns a nonzero exit status.
The tiny checker also accepts `--edge-cases` to exercise one text token and tiny
image sequences. A text prefix that remains replicated with Ring padding gathers
the full image KV; an image whose padding spans several shards runs replicated
SP attention while retaining TP. These correctness fallbacks are not Ring or SP
performance measurements. Padded Ring batches use the existing kernel one sample
at a time; ordinary batches keep their existing execution path.
Run `single` before the parallel and offload cases in the same invocation to
save a separate `single-card-comparison.json` against that hardware's native
baseline. `vae-spatial` also compares against a preceding `vae-tiled` case.
`comparison.json` always retains the official reference comparison.

```bash
# Command construction only: does not import Torch, use a GPU, or create results.
bash test/manual/diffusion/run_ovis_image_matrix.sh --dry-run --quick --max-gpus 4

# Queue this command through the main session's existing overflow allocation.
# No new allocation is requested by this script.
bash test/manual/diffusion/run_ovis_image_matrix.sh --quick --max-gpus 4

# Complete default-profile single-card oracle; omit --quick for 1024/50.
bash test/manual/diffusion/run_ovis_image_matrix.sh reference single

# Selected full-checkpoint parallelism cases, still executed serially.
bash test/manual/diffusion/run_ovis_image_matrix.sh --quick \
  single tp2 ulysses2 ring2 cfg2 tp2-sp2 prompt-batch
```

| Case | GPUs | Settings and evidence |
| --- | ---: | --- |
| `components` | 1 | Tiny DiT/VAE numerical oracles plus configuration/tokenizer contracts. |
| `reference`, `single` | 1 | Single-process Diffusers; native TP1/SP1, same scheduler and initial latents. |
| `no-cfg` | 1 | Guidance 1, independent reference for the single conditional branch. |
| `batch2` | 1 | One prompt, two samples; matching reference batch. |
| `prompt-batch` | 1 | Two internal prompts, two samples per prompt; matching four-sample reference. |
| `component` | 1 | Component CPU offload for transformer, text encoder, and VAE. |
| `layerwise` | 1 | Layerwise offload for transformer, text encoder, and VAE. |
| `tiny-tp2`, `tiny-ulysses2` | 2 | Checkpoint-free production DiT against a tiny oracle, including odd text/image lengths. |
| `tiny-ring2` | 2 | Tiny BF16 DiT with FlashAttention ring rotation. |
| `tiny-tp2-sp2` | 4 | Tiny DiT with TP2 × Ulysses2. |
| `tp2` | 2 | Full checkpoint, DiT TP2 and native Qwen encoder TP2; inspect the actual loaded encoder group and conditioning errors. |
| `ulysses2` | 2 | Full checkpoint, Ulysses2 sequence parallelism. |
| `ring2` | 2 | Full checkpoint, Ring2 with `--attention fa`. |
| `cfg2` | 2 | Positive/negative CFG ranks; Ovis preserves the serial BF16 combination order. |
| `tp2-sp2` | 4 | Full checkpoint, DiT TP2 × Ulysses2 and native Qwen encoder TP2; four GPUs are required. |
| `vae-tiled` | 1 | Native and reference both enable VAE tiling at 1024 × 1536, above the native threshold. |
| `vae-spatial` | 2 | Ulysses2 plus `--vae-tiling --vae-sp` at 1024 × 1536; compared with the same-size tiled reference. |
| `http` | 1 | Real server: repeated request, changed prompt/size then restore, and two distinct outputs. Always 512/4 with SDPA and CPU RNG. |

The `vae_sp` control requires `vae_tiling=True`; native defaults disable both
and synchronize `vae_config.use_parallel_decode=False`. VAE spatial and tiled
results must be reported separately from ordinary decode. The monolithic native
encoder inherits the DiT TP group when the loader does not fold it. `tp2` and
`tp2-sp2` therefore cover the actual Qwen encoder TP2 path. The serving
`--encoder-tp` option belongs to disaggregated/pool serving and does not control
this monolithic runner; there is no separate encoder-TP case here.
Each native rank records `resolved_encoder_tp_group` from the loaded encoder's
bound group and first layer: group ranks, actual QKV/row/MLP TP sizes, local head
counts, and weight shapes/dtypes. Rank 0's group is also saved under
`resolved_config`; inspect these records rather than a requested ServerArgs field.

Ovis opts into HF-compatible Qwen numerics. In low-precision TP, each row
projection temporarily gathers its input and weight before executing the full
GEMM. Stored parameters remain sharded, and gathered weights are released after
the projection; no additional weight cache is introduced. This trades extra
communication and repeated row-projection computation for the original
conditioning tolerances. FP32 split-K accumulation alone did not meet those
tolerances on the full 28-layer encoder. Report actual latency and peak memory
rather than assuming that TP accelerates the text encoder.

## Record actual results

Populate this table only from completed runs. Keep failures and unavailable
cases visible. The matrix's `pass` status means that execution/comparison
completed; acceptance still requires inspecting numerical errors and images.
Do not treat approximate-cache on/off outputs as an exact-equality oracle, and
do not infer a quality or performance baseline from a four-step smoke image.

| Validation | Profile / hardware / native SHA | Evidence | Actual result |
| --- | --- | --- | --- |
| Component numerical oracles | To fill | Log and tolerances | Pending |
| Full default-profile native vs pinned reference | To fill | Conditioning/prediction/trajectory/image errors, PSNR/SSIM, images | Pending |
| Offload matrix | To fill | Resolved residency, tensors, peak memory | Pending |
| TP2 / Ulysses2 / Ring2 / CFG2 | To fill | Actual encoder groups, conditioning errors and per-case comparison | Pending |
| TP2 × SP2 | To fill | Four-GPU tiny and full-model logs | Pending |
| Tiled / spatial VAE | To fill | Decode mode, outputs, memory, reference comparison | Pending |
| Repeated HTTP requests | To fill | Server lifecycle log and same-seed comparisons | Pending |

No GPU results, throughput baseline, or measured quality threshold are claimed
by this document. The main session fills actual evidence after queued runs.
