# MinWM 5B realtime parity harness

This harness compares the current minWM `main` V3 inference path with SGLang's
realtime WebSocket API. The formal ten cases lock prompt, seed, first frame,
geometry, chunk count, and the exact 81-class `primitive_token_residual` action
label. `cases_720p_5s.json` is a separate three-case long-video regression using
continuous primitive weights.

For an architecture-first review of the implementation, numerical-parity
decisions, sequence-parallel status, and the required comprehension quiz, read
[`CHANGE_GUIDE.zh-CN.md`](CHANGE_GUIDE.zh-CN.md).

## Inputs

- Latest rerun checkpoint:
  `wan22-5B-stage3-dmd-8-0721-6a531f0e067/global_step_003200/ema_student/model.pt`,
  identified by byte count, VersionId, ETag, and CRC64NVME in the companion
  implementation document. The older `global_step_005600` result below remains a
  historical accepted artifact.
- minWM checkout: commit from `main`, recorded by the baseline runner.
- Complete Wan2.2 TI2V 5B donor directory. Current minWM V3 first constructs its
  DiT from donor `transformer/` before replacing it with `model.pt`, so the baseline
  requires that directory even though its final generator weights come entirely
  from the requested checkpoint. The SGLang converter links only `text_encoder`,
  `tokenizer`, `vae`, and `scheduler`; it never copies the donor transformer into
  the serving model.
- One NVIDIA GPU. Run baseline and SGLang sequentially on the same machine and
  keep the GPU type, PyTorch/CUDA versions, and attention backend fixed.

Convert the native student checkpoint next to the AWS-local 10 GB file:

```bash
python python/sglang/multimodal_gen/tools/convert_minwm_checkpoint.py \
  --minwm-checkpoint /fsx/minwm/model.pt \
  --donor-diffusers-dir /fsx/minwm/Wan2.2-TI2V-5B-from-diffusers \
  --output-dir /fsx/minwm/sglang-model \
  --link-donor \
  --source-version-id FbCvgw5rl1UXt9MKBgpxYpb4BJoUyUEi \
  --source-etag 56dbc7ce13f26c55d0bfd255e471d318-191
```

## One-command ten-case run

```bash
export MINWM_ROOT=/workspace/minWM
export MINWM_CHECKPOINT=/fsx/minwm/model.pt
export MINWM_PRETRAINED_DIR=/fsx/minwm/Wan2.2-TI2V-5B-from-diffusers
export MINWM_MODEL_DIR=/fsx/minwm/sglang-model
export MINWM_PARITY_PROFILE=bitwise
./benchmark/minwm_realtime_parity/run_all.sh /fsx/minwm/results/run-001
```

The server is launched with `--performance-mode speed` so the 5B DiT stays GPU
resident when memory permits. Whole-model `torch.compile` defaults to `false`
because it changes this checkpoint's numerical trajectory; opt in with
`MINWM_ENABLE_TORCH_COMPILE=true` only after treating it as a separate parity
profile. The default attention backend is `fa`; override it with
`MINWM_ATTENTION_BACKEND`. A failed bitwise profile is evidence to inspect, not
permission to silently relax the threshold. Select a numerical profile only
after its bound has been reviewed against results from the same backend matrix.

`primitive_token_residual` does not require a magnitude. `action_labels` and
`camera_actions` are binary inputs; continuous magnitude is sent as flattened
per-decoded-frame `action_weights` rows ordered `[w,a,s,d,i,j,k,l]`, each in
`[0,1]`. For example, `w=0.8` is `[0.8,0,0,0,0,0,0,0]`. The adapter groups four
rows into each latent-frame window. An init/event must select only one action form.
The 0721 checkpoint inherited `action_output_format=label_81` during training, so
binary `w` remains its canonical in-distribution input. Fractional values exercise
the continuous interpolation path in current minWM `main`; they are not calibrated
physical speeds without a separate controllability study.

Native minWM KV is unbounded (`local_attn_size=-1`, `sink_size=0`). For bounded
`max_chunks=N`, SGLang allocates the complete `1 + 4N` latent-frame horizon; an
unbounded session grows the cache. `--kv-cache-num-frames 45` and 128 are explicit
performance ablations, not native minWM windows.

Packed attention follows minWM `main`'s hardware fallback: FA4 on Blackwell when
available, FA3 on Hopper when available, otherwise FA2. Forcing FA4 on the H200
while the baseline selected FA2 caused first-generated-frame drift, so backend
identity is part of the parity contract rather than an implementation detail.

For a smoke run, pass `--case CASE_ID` separately to both Python runners before
running `compare_results.py` on a full manifest. The official report always
contains all ten cases.

## Throughput comparison

`benchmark_realtime_throughput.py` measures one persistent API session with 20
warmup chunks and 200 measured chunks. It reports GPU scheduler-forward FPS,
whole-server chunk FPS, client-observed FPS, and p50/p95/p99 stage latencies without
retaining the multi-gigabyte raw frame stream:

```bash
python benchmark/minwm_realtime_parity/benchmark_realtime_throughput.py \
  --output /fsx/minwm/results/exact-kv45.json \
  --profile-name exact-packed-det-kv45 \
  --kv-cache-num-frames 45
```

TTFF is measured from the start of the init send through the last frame payload of
chunk zero. It includes first-frame/T5/VAE setup and first-shape compilation when
enabled; steady FPS excludes that startup boundary.

The long-video contract uses current minWM eval's valid 720p tier, 1248x704. Exact
1280x720 is incompatible with the `/16` VAE followed by the DiT's 2x2 spatial
patch. At 24 FPS, eight fixed 16-frame chunks plus the reference produce 129 frames
or 5.375 seconds; the harness does not resize or trim that boundary.
`MINWM_LONG_ENABLE_TORCH_COMPILE` defaults to `false`: the 1248x704 whole-DiT
compile failed in Torch 2.11 Inductor before producing an output, so it has not
passed the user's strict-bitwise gate. minWM's native compiled fused segments
remain enabled in both baseline and serving paths.

Use the same converted MinWM 5B checkpoint for each server profile. A MinWM
checkpoint cannot be loaded into the LingBot model class: MinWM is a 30-layer,
width-3072, 48-channel model with token-residual action, while LingBot World 2 is a
40-layer, width-5120, 36-to-16-channel model with Plucker/camera conditioning. The
fair same-5B implementation A/B is therefore the exact source-shaped MinWM path
versus its SGLang dense/optimized ablations. Keep hardware, weights, request,
action label, KV window, and software image fixed.

The official LingBot World 2 release currently lists a 14B causal-fast checkpoint
and future 1.3B checkpoints, but no 5B checkpoint. Do not relabel a differently
shaped community model as a same-architecture 5B control.

For context, one MinWM 5B replica and one LingBot 14B SP8 stream have similar
per-GPU dense-compute proxies: about 33.86 versus 34.96 TMAC per chunk per GPU.
Thus similar single-stream latency is plausible even though MinWM is smaller; the
node-level capacity comparison is eight independent 5B replicas versus one SP8
LingBot stream, and must be measured separately from single-stream latency.

## Synchronized player

After comparison, one command validates all ten baseline/SGLang pairs and opens
the generated `player/index.html`. All ten cases are laid out on the page; use a
case card's **Play both** button for synchronized play/pause. Seek, speed, and
frame stepping also control that pair.
The page shows both relative MP4 paths and a ready/error state so a missing or
unreadable visualization artifact is explicit:

```bash
./benchmark/minwm_realtime_parity/play.sh /fsx/minwm/results/run-001
```

The generated page embeds the ten-case report, so `player/index.html` can also be
opened directly with a `file://` URL; it does not depend on a local HTTP server.
Keep the `player/` and sibling `cases/` directories together so the relative MP4
paths remain valid.

Lossless `baseline.npy` and `sglang.npy` arrays are the metric source; MP4 files
are only visualization artifacts and are never used for numeric acceptance.

## Latest-checkpoint formal run

The 2026-07-22 Spot B200 formal result for the requested latest checkpoint is:

```text
s3://leap-world-us-east-2/world-model/evals/minwm/realtime-parity/20260722-codex-578d/results/latest-ckpt-v6-full-attempt07-west-spot/ten-case/
```

All ten strict-bitwise cases pass: generated max absolute error `0`, RMSE `0`,
minimum SSIM `1.0`, and 10/10 byte-identical generated-frame arrays. It contains
ten baseline/SGLang pairs and 20 MP4 files. This checkpoint's native matching
config is non-varlen causal T2V without a first-frame processor; the formal API
run is intentionally the V3 first-frame/action compatibility contract and must
not be relabeled as native eval.

On a host with the project S3 mount, open the latest pair player with:

```bash
./benchmark/minwm_realtime_parity/play.sh \
  /s3/world-model/evals/minwm/realtime-parity/20260722-codex-578d/results/latest-ckpt-v6-full-attempt07-west-spot/ten-case
```

The latest throughput matrix is kept separately under
`latest-ckpt-v7-profiles-attempt08-west-spot` so a profiles-only rerun cannot
overwrite the accepted videos or parity report. All six profiles completed. On one
B200, exact KV45 measured `23.075 FPS`; same-weight LingBot-style dense measured
`24.713 FPS`; dense plus optimized components measured `25.541 FPS`; and the
non-bitwise whole-compile speed ceiling measured `32.222 FPS`. Exact loses `6.63%`
to dense attention and `9.65%` to the optimized non-whole-compiled path. The pure
deterministic flag showed no measurable penalty. Full TTFF, p50/p95/p99, memory,
stage timing, and isolated deltas are in `throughput-summary.json` and the companion
implementation document.

The separate 720p long-video H200 attempt 15 used 1248x704, 24 FPS, 129 frames
(5.375 seconds), eight chunks, and native full-history KV. Its three continuous
action cases (`w=0.8`, idle, and `w=0.6+l=0.4`) all pass strict bitwise with zero
generated-frame error. Aggregate steady client/scheduler throughput is
`10.393/10.365 FPS`, chunk p50 is `1525.20 ms`, and peak memory is `53,159 MB`.
First-case TTFF is `10.343 s` because it includes first-shape fused-segment
compilation; the next two are `1.963 s` and `1.960 s`. The verified flat player is
at `results/latest-checkpoint-720p-5s-h200/latest/player/index.html`. These H200
720p values are not substituted for the formal B200 832x480 matrix.

Artifact export publishes `720p-artifacts-ready` only after the complete result
tree has been copied and can optionally retain the pod for a capture window. The
metric collector also short-circuits exact-equal lossless arrays instead of
recomputing float64 cosine and per-frame SSIM identities.

## Historical accepted run

The previous checkpoint's same-runtime B200 ten-case artifact is:

```text
/s3/world-model/evals/minwm/realtime-parity/20260721-codex-578d/results/ten-case-main-same-env-final-attempt34
```

On a host with the project S3 mount, open its 20 synchronized videos with:

```bash
./benchmark/minwm_realtime_parity/play.sh \
  /s3/world-model/evals/minwm/realtime-parity/20260721-codex-578d/results/ten-case-main-same-env-final-attempt34
```

All ten reference and generated lossless arrays are byte-for-byte identical. The
checked-in strict `bitwise` profile passes 10/10 cases with generated max absolute
error `0`, RMSE `0`, and SSIM `1.0`; no numerical fallback profile is used. The
effective serving overlay SHA-256 is
`42ef254699d6e7837e7c0caaac077e1ce20bee78aa4a2aec4e3850b0af7bf4bc`.

The earlier failed `ten-case-main-same-env-final-attempt23` artifact is retained as
diagnostic history for the pre-layout/packed-attention implementation. Its observed
errors were not used to widen `thresholds.json`.
