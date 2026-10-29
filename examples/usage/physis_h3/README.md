# Physis H3 same-step inference

This pipeline supports `Physis-AI/physisserve-minimaxh3_33b-same_step-ls-dev`
with SGLang's native H3 DiT and Qwen3-VL text encoder. It uses the Diffusers
`AutoencoderKLMiniMaxH3` decoder, requiring `diffusers>=0.40.0`. Install a
Diffusers version compatible with the rest of your environment before launching.
The model directory name is detected automatically when it contains
`physisserve-minimaxh3`.

## Generate a video

Run from the repository root. `--degree` counts DiT GPUs; timestep/block modes
also require one dedicated GPU for the text encoder and VAE.

```bash
# Ulysses: four GPUs in total.
python examples/usage/physis_h3/generate.py \
  --model-path /path/to/physisserve-minimaxh3_33b-same_step-ls-dev \
  --mode off --degree 4

# Timestep pipeline: four DiT GPUs plus one decoder GPU.
python examples/usage/physis_h3/generate.py \
  --model-path /path/to/physisserve-minimaxh3_33b-same_step-ls-dev \
  --mode timestep --degree 4

# Block pipeline: seven DiT GPUs plus one decoder GPU.
python examples/usage/physis_h3/generate.py \
  --model-path /path/to/physisserve-minimaxh3_33b-same_step-ls-dev \
  --mode block --degree 7
```

The default request produces 158 frames (47 latent chunks) at 1344×768 and 24
FPS. Use `--height 256 --width 256 --num-frames 22` for a shorter,
lower-resolution video.
The first five latents are denoised jointly. Subsequent latents are generated
one at a time. Use `--bootstrap-chunks 1` to initialize a single latent instead.
The Python equivalent is `pipeline_config={"physis_bootstrap_chunks": 1}`.
Width and height must be multiples of 32. Supported frame counts are `17n+5`
with `n>=1`, matching complete H3 decoder groups. Each request accepts one
positive text prompt, one integer seed in `[0, 2**31)`, and one video.
Image/audio conditioning, negative prompts and CFG are unsupported. The
checkpoint uses a fixed four-step sigma schedule; `num_inference_steps` must
be 4 and `guidance_scale` must be 1.

## Parallel execution and memory

All modes use same-step noisy KV history. Each denoising step keeps
its own video history: a near-clean opening sink plus 18 recent chunks. The sink
forward uses sigma=0.001 and a latent re-noised to that level; the opening latent
returned for VAE decoding stays clean. Keys are
cached before RoPE; rolling positions preserve the five-latent H3 clock. Text
rows are recomputed and remain part of every block's packet. Request seeds are
mapped deterministically to chunk/step noise using the reference seed schedule,
independently of placement. Initial noise is drawn in bfloat16; subsequent
re-noising uses float32 arithmetic before rounding back to bfloat16.

Timestep placement assigns whole denoising steps. Block placement partitions
the ordered `(step, block)` sequence and can split a step across GPUs. For four
steps and four DiT GPUs, both placements assign the same ranges. Pipeline
parallelism with more than four denoising GPUs requires block placement.
This is SGLang's static, single-request pipeline; it does not implement
PhysisServe's multi-stream placement search.

DiT weights are replicated on the denoising GPUs. KV ownership follows
the assigned ranges. The opening group and its near-clean sink are bootstrapped on
each denoising GPU before the remaining chunks enter the pipeline. The other
opening latents retain their same-step KV at their assigned owners. This means
short clips can be dominated by startup rather than steady-state throughput.
Joint initialization also needs temporary KV for the opening group. Set
`physis_bootstrap_chunks` no larger than `physis_history_chunks + 1`.

The H3 VAE decodes the complete latent sequence once; this pipeline returns a
completed video. The Ulysses example offloads the text encoder after use to leave
room for KV; pipeline modes keep it on the decoder GPU. GPU memory use grows with
resolution and retained KV history.
Eager execution is required; compilation, FSDP, quantization, LoRA and
layerwise offload are rejected.
