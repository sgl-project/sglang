# Post-Processing

SGLang diffusion supports optional post-processing steps that run after
generation to improve temporal smoothness (frame interpolation) or spatial
resolution (upscaling). These steps are independent of the diffusion model and
can be combined in a single run.

When both are enabled, **frame interpolation runs first** (increasing the frame
count), then **upscaling runs on every frame** (increasing the spatial
resolution).

---

## Frame Interpolation (video only)

Frame interpolation synthesizes new frames between each pair of consecutive
generated frames, producing smoother motion without re-running the diffusion
model.

The `--frame-interpolation-exp` flag controls how many rounds of interpolation
to apply: each round inserts one new frame into every gap between adjacent
frames, so the output frame count follows the formula:

> **(N − 1) × 2^exp + 1**
>
> e.g. 5 original frames with `exp=1` → 4 gaps × 1 new frame + 5 originals = **9** frames;
> with `exp=2` → **17** frames.

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--enable-frame-interpolation` | Enable frame interpolation. Model weights are downloaded automatically on first use. |
| `--frame-interpolation-exp {EXP}` | Interpolation exponent — `1` = 2× temporal resolution, `2` = 4×, etc. (default: `1`) |
| `--frame-interpolation-scale {SCALE}` | RIFE inference scale; use `0.5` for high-resolution inputs to save memory (default: `1.0`) |
| `--frame-interpolation-model-path {PATH}` | Local directory or HuggingFace repo ID containing RIFE `flownet.pkl` weights (default: `elfgum/RIFE-4.22.lite`, downloaded automatically) |

### Supported Models

Frame interpolation uses the [RIFE](https://github.com/hzwer/Practical-RIFE)
(Real-Time Intermediate Flow Estimation) architecture. Only **RIFE 4.22.lite**
(`IFNet` with 4-scale `IFBlock` backbone) is supported. The network topology is
hard-coded, so custom weights provided via `--frame-interpolation-model-path`
must be a `flownet.pkl` checkpoint that is compatible with this architecture.

Other RIFE versions (e.g., older `v4.x` variants with different block counts)
or entirely different frame interpolation methods (FILM, AMT, etc.) are **not
supported**.

| Weight | HuggingFace Repo | Description |
|--------|------------------|-------------|
| RIFE 4.22.lite *(default)* | [`elfgum/RIFE-4.22.lite`](https://huggingface.co/elfgum/RIFE-4.22.lite) | Lightweight model, downloaded automatically on first use |

### Example

Generate a 5-frame video and interpolate to 9 frames ((5 − 1) × 2¹ + 1 = 9):

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "A dog running through a park" \
  --num-frames 5 \
  --enable-frame-interpolation \
  --frame-interpolation-exp 1 \
  --save-output
```

---

## Upscaling (image and video)

Upscaling increases the spatial resolution of generated images or video frames
using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). The model weights
are downloaded automatically on first use and cached for subsequent runs.

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--enable-upscaling` | Enable post-generation upscaling using Real-ESRGAN. |
| `--upscaling-scale {SCALE}` | Desired upscaling factor (default: `4`). The 4× model is used internally; if a different scale is requested, a bicubic resize is applied after the network output. |
| `--upscaling-model-path {PATH}` | Local `.pth` file, HuggingFace repo ID, or `repo_id:filename` for Real-ESRGAN weights (default: `ai-forever/Real-ESRGAN` with `RealESRGAN_x4.pth`, downloaded automatically). Use the `repo_id:filename` format to specify a custom weight file from a HuggingFace repo (e.g. `my-org/my-esrgan:weights.pth`). |

### Supported Models

Upscaling supports two Real-ESRGAN network architectures. The correct
architecture is **auto-detected** from the checkpoint keys, so you only need to
point `--upscaling-model-path` at a valid `.pth` file:

| Architecture | Example Weights | Description |
|--------------|-----------------|-------------|
| **RRDBNet** | `RealESRGAN_x4plus.pth` | Heavier model with higher quality; best for photos |
| **SRVGGNetCompact** | `RealESRGAN_x4.pth` *(default)*, `realesr-animevideov3.pth`, `realesr-general-x4v3.pth` | Lightweight model; faster inference, good for video |

The default weight file is
[`ai-forever/Real-ESRGAN`](https://huggingface.co/ai-forever/Real-ESRGAN) with
`RealESRGAN_x4.pth` (SRVGGNetCompact, 4× native scale).

Other super-resolution models (e.g., SwinIR, HAT, BSRGAN) are **not supported**
— only Real-ESRGAN checkpoints using the two architectures above are
compatible.

### Examples

Generate a 1024×1024 image and upscale to 4096×4096:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --prompt "A cat sitting on a windowsill" \
  --output-size 1024x1024 \
  --enable-upscaling \
  --save-output
```

Generate a video and upscale each frame by 4×:

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A curious raccoon" \
  --enable-upscaling \
  --upscaling-scale 4 \
  --save-output
```

---

## Combining Frame Interpolation and Upscaling

Frame interpolation and upscaling can be combined in a single run.
Interpolation is applied first (increasing the frame count), then upscaling is
applied to every frame (increasing the spatial resolution).

Example — generate 5 frames, interpolate to 9 frames, and upscale each frame
by 4×:

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A curious raccoon" \
  --num-frames 5 \
  --enable-frame-interpolation \
  --frame-interpolation-exp 1 \
  --enable-upscaling \
  --upscaling-scale 4 \
  --save-output
```
