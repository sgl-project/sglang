---
title: "Post-Processing"
metatags:
    description: "Use SGLang Diffusion post-processing for frame interpolation and spatial upscaling after generation."
---

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

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50%"}} />
    <col style={{width: "50%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--enable-frame-interpolation</code></td>
      <td>Enable frame interpolation. Model weights are downloaded automatically on first use.</td>
    </tr>
    <tr>
      <td><code>--frame-interpolation-exp &#123;EXP&#125;</code></td>
      <td>Interpolation exponent — <code>1</code> = 2× temporal resolution, <code>2</code> = 4×, etc. (default: <code>1</code>)</td>
    </tr>
    <tr>
      <td><code>--frame-interpolation-scale &#123;SCALE&#125;</code></td>
      <td>RIFE inference scale; use <code>0.5</code> for high-resolution inputs to save memory (default: <code>1.0</code>)</td>
    </tr>
    <tr>
      <td><code>--frame-interpolation-model-path &#123;PATH&#125;</code></td>
      <td>Local directory or HuggingFace repo ID containing RIFE <code>flownet.pkl</code> weights (default: <code>elfgum/RIFE-4.22.lite</code>, downloaded automatically)</td>
    </tr>
  </tbody>
</table>

### Supported Models

Frame interpolation uses the [RIFE](https://github.com/hzwer/Practical-RIFE)
(Real-Time Intermediate Flow Estimation) architecture. Only **RIFE 4.22.lite**
(`IFNet` with 4-scale `IFBlock` backbone) is supported. The network topology is
hard-coded, so custom weights provided via `--frame-interpolation-model-path`
must be a `flownet.pkl` checkpoint that is compatible with this architecture.

Other RIFE versions (e.g., older `v4.x` variants with different block counts)
or entirely different frame interpolation methods (FILM, AMT, etc.) are **not
supported**.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "33.33%"}} />
    <col style={{width: "33.33%"}} />
    <col style={{width: "33.33%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Weight</th>
      <th>HuggingFace Repo</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RIFE 4.22.lite *(default)*</td>
      <td><a href="https://huggingface.co/elfgum/RIFE-4.22.lite"><code>elfgum/RIFE-4.22.lite</code></a></td>
      <td>Lightweight model, downloaded automatically on first use</td>
    </tr>
  </tbody>
</table>

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

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50%"}} />
    <col style={{width: "50%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--enable-upscaling</code></td>
      <td>Enable post-generation upscaling using Real-ESRGAN.</td>
    </tr>
    <tr>
      <td><code>--upscaling-scale &#123;SCALE&#125;</code></td>
      <td>Desired upscaling factor (default: <code>4</code>). The 4× model is used internally; if a different scale is requested, a bicubic resize is applied after the network output.</td>
    </tr>
    <tr>
      <td><code>--upscaling-model-path &#123;PATH&#125;</code></td>
      <td>Local <code>.pth</code> file, HuggingFace repo ID, or <code>repo_id:filename</code> for Real-ESRGAN weights (default: <code>ai-forever/Real-ESRGAN</code> with <code>RealESRGAN_x4.pth</code>, downloaded automatically). Use the <code>repo_id:filename</code> format to specify a custom weight file from a HuggingFace repo (e.g. <code>my-org/my-esrgan:weights.pth</code>).</td>
    </tr>
  </tbody>
</table>

### Supported Models

Upscaling supports two Real-ESRGAN network architectures. The correct
architecture is **auto-detected** from the checkpoint keys, so you only need to
point `--upscaling-model-path` at a valid `.pth` file:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "33.33%"}} />
    <col style={{width: "33.33%"}} />
    <col style={{width: "33.33%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Architecture</th>
      <th>Example Weights</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>RRDBNet</strong></td>
      <td><code>RealESRGAN_x4plus.pth</code></td>
      <td>Heavier model with higher quality; best for photos</td>
    </tr>
    <tr>
      <td><strong>SRVGGNetCompact</strong></td>
      <td><code>RealESRGAN_x4.pth</code> *(default)*, <code>realesr-animevideov3.pth</code>, <code>realesr-general-x4v3.pth</code></td>
      <td>Lightweight model; faster inference, good for video</td>
    </tr>
  </tbody>
</table>

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
