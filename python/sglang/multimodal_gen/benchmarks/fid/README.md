<!-- SPDX-License-Identifier: Apache-2.0 -->

### FID Benchmark

This benchmark computes FID between a **reference batch** (COCO validation images) and a **sample batch** (images generated from the same prompts).

This benchmark was copied and adapted from the xDiT FID benchmark: https://github.com/xdit-project/xDiT/tree/main/benchmark/fid.

### Prerequisites

Install the additional dependency used for FID:

```bash
pip3 install pytorch-fid
```

Notes:

- `preprocess_dataset.py` defaults to resizing reference images to `256x256`. If Pillow is not installed, please install it (`pip3 install pillow`)

### 1) Prepare the COCO validation set and prompts

Download COCO validation images and captions metadata:

- Validation images zip: http://images.cocodataset.org/zips/val2014.zip
- Captions metadata: http://images.cocodataset.org/annotations/annotations_trainval2014.zip

After unzipping, you should have something like:

```text
annotations/
    captions_val2014.json
    ...
val2014/
    COCO_val2014_000000xxxxxx.jpg
    COCO_val2014_000001xxxxxx.jpg
    ...
```

### 2) Create the reference batch (prompts + resized ref images)

From the repo root, run:

```bash
python3 -m sglang.multimodal_gen.benchmarks.fid.preprocess_dataset \
    --caption-file /path/to/annotations/captions_val2014.json \
    --coco-val-dir /path/to/val2014 \
    --out-dir /path/to/fid_runs \
    --num-samples 5000 \
    --shuffle \
    --ref-size 256
```

Outputs are written under a run-specific subfolder:

- Prompts: `/path/to/fid_runs/n5000_seed42_ref256/prompt.txt` (plain text, one prompt per line, no header or extra columns)
- Reference images: `/path/to/fid_runs/n5000_seed42_ref256/`

The generated `prompt.txt` and reference images are selected from the same COCO rows. The script keeps one caption per image so the prompt count and reference image count match.

> **Sample count matters.** FID uses 2048-dim Inception features, so a stable, full-rank estimate needs at least ~2000 samples (5k–10k recommended). Tiny runs such as `--num-samples 100` produce a hugely inflated, meaningless FID.

### 3) Generate the sample batch

Use the generated `prompt.txt` to generate images:

```bash
sglang generate \
    --model-path black-forest-labs/FLUX.1-dev \
    --num-gpus 1 \
    --tp-size 1 \
    --prompt-file-path /path/to/fid_runs/n5000_seed42_ref256/prompt.txt \
    --height 256 \
    --width 256 \
    --num-inference-steps 30 \
    --guidance-scale 3.5 \
    --seed 42 \
    --save-output \
    --output-path /path/to/fid_runs/n5000_samples
```

> **Faster large runs.** A 256×256 image fits comfortably on a single GPU, so tensor parallelism only adds cross-GPU communication overhead here. To generate many samples quickly, split `prompt.txt` into shards and run one single-GPU `sglang generate` worker per GPU, then merge all outputs into one folder before computing FID.

### 4) Compute FID

Run:

```bash
python3 -m sglang.multimodal_gen.benchmarks.fid.compute_fid \
    --ref-dir /path/to/fid_runs/n5000_seed42_ref256 \
    --sample-dir /path/to/fid_runs/n5000_samples \
    --device cuda
```

### Reference results

Sanity-check numbers on COCO val2014 (references resized to `256x256`, generation at `256x256`, 30 inference steps, `--seed 42`, ~10k samples), computed with `pytorch-fid` (InceptionV3, `dims=2048`):

| Model | Resolution | Steps | Guidance | Samples | FID |
|-------|------------|-------|----------|---------|-----|
| FLUX.1-dev | 256×256 | 30 | 3.5 | ~10k | 30.4 |
| FLUX.2-dev | 256×256 | 30 | 4.0 | ~10k | 25.7 |

FID depends on the sampling settings (steps, guidance, resolution) and the number of samples, so treat these as approximate references rather than exact targets.
