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
    --num-samples 100 \
    --shuffle \
    --ref-size 256
```

Outputs are written under a run-specific subfolder:

- Prompts: `/path/to/fid_runs/n100_seed42_ref256/prompt.txt` (plain text, one prompt per line, no header or extra columns)
- Reference images: `/path/to/fid_runs/n100_seed42_ref256/`

The generated `prompt.txt` and reference images are selected from the same COCO rows. The script keeps one caption per image so the prompt count and reference image count match.

### 3) Generate the sample batch

Use the generated `prompt.txt` to generate images:

```bash
sglang generate \
    --model-path black-forest-labs/FLUX.1-dev \
    --num-gpus 4 \
    --tp-size 4 \
    --dit-precision bf16 \
    --vae-precision bf16 \
    --text-encoder-precisions bf16 bf16 \
    --prompt-file-path /path/to/fid_runs/n100_seed42_ref256/prompt.txt \
    --height 256 \
    --width 256 \
    --num-inference-steps 15 \
    --save-output \
    --output-path /path/to/fid_runs/n100_samples
```

### 4) Compute FID

Run:

```bash
python3 -m sglang.multimodal_gen.benchmarks.fid.compute_fid \
    --ref-dir /path/to/fid_runs/n100_seed42_ref256 \
    --sample-dir /path/to/fid_runs/n100_samples \
    --device cuda
```
