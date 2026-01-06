# Data Processing

This is the data processing pipeline for 3D shape and texture generation.

**Notes**:
1. This implementation is a simplified version of our industrial pipeline.
2. The rendering script is based on [TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/blender_script/render.py).

## Rendering

### Motivation
The rendering script `render/render.py` serves three main purposes:
1. Converting complex 3D formats to PLY files using Blender for further processing.
2. Rendering condition images for DiT training.
3. Rendering orthogonal images, PBR materials, and conditional signals (world-space normals and positions) for texture generation.

### Requirements
The rendering scripts are executed with Blender 4.1. You need to install `opencv`, `OpenEXR`, and `Imath` using Blender's Python. Here is an example for a Macbook:
```bash
/Applications/Blender.app/Contents/Resources/4.1/python/bin/python3.11 -m pip install OpenEXR Imath opencv-python
```

### Execution
The first two purposes can be executed with a single command:
```bash
$BLENDER_PATH -b -P render/render.py -- \
    --object ${INPUT_FILE} --geo_mode --resolution 512 \
    --output_folder $OUTPUT_FOLDER
```
For the third purpose, simply remove the `--geo_mode` flag.

## Watertight Mesh Processing and Sampling

### Motivation
To learn an SDF representation for 3DShape2VecSets, we require a watertight input mesh. This pipeline processes raw triangle meshes to generate three essential data types:
1. **Surface samples** - Input points for the encoder.
2. **Volume samples** - Query points for SDF evaluation in the decoder.
3. **Volume SDFs** - Ground-truth signed distance values for VAE training.

### Execution
Process a triangle mesh (OBJ/OFF format) to generate:
1. Watertight mesh (`${OUTPUT_NAME}_watertight.obj`).
2. Surface point samples (`${OUTPUT_NAME}_surface.npz`).
3. Volume samples with SDFs (`${OUTPUT_NAME}_sdf.npz`).

**Command:**
```bash
python3 watertight/watertight_and_sample.py \
    --input_obj ${INPUT_MESH} \
    --output_prefix ${OUTPUT_NAME}
```

### Output Data Format

#### 1. Surface Samples (`${OUTPUT_NAME}_surface.npz`)
Contains two point cloud arrays in numpy NPZ format:

| Key             | Shape    | Format   | Description                     |
|-----------------|----------|----------|---------------------------------|
| `random_surface` | `(N, 6)` | `float16`| Uniform point samples on surface |
| `sharp_surface`  | `(M, 6)` | `float16`| Samples near sharp mesh edges   |

#### 2. Volume SDF Samples (`${OUTPUT_NAME}_sdf.npz`)
Contains three sample types stored as array pairs. For each type `${type}`:

| Sample Type     | Points Array         | SDF Labels Array     | Shape    | Format   | Description             |
|-----------------|----------------------|----------------------|----------|----------|-------------------------|
| `vol`          | `vol_points`        | `vol_label`         | `(P, 3)/(P,)` | `float16`| Random spatial samples |
| `random_near`   | `random_near_points` | `random_near_label`  | `(Q, 3)/(Q,)` | `float16`| Samples near surface   |
| `sharp_near`    | `sharp_near_points`  | `sharp_near_label`   | `(R, 3)/(R,)` | `float16`| Samples near sharp edges |

**Data Specifications**:
- All point coordinates (`*_points` arrays) contain 3D positions stored as `float16` values.
- All SDF values (`*_label` arrays) are `float16` scalars representing:
  - **Positive values**: Outside the surface.
  - **Negative values**: Inside the surface.
  - **Zero values**: On the surface.
- Array dimensions:
  - `N`, `M`, `P`, `Q`, `R` represent sample counts (vary per shape).
  - `3` indicates XYZ coordinates.
  - `6` indicates XYZ/Normal coordinates.
- All arrays are stored uncompressed in numpy's NPZ format.

## Overall Script
Modify the first four variables in `pipeline.sh`:
1. **INPUT_FILE** The path to each 3D data file.
2. **OUTPUT_FOLDER** The overall path for the output dataset.
3. **NAME** The naming for the output path of each data.
4. **BLENDER_PATH** The executable path for Blender.

Then run the following script:
```bash
bash pipeline.sh
```