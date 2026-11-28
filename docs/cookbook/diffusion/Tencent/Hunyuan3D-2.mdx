---
title: Hunyuan3D-2
description: "Deploy Hunyuan3D-2 with SGLang Diffusion to generate textured or shape-only 3D meshes from an image."
metatags:
    description: "Serve Hunyuan3D-2 for image-to-3D mesh generation, with optional texture painting, using SGLang Diffusion."
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/tencent/hunyuan3d-2.jsx';

<DiffusionModelTags tags={["3D mesh", "image-to-3D", "textured GLB", "shape + paint stages", "1 GPU"]} />

## 1. Quick start

Install SGLang with diffusion support by following the [SGLang Diffusion installation guide](/docs/sglang-diffusion/installation), then start the server and request a mesh.

<Deployment config={config} />

Mesh generation is asynchronous: `POST /v1/meshes` returns a job ID, `GET
/v1/meshes/<id>` reports its status, and `GET /v1/meshes/<id>/content`
downloads the GLB file once the status is `completed`.

## 2. Model capabilities

[Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) from Tencent turns one reference image into a 3D asset in two stages. A flow-matching DiT generates the shape, which is decoded into a mesh; the paint stage then removes lighting from the reference image and synthesizes a texture map for that mesh.

Choose the textured output for finished assets and shape-only when you texture the mesh elsewhere or only need geometry, which is much faster. The input should show one object on a clean background. The license is the Tencent Hunyuan Community License, which excludes some territories; read it before deploying.

| Output | What runs | Returns |
| --- | --- | --- |
| Textured mesh (default) | Shape DiT, mesh decode, delight, multiview paint, texture baking | Textured GLB |
| Shape only | Shape DiT and mesh decode | Untextured OBJ |

## 3. Deployment details

Hunyuan3D-2 runs on one GPU and does not use tensor or sequence parallelism.
The shape DiT runs 50 steps with guidance 5.0 and the paint model 30 steps by
default.

| Output | Hardware | Mesh request | Peak GPU memory |
| --- | --- | --- | --- |
| Textured GLB | H200 | 27.6 s | 8.9 GiB |
| Shape-only OBJ | H200 | 8.5 s | 4.6 GiB |

Measured with `sglang serve` on SGLang `main` at `385262cabbae1`: warmed `/v1/meshes`
requests with a fixed input image and seed; the time runs from the POST until
the mesh download finishes. Most of the shape-only time is mesh extraction and
export; the shape DiT itself takes about 2 s.

See the [CLI reference](/docs/sglang-diffusion/api/cli) for the remaining server flags.

## 4. Request examples

### 4.1 Offline generation

```bash Command
sglang generate --model-path tencent/Hunyuan3D-2 \
  --image-path input.png --seed 0 --save-output
```

This offline command has not been run end to end in the current release; the
HTTP server above is the verified path.

For a shape-only server, create `shape_only.json` with `{"paint_enable": false}`
and pass `--pipeline-config-path shape_only.json` to `sglang serve`.

## 5. Run in ComfyUI

The [SGLDiffusion plugin](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion) has image and video nodes but no mesh node, so Hunyuan3D-2 cannot be run from ComfyUI through SGLang. Use the HTTP API above.
