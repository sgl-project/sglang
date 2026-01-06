# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch

from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh


vae = ShapeVAE.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    use_safetensors=False,
    variant='fp16',
)


loader = SharpEdgeSurfaceLoader(
    num_sharp_points=0,
    num_uniform_points=81920,
)
mesh_demo = 'demos/demo.glb'
surface = loader(mesh_demo).to('cuda', dtype=torch.float16)
print(surface.shape)

latents = vae.encode(surface)
latents = vae.decode(latents)
mesh = vae.latents2mesh(
    latents,
    output_type='trimesh',
    bounds=1.01,
    mc_level=0.0,
    num_chunks=20000,
    octree_resolution=256,
    mc_algo='mc',
    enable_pbar=True
)

mesh = export_to_trimesh(mesh)[0]
mesh.export('output.obj')
