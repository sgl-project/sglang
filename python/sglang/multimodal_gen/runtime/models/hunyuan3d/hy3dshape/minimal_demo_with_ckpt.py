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

from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

model_path = 'tencent/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)


import torch
import yaml
from hy3dshape.utils import instantiate_from_config 
# For example, you can convert deepspeed weights to a single file
# cd output_folder/dit/overfitting_depth_16_token_4096_lr1e4/ckpt/ckpt-step=00004000.ckpt
# python3 zero_to_fp32.py ./ ./out --max_shard_size 30GB
# then you can get output_folder/dit/overfitting_depth_16_token_4096_lr1e4/ckpt/ckpt-step=00004000.ckpt/out/pytorch_model.bin
ckpt_cfg_path = 'output_folder/dit/overfitting_depth_16_token_4096_lr1e4/hunyuandit-mini-overfitting-flowmatching-dinol518-bf16-lr1e4-4096.yaml'
ckpt_path = 'output_folder/dit/overfitting_depth_16_token_4096_lr1e4/ckpt/ckpt-step=00004000.ckpt/out/pytorch_model.bin'
config = yaml.safe_load(open(ckpt_cfg_path, 'r'))
model = instantiate_from_config(config['model']['params']['denoiser_cfg'])
sd = torch.load(ckpt_path)
sd = {k.replace('_forward_module.model.', ''):v for k,v in sd.items()}
msg = model.load_state_dict(sd)
print(msg)
model = model.cuda().half()
pipeline_shapegen.model = model


image = 'tools/mini_testset/images/015.png'

# image = Image.open(image_path).convert("RGBA")
# if image.mode == 'RGB':
#     rembg = BackgroundRemover()
#     image = rembg(image)

# mesh = pipeline_shapegen(image=image, guidance_scale=1.0)[0]
mesh = pipeline_shapegen(image=image)[0]
mesh.export('demo.glb')
