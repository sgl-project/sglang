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

from .pipeline import HunyuanPaintPipeline
from .unet.model import HunyuanPaint
from .unet.modules import (
    Dino_v2,
    Basic2p5DTransformerBlock,
    ImageProjModel,
    UNet2p5DConditionModel,
)
from .unet.attn_processor import (
    PoseRoPEAttnProcessor2_0,
    SelfAttnProcessor2_0,
    RefAttnProcessor2_0,
)

__all__ = [
    'HunyuanPaintPipeline',
    'HunyuanPaint',
    'Dino_v2',
    'Basic2p5DTransformerBlock',
    'ImageProjModel',
    'UNet2p5DConditionModel',
    'PoseRoPEAttnProcessor2_0',
    'SelfAttnProcessor2_0',
    'RefAttnProcessor2_0',
]
