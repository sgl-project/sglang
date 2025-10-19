"""
ModelOpt related constants
"""

QUANT_CFG_CHOICES = {
    "fp8": "FP8_DEFAULT_CFG",
    "int4_awq": "INT4_AWQ_CFG",  # TODO: add support for int4_awq
    "w4a8_awq": "W4A8_AWQ_BETA_CFG",  # TODO: add support for w4a8_awq
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "nvfp4_awq": "NVFP4_AWQ_LITE_CFG",  # TODO: add support for nvfp4_awq
}
