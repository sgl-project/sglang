# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from sglang.multimodal_gen.runtime.utils.logging_utils import suppress_loggers

# globally suppress some obsessive loggers
suppress_loggers(
    ["imageio", "imageio_ffmpeg", "PIL", "PIL_Image", "multipart", "filelock"]
)
