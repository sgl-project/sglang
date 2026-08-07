# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import os

if os.environ.get("SGLANG_LIGHTWEIGHT_RUNTIME") != "1":
    from sglang.multimodal_gen.runtime.utils.logging_utils import globally_suppress_loggers

    globally_suppress_loggers()
