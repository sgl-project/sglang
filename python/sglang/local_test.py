import os
import random

import numpy as np
import torch

import sglang as sgl

model_path = "/shared/public/models/Qwen/Qwen2.5-1.5B-Instruct/"

# Set a fixed seed for reproducibility
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(SEED)

    runtime = sgl.Runtime(model_path=model_path)
    print(runtime.generate("Who is Steve Jobs?", {"top_k": 1}))
    runtime.shutdown()

    set_seed(SEED)
    engine = sgl.Engine(model_path=model_path)
    # bug: default sampling param should be {}
    print(engine.generate("Who is Steve Jobs?", {"top_k": 1}))
    engine.shutdown()
