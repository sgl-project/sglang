import json
import os
import random

import numpy as np
import torch

import sglang as sgl

model_path = "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/"  # "/shared/public/models/Qwen/Qwen2.5-1.5B-Instruct/"

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
    print("==== Answer 1 ====")
    ans = json.loads(runtime.generate("Who is Steve Jobs?", {"top_k": 1}))
    print(ans["text"])
    # runtime.shutdown()

    sgl.flush_cache()
    set_seed(SEED)
    # engine = sgl.Engine(model_path=model_path)
    # bug: default sampling param should be {}
    print("==== Answer 2 ====")
    ans = json.loads(runtime.generate("Who is Steve Jobs?", {"top_k": 1}))
    print(ans["text"])

    sgl.flush_cache()
    set_seed(SEED)
    # engine = sgl.Engine(model_path=model_path)
    # bug: default sampling param should be {}
    print("==== Answer 3 ====")
    ans = json.loads(runtime.generate("Who is Steve Jobs?", {"top_k": 1}))
    print(ans["text"])

    runtime.shutdown()
