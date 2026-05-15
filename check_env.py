import os
import sys

print("Executable:", sys.executable)
print("Path:", sys.path)
try:
    import torch

    print("Torch version:", torch.__version__)
    print("Torch file:", torch.__file__)
except ImportError as e:
    print("ImportError:", e)

try:
    import sglang

    print("SGLang file:", sglang.__file__)
except ImportError as e:
    print("SGLang ImportError:", e)
