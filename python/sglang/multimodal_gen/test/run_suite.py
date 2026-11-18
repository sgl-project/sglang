"""Entry point for the AMD/NPU/XPU diffusion lanes.

CUDA lanes reach the same runner through ``test/run_suite.py`` and
``sglang.test.ci.diffusion_suite_bridge``.
"""

from sglang.multimodal_gen.test.runner.diffusion_suite_runner import main

if __name__ == "__main__":
    main()
