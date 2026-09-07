"""Compatibility entry point; CI dispatches diffusion via ``test/run_suite.py``."""

from sglang.multimodal_gen.test.runner.diffusion_suite_runner import main

if __name__ == "__main__":
    main()
