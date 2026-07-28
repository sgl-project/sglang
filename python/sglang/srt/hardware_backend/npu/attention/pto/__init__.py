"""PTO (Parallel Tile Operation) operators vendored from pypto-gym for Ascend NPU.

These wrap Hisilicon's pypto-gym kernels (commit 5405139) for use inside sglang's
NPU attention backends. They require the `pypto` package (JIT tile compiler) to be
installed — see scripts/install_pto_k3.sh for the standard install.
"""
