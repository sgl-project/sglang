"""Probe how miles resolves Megatron's async checkpoint strategy inside a ROCm image.

Throwaway verification helper for radixark/miles#2816. Megatron defaults
--async-strategy to "nvrx", which needs nvidia-resiliency-ext; the ROCm images strip
that CUDA-only package, so --async-save dies at the first checkpoint write. Run with
the strategy the caller expects, e.g.

    python3 verify_async_ckpt_strategy.py nvrx    # unpatched image, reproduces the bug
    python3 verify_async_ckpt_strategy.py mcore   # after applying the fix
"""

import importlib.util
import subprocess
import sys
from argparse import Namespace


def report_environment() -> None:
    miles_head = subprocess.run(
        ["git", "-C", "/root/miles", "log", "--oneline", "-1"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"miles HEAD:                     {miles_head or '<unavailable>'}")

    import torch

    print(f"torch:                          {torch.__version__} (hip={torch.version.hip})")
    print(f"nvidia_resiliency_ext present:  {importlib.util.find_spec('nvidia_resiliency_ext') is not None}")

    from megatron.training.utils import has_nvrx_checkpointing_async_support

    print(f"megatron nvrx async support:    {has_nvrx_checkpointing_async_support()}")


def resolve_async_strategy() -> str:
    """Ask miles to fill in Megatron defaults for an --async-save run, as train.py would."""
    from miles.backends.megatron_utils.arguments import set_default_megatron_args

    args = Namespace(
        true_on_policy_mode=False,
        optimizer="adam",
        debug_disable_optimizer=False,
        multi_lora_n_adapters=0,
        fp16=False,
        seq_length=4096,
        async_save=True,
        async_strategy="nvrx",
        use_persistent_ckpt_worker=True,
        multi_latent_attention=False,
        rope_type="rope",
        vocab_size=None,
        padded_vocab_size=None,
        tokenizer_model="/root/models/Qwen3-4B",
        tokenizer_type="HuggingFaceTokenizer",
        hf_checkpoint="/root/models/Qwen3-4B",
    )
    return set_default_megatron_args(args).async_strategy


def load_backend(strategy: str) -> str:
    """Import the async-checkpoint backend, the way the first checkpoint write does."""
    from megatron.core.dist_checkpointing.strategies.torch import get_async_strategy

    try:
        resolved, _ = get_async_strategy(strategy)
    except ModuleNotFoundError as exc:
        return f"ModuleNotFoundError: {exc}"
    return f"loaded {resolved}"


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("nvrx", "mcore"):
        print(f"usage: {sys.argv[0]} <nvrx|mcore>", file=sys.stderr)
        return 2
    expected = sys.argv[1]

    report_environment()
    actual = resolve_async_strategy()
    print(f"resolved async_strategy:        {actual} (expected {expected})")

    # Whatever miles settled on is what Megatron imports at the first save, so pull
    # that backend up now instead of discovering it 20 minutes into a training run.
    outcome = load_backend(actual)
    print(f"backend import:                 {outcome}")

    if actual != expected:
        print(f"FAIL: expected async_strategy {expected!r}, got {actual!r}")
        return 1
    if expected == "nvrx" and not outcome.startswith("ModuleNotFoundError"):
        print("FAIL: expected the nvrx backend to be missing on a ROCm image")
        return 1
    if expected == "mcore" and outcome != "loaded mcore":
        print("FAIL: expected the mcore backend to import cleanly")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
