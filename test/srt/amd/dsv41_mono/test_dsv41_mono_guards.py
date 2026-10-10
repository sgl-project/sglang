"""CPU test of the FFN launch's model guard (``unsupported_model``): the checkpoint's config passes, and
each routing, activation and shape setting the kernels build in fails when it differs or is missing.
The normal MoE path's warmup runs every width up to MAX_ROWS once.

Run in the image: python3 test_dsv41_mono_guards.py [model path]
"""

import sys
import types

from sglang.srt.models.deepseek_common.amd.dsv41_mono_decode import (
    _MODEL,
    unsupported_model,
)


def load_config(path):
    """The config SGLang builds the model's layers from."""
    try:
        from sglang.srt.utils.hf_transformers_utils import get_config

        cfg = get_config(path, trust_remote_code=True)
    except ImportError:
        from sglang.srt.configs.model_config import ModelConfig

        cfg = ModelConfig(path, trust_remote_code=True).hf_config
    if not hasattr(cfg, "hidden_size") and hasattr(cfg, "get_text_config"):
        cfg = cfg.get_text_config()
    return cfg


def bad_value(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v * 2 if v else 1
    return v + "_other"


def warm_widths():
    """The normal-path warmup's token counts over two calls: 1..MAX_ROWS once."""
    import torch

    from sglang.srt.models.deepseek_common.amd import dsv41_mono_decode as m

    seen = []
    moe = types.SimpleNamespace(
        forward_normal=lambda x, return_moe_output: seen.append(x.shape[0])
    )
    sync, torch.cuda.synchronize = torch.cuda.synchronize, lambda device=None: None
    try:
        m._warm_normal_path(moe, torch.device("cpu"))
        m._warm_normal_path(moe, torch.device("cpu"))
    finally:
        torch.cuda.synchronize = sync
    return seen == list(range(1, m.MAX_ROWS + 1))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-V4.1-Flash"
    cfg = load_config(path)
    fails = []
    why = unsupported_model(cfg)
    print(f"checkpoint config ({type(cfg).__name__}): {why or 'supported'}")
    if why is not None:
        fails.append("checkpoint")
    base = {k: getattr(cfg, k, None) for k in _MODEL}
    cases = []
    for key, want in _MODEL.items():
        cases.append((f"{key} = {bad_value(want)!r}", {key: bad_value(want)}, key))
        cases.append((f"{key} missing", {key: None}, key))
    for name, change, key in cases:
        c = types.SimpleNamespace(**{**base, **change})
        why = unsupported_model(c)
        ok = why is not None and key in why
        print(f"  {'PASS' if ok else 'FAIL'} reject {name}: {why}")
        if not ok:
            fails.append(name)
    ok = warm_widths()
    print(
        f"  {'PASS' if ok else 'FAIL'} normal-path warmup runs 1..MAX_ROWS tokens once"
    )
    if not ok:
        fails.append("warmup")
    print(
        f"{len(cases)} negative cases + warmup; "
        + ("ALL PASS" if not fails else f"FAILED: {fails}")
    )
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
