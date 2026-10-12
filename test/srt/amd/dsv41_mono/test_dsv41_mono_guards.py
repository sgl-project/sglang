"""CPU test of the launches' guards (model, speculative, step mode, draft layers) and the normal MoE path warmup (python3, in the image)."""

import os
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


def spec_cases(cfg):
    """The deployment guard lets None / 'none' / DSpark (static verify) through and rejects the rest."""
    from sglang.srt.models.deepseek_common.amd import dsv41_mono_decode as m

    only, static, gpu = (
        "supports speculative decoding with DSpark only",
        "needs SGLANG_RAGGED_VERIFY_MODE=static with DSpark",
        "needs gfx950",
    )
    fails = []
    get_spec, gfx95 = m.get_spec, m.is_gfx95_supported
    ragged = os.environ.pop("SGLANG_RAGGED_VERIFY_MODE", None)
    # without a GPU the next check stops at "needs gfx950"
    m.is_gfx95_supported = lambda: False
    try:
        cases = [
            (None, None, gpu),
            ("none", None, gpu),
            ("NONE", None, gpu),
            ("DSPARK", None, gpu),
            ("dspark", "static", gpu),
            ("DSPARK", "compact", static),
            ("NEXTN", None, only),
            ("EAGLE", None, only),
            ("EAGLE3", None, only),
        ]
        for algo, mode, want in cases:
            if mode is None:
                os.environ.pop("SGLANG_RAGGED_VERIFY_MODE", None)
            else:
                os.environ["SGLANG_RAGGED_VERIFY_MODE"] = mode
            m.get_spec = lambda: types.SimpleNamespace(speculative_algorithm=algo)
            why = m._unsupported(cfg)
            ok = why == want
            print(
                f"  {'PASS' if ok else 'FAIL'} speculative_algorithm={algo!r} ragged={mode!r}: {why}"
            )
            if not ok:
                fails.append(f"speculative {algo} {mode}")
    finally:
        m.get_spec, m.is_gfx95_supported = get_spec, gfx95
        os.environ.pop("SGLANG_RAGGED_VERIFY_MODE", None)
        if ragged is not None:
            os.environ["SGLANG_RAGGED_VERIFY_MODE"] = ragged
    return fails, len(cases)


def step_cases():
    """The FFN launch (and K1 through it) takes decode and target-verify steps of 1..MAX_ROWS rows, nothing else."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.deepseek_common.amd import dsv41_mono_decode as m

    ffn, k1 = object(), object()
    layer = types.SimpleNamespace(mono_ffn=ffn, mono_k1=k1)
    fails, n = [], 0
    for mode in ForwardMode:
        takes = mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY)
        for rows in (0, 1, 4, 6, 12, 48, 49):
            fb = types.SimpleNamespace(forward_mode=mode)
            want = takes and 1 <= rows <= m.MAX_ROWS
            pending = (
                object(),
                types.SimpleNamespace(shape=(rows,)),
                object(),
                object(),
            )
            got = (
                m.ffn_launch(layer, fb, rows) is ffn,
                m.k1_launch(layer, fb, pending, object()) is k1,
            )
            n += 1
            if got != (want, want):
                fails.append(f"{mode.name} rows={rows}: {got}")
    print(
        f"  {'PASS' if not fails else 'FAIL'} launches by forward mode x rows ({n} cases): {fails or 'decode / target verify, 1..48 rows'}"
    )
    return fails, n


def draft_layer_cases(cfg):
    """init_layer leaves draft layers (mlp.is_nextn) on the normal path and still checks target layers."""
    from sglang.srt.models.deepseek_common.amd import dsv41_mono_decode as m

    fails = []
    old, gfx95, get_spec = (
        os.environ.get("SGLANG_ROCM_MONO_DECODE"),
        m.is_gfx95_supported,
        m.get_spec,
    )
    os.environ["SGLANG_ROCM_MONO_DECODE"] = "1"
    m.is_gfx95_supported = lambda: False
    m.get_spec = lambda: types.SimpleNamespace(speculative_algorithm="DSPARK")
    try:
        draft = types.SimpleNamespace(
            mlp=types.SimpleNamespace(is_nextn=True), config=None
        )
        m.init_layer(draft)
        ok = draft.mono_ffn is None and draft.mono_k1 is None
        print(f"  {'PASS' if ok else 'FAIL'} draft layer: no launches, no check")
        if not ok:
            fails.append("draft layer")
        target = types.SimpleNamespace(
            mlp=types.SimpleNamespace(is_nextn=False), config=cfg
        )
        try:
            m.init_layer(target)
            why = None
        except ValueError as e:
            why = str(e)
        ok = why == "SGLANG_ROCM_MONO_DECODE needs gfx950."
        print(f"  {'PASS' if ok else 'FAIL'} target layer still checked: {why}")
        if not ok:
            fails.append("target layer")
    finally:
        m.is_gfx95_supported, m.get_spec = gfx95, get_spec
        if old is None:
            os.environ.pop("SGLANG_ROCM_MONO_DECODE", None)
        else:
            os.environ["SGLANG_ROCM_MONO_DECODE"] = old
    return fails


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
    f, ns = spec_cases(cfg)
    fails += f
    f, nm = step_cases()
    fails += f
    fails += draft_layer_cases(cfg)
    print(
        f"{len(cases)} negative cases + warmup + {ns} speculative + {nm} step + 2 layer cases; "
        + ("ALL PASS" if not fails else f"FAILED: {fails}")
    )
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
