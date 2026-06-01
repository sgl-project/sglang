"""Two-pass devbox driver around test_nightly_precision_regression._test_one_model.

Set SGLANG_PRECISION_HF_REPO + HF_TOKEN to exercise the HF fetch/push path
through _test_one_model itself — no extra push logic lives here, so verify
runs exactly the same code as CI.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "python"))
sys.path.insert(0, str(_HERE / "test"))
sys.path.insert(0, str(_HERE))

from registered.debug_utils import test_nightly_precision_regression as t  # noqa: E402

from sglang.test import precision_baseline_store as hfs  # noqa: E402
from sglang.test.test_utils import ModelLaunchSettings  # noqa: E402

MODEL = os.environ.get("VERIFY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TP_SIZE = int(os.environ.get("VERIFY_TP_SIZE", "1"))
BASELINE_DIR = Path(
    os.environ.get("VERIFY_BASELINE_DIR", str(_HERE / "run_logs/baselines_mini"))
)
BASE_URL = os.environ.get("VERIFY_BASE_URL", "http://127.0.0.1:11000")
DIFF_THRESHOLD = float(os.environ.get("VERIFY_DIFF_THRESHOLD", "1e-3"))
RESET_BASELINE = os.environ.get("VERIFY_RESET_BASELINE", "0") == "1"


def main() -> int:
    hf_cfg = hfs.HfStoreConfig.from_env()
    print(
        f"[verify] MODEL={MODEL} TP={TP_SIZE} BASELINE_DIR={BASELINE_DIR}", flush=True
    )
    print(f"[verify] HF repo={hf_cfg.repo}", flush=True)

    if RESET_BASELINE and BASELINE_DIR.exists():
        shutil.rmtree(BASELINE_DIR)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    model_setup = ModelLaunchSettings(MODEL, tp_size=TP_SIZE)

    # Delegate fetch/push entirely to _test_one_model's internal _maybe_hf_*
    # path so verify exercises the same code that runs in CI.
    print("[verify] ==== pass 1 ====", flush=True)
    pass1 = t._test_one_model(
        model_setup=model_setup,
        baseline_dir=BASELINE_DIR,
        diff_threshold=DIFF_THRESHOLD,
        force_update=False,
        base_url=BASE_URL,
        hf_cfg=hf_cfg,
    )
    print(f"[verify] pass1 result: {pass1}", flush=True)

    print("[verify] ==== pass 2 ====", flush=True)
    pass2 = t._test_one_model(
        model_setup=model_setup,
        baseline_dir=BASELINE_DIR,
        diff_threshold=DIFF_THRESHOLD,
        force_update=False,
        base_url=BASE_URL,
        hf_cfg=hf_cfg,
    )
    print(f"[verify] pass2 result: {pass2}", flush=True)

    print("[verify] ==== summary ====", flush=True)
    print(f"  pass1: {pass1[1]} ({pass1[2]})", flush=True)
    print(f"  pass2: {pass2[1]} ({pass2[2]})", flush=True)

    ok = pass1[1] in ("BASELINE_ESTABLISHED", "PASSED") and pass2[1] == "PASSED"
    print(f"[verify] overall: {'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
