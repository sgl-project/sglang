"""Exercise the dependency-free H100 planner and report verifier on the same suite."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

_PLANNER_DIR = Path(__file__).resolve().parents[5] / "scripts/ci/utils/diffusion"


@pytest.mark.parametrize(
    "changed, deferred",
    [
        (None, set()),  # scheduled, release or manual full run
        (["runtime/models/dits/base.py"], set()),
        (["runtime/pipelines/wan_pipeline.py", "registry.py"], set()),
        (
            ["runtime/pipelines/wan_pipeline.py", "configs/sample/qwenimage.py"],
            {"flux_2_klein_base_image_t2i"},
        ),
        (
            ["runtime/models/dits/ltx_2.py"],
            {
                "flux_2_klein_base_image_t2i",
                "qwen_image_edit_ti2i",
                "wan2_1_t2v_14b_2gpu",
                "wan2_1_i2v_14b_480P_2gpu",
            },
        ),
    ],
)
def test_h100_plan_and_coverage_match_registered_work(
    tmp_path, monkeypatch, changed, deferred
):
    monkeypatch.syspath_prepend(str(_PLANNER_DIR))
    verifier = runpy.run_path(str(_PLANNER_DIR / "verify_diffusion_coverage.py"))
    repo_root = Path(__file__).resolve().parents[5]
    env = {**os.environ, "GITHUB_OUTPUT": str(tmp_path / "outputs")}
    env.pop("DIFFUSION_PR_FILES", None)
    if changed is not None:
        files = tmp_path / "files.json"
        files.write_text(
            json.dumps([f"python/sglang/multimodal_gen/{p}" for p in changed])
        )
        env["DIFFUSION_PR_FILES"] = str(files)
    subprocess.run(
        [sys.executable, str(_PLANNER_DIR / "compute_diffusion_partitions.py")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    outputs = dict(
        line.split("=", 1) for line in (tmp_path / "outputs").read_text().splitlines()
    )
    families = tuple(json.loads(outputs["h100-pr-families"]))
    expected = verifier["get_expected_cases"](repo_root, families)
    full = verifier["get_expected_cases"](repo_root)
    assert deferred <= set.union(*full.values())
    for suite, key in [("1-gpu", "plan-1gpu"), ("2-gpu", "plan-2gpu")]:
        parts = json.loads(outputs[key])["partitions"]
        scheduled = [case for part in parts for case in part["case_ids"]]
        standalone = [path for part in parts for path in part["standalone_files"]]
        assert expected[suite] == full[suite] - deferred
        assert len(scheduled) == len(set(scheduled))
        assert expected[suite] == set(scheduled) | {
            f"standalone:{p}" for p in standalone
        }
