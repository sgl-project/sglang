# SPDX-License-Identifier: Apache-2.0
"""Opt-in two-NPU H3 comparison with the same online MXFP8 linear weights.

Set SGLANG_TEST_H3_BF16_MODEL_PATH to an unquantized checkpoint to enable.
Every case uses --quantization mxfp8 for online weight quantization on NPU.
SD cases select only fa + precision and exercise automatic H3 text-encoder
routing. FP8 defaults to HIGH_PRECISION; only the alternate mode overrides it.
Pre-quantized checkpoints require their own loader and are not used here.
VAE tiles run locally to avoid depending on an unrelated NPU gather workaround.
The existing FA baseline uses BF16 attention: arbitrary H3 packed boundaries
need a separate fix before the existing backend can serve as an MXFP8 baseline.
These cases enable baseline-generation mode for collection without existing
performance thresholds. The environment is restored after this test class.
"""

import json
import os
import shlex
from dataclasses import replace
from pathlib import Path

import pytest

from sglang.multimodal_gen.test.server.ascend.testcase_configs_npu import TWO_NPU_CASES
from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)

MODEL_PATH = os.environ.get("SGLANG_TEST_H3_BF16_MODEL_PATH")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set SGLANG_TEST_H3_BF16_MODEL_PATH to an unquantized H3 checkpoint",
)


def _cases():
    base = next(case for case in TWO_NPU_CASES if case.id == "minimax_h3_t2va_2npu")
    cases = []
    for ulysses in (False, True):
        for name, backend, config in (
            ("fa_bf16", "fa", {}),
            ("sd_fp8", "fa", {"precision": "fp8"}),
            (
                "sd_fp8_c8v16",
                "fa",
                {"precision": "fp8", "fp8_fa_mode": "C8V16_TILING512"},
            ),
            ("sd_mxfp8", "fa", {"precision": "mxfp8"}),
            ("sd_mxfp4", "fa", {"precision": "mxfp4"}),
        ):
            server_args = replace(
                base.server_args,
                model_path=MODEL_PATH or base.server_args.model_path,
                tp_size=1 if ulysses else 2,
                ulysses_degree=2 if ulysses else 1,
                ring_degree=1,
                enable_cache_dit=False,
                extras=[
                    "--model-variant",
                    "fl2va",
                    "--performance-mode",
                    "manual",
                    "--dit-cpu-offload",
                    "false",
                    "--sp-degree",
                    "2" if ulysses else "1",
                    "--vae-config.use-parallel-tiling",
                    "false",
                    "--quantization",
                    "mxfp8",
                    "--attention-backend",
                    backend,
                    "--attention-backend-config",
                    shlex.quote(json.dumps(config)),
                    # Match the SD cases' automatic text-encoder default in the
                    # floating-point baseline as well.
                    *(
                        ["--component-attention-backends", "text_encoder=torch_sdpa"]
                        if name == "fa_bf16"
                        else []
                    ),
                    "--component-residency",
                    "text_encoder=layerwise-offload",
                ],
                env_vars={
                    "SGLANG_CACHE_DIT_ENABLED": "false",
                    "SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION": "false",
                    "SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS": "1",
                    "HCCL_BUFFSIZE": "256",
                },
            )
            cases.append(
                replace(
                    base,
                    id=f"h3_quant_{name}_{'ulysses2' if ulysses else 'tp2'}",
                    server_args=server_args,
                    run_perf_check=False,
                    run_consistency_check=False,
                    perf_repeat_requests=3,
                    perf_warmup_requests=1,
                )
            )
    return cases


class TestAscendQuantFAH3(DiffusionServerBase):
    case = diffusion_case_fixture(_cases())

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def _collect_baselines(cls):
        # Keep collection enabled through each server fixture's final summary.
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("SGLANG_GEN_BASELINE", "1")
            yield

    def run_and_collect(self, *args, **kwargs):
        record, content = super().run_and_collect(*args, **kwargs)
        self._output_index += 1
        output = self._output_dir / f"{self._case_id}_{self._output_index}.mp4"
        output.write_bytes(content)
        return record, content

    def test_diffusion_generation(self, case, diffusion_server, tmp_path):
        self._case_id = case.id
        self._output_index = 0
        self._output_dir = Path(
            os.environ.get("SGLANG_TEST_H3_OUTPUT_DIR", str(tmp_path))
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        super().test_diffusion_generation(case, diffusion_server)
        logs = diffusion_server.stdout_file.read_text(
            encoding="utf-8", errors="replace"
        )
        if "sd_" in case.id:
            assert "Using Ascend FA: provider=mindiesd," in logs
