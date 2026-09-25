"""MI455x (gfx1250) output-quality gate: the CI's Wan2.2-T2V-A14B case on one GPU.

The shared diffusion lanes already know how to judge output quality: after
generating, ``DiffusionServerBase`` extracts the first/middle/last frame and
scores them against a pinned ground truth on CLIP similarity, SSIM, PSNR and
mean absolute pixel difference (``_validate_consistency`` in
``multimodal_gen/test/server/test_server_common.py``). What the AMD lanes do not
do is run it -- ``pr-test-amd.yml`` and ``pr-test-amd-rocm720.yml`` hand the
suite ``SGLANG_SKIP_CONSISTENCY=1``, so ROCm today is only checked for "did it
produce bytes", not "are those bytes still the same video".

This file turns that check back on for one configuration. The case is the CI's
own ``wan2_2_t2v_a14b_2gpu`` from ``server/gpu_cases.py`` -- same model, same
default T2V sampling params -- with two deliberate changes:

  * one GPU instead of two, so ``--ulysses-degree=2`` is dropped;
  * the ROCm environment the gfx1250 Wan2.2 runs are qualified under.

Keeping the sampling params identical to the CI case is the point: it makes a
gfx1250 result comparable to the numbers the h100 lane already publishes for
this model, instead of to a one-off configuration.

Ground truth is gfx1250's own output (see
``consistency_thresholds/gfx1250.json``), so a failure here means a kernel or
scheduling change moved the picture on this architecture -- not that AMD and
NVIDIA disagree, which they always will at the pixel level. It belongs under
``diffusion-ci/consistency_gt/sglang_generated/gfx1250`` in
sgl-project/ci-data-diffusion; until it is published there, point the run at a
local copy with ``SGLANG_CONSISTENCY_GT_DIR``.
``scripts/ci/amd/local/run_mi45x_diffusion_quality.sh`` drives both recording
and checking.
"""

from __future__ import annotations

import logging
import os

import pytest

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionServerArgs,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST,
)
from sglang.test.ci.ci_register import register_amd_ci

logger = logging.getLogger(__name__)

# Model load off a cold page cache dominates; the generation itself is the CI
# default T2V shape, not a long one.
register_amd_ci(
    est_time=2400,
    suite="nightly-amd-1-gpu-mi45x-diffusion-quality",
    nightly=True,
)

# Local mirrors of the weights, in the layouts the MI455x boxes use. Checked
# before the HF id because the hub cache on these machines holds only the
# metadata for this repo -- the real 118 GB copy lives on disk.
MODEL_ROOTS = ("/dockerx/data/models", "/data/models")
MODEL_DIRNAME = "Wan2.2-T2V-A14B-Diffusers"

# The ROCm environment the gfx1250 Wan2.2 runs are qualified under. These are
# not tuning knobs the test is free to drop: FlyDSL layernorm and AITER FP8
# attention are off because they are not yet trusted on this architecture, and
# the VAE settings decide the decode path that produces the frames being scored,
# so changing any of them invalidates the ground truth.
GFX1250_WAN22_ENV = {
    # gfx1250 reports compute capability (12, 5); without this the diffusion
    # platform layer can resolve to the CUDA backend.
    "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "rocm",
    # Inductor autotuning is both slow to warm and non-deterministic in which
    # kernel it picks, which shows up directly in the frames.
    "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
    "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "0",
    # gfx1250 has no CK kernels yet; AITER must take the non-CK path.
    "ENABLE_CK": "0",
    "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "1",
    "SGLANG_USE_ROCM_VAE_CONV2D_BF16": "1",
    "SGLANG_USE_ROCM_FLYDSL": "0",
    "SGLANG_DIFFUSION_AITER_FP8_ATTN": "0",
    "PYTORCH_NO_HIP_MEMORY_CACHING": "1",
    "HIP_FORCE_DEV_KERNARG": "0",
}


def _resolve_model_path() -> str:
    override = os.environ.get("SGLANG_MI45X_MODEL_ROOT")
    roots = (override, *MODEL_ROOTS) if override else MODEL_ROOTS
    for root in roots:
        candidate = os.path.join(root, MODEL_DIRNAME)
        if os.path.isdir(candidate):
            return candidate
    return DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST


QUALITY_CASES = [
    DiffusionTestCase(
        # The GT filenames the harness derives from this id already carry the
        # GPU count ("..._1gpu_frame_0.png"), so the id must not repeat it.
        "wan2_2_t2v_a14b_gfx1250",
        DiffusionServerArgs(
            model_path=_resolve_model_path(),
            modality="video",
            num_gpus=1,
            # Both A14B towers (~26 GB each) sit resident in the ~430 GB a
            # gfx1250 reports, so the weights never need to stream from host.
            dit_layerwise_offload=False,
            extras=["--enable-torch-compile false"],
            env_vars=GFX1250_WAN22_ENV,
        ),
        # sampling_params=None inherits the default T2V params, which is exactly
        # what the CI's wan2_2_t2v_a14b_2gpu case uses. Do not pin a different
        # prompt or resolution here: that would make the numbers incomparable to
        # the h100 lane's for the same model.
        run_perf_check=False,
        run_consistency_check=True,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
]


@pytest.mark.skipif(
    not current_platform.is_hip(),
    reason="gfx1250 Wan2.2 quality gate is ROCm-only",
)
class TestWan22T2VA14BQualityMI45x(DiffusionServerBase):
    """Scores a single-GPU gfx1250 Wan2.2 run against pinned gfx1250 frames."""

    case = diffusion_case_fixture(QUALITY_CASES)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
