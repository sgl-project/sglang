import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common import deepseek_weight_loader as loader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGlmDcpKvBDtype(unittest.TestCase):
    def test_opt_in_retains_bf16_only_for_glm_dcp_on_gfx95(self):
        for gfx95 in (False, True):
            for architecture in (None, "DeepseekV3ForCausalLM", "GlmMoeDsaForCausalLM"):
                for dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
                    for dcp in (1, 4):
                        for keep in (False, True):
                            with (
                                self.subTest(
                                    gfx95=gfx95,
                                    arch=architecture,
                                    dtype=dtype,
                                    dcp=dcp,
                                    keep=keep,
                                ),
                                patch.object(loader, "_use_aiter_gfx95", gfx95),
                                patch.object(
                                    loader,
                                    "get_parallel",
                                    return_value=SimpleNamespace(attn_dcp_size=dcp),
                                ),
                                envs.SGLANG_DSA_DCP_KEEP_KV_B_BF16.override(keep),
                            ):
                                quantize = loader._quantize_glm_kv_b_to_fp8(
                                    architecture=architecture, w_dtype=dtype
                                )
                                original = (
                                    gfx95
                                    and architecture == "GlmMoeDsaForCausalLM"
                                    and dtype == torch.bfloat16
                                )
                                self.assertEqual(
                                    quantize, original and not (keep and dcp > 1)
                                )


if __name__ == "__main__":
    unittest.main()
