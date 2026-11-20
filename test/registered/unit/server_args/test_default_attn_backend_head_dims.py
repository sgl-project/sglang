"""Default MHA attention backend vs. the head dims FlashInfer's FA2 prefill handles.

Outside Hopper and SM100 (Ampere, Ada, SM120), `get_default_attn_backend` picks
flashinfer for MHA models, and SGLang's flashinfer backend prefills with FA2.
FA2 handles only head dims that are multiples of 64, and a vo dim above 256 only
in multiples of 256. Other dims read out of bounds or return wrong results, and
FlashInfer 0.7.5 rejects them in plan() (flashinfer-ai/flashinfer#4312). Models
with such head dims default to triton instead.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import model_override_base as base_module
from sglang.srt.arg_groups.model_override_base import get_default_attn_backend
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _model_config(
    head_dim, v_head_dim=None, *, swa_head_dim=None, has_attention_sinks=False
):
    v_head_dim = head_dim if v_head_dim is None else v_head_dim
    swa_head_dim = head_dim if swa_head_dim is None else swa_head_dim
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        swa_head_dim=swa_head_dim,
        swa_v_head_dim=swa_head_dim,
        has_attention_sinks=has_attention_sinks,
    )


class TestDefaultAttnBackendHeadDims(CustomTestCase):
    def _default(self, model_config, *, has_flashinfer=True):
        # An Ampere/Ada/SM120 CUDA host: none of the fa3 / trtllm_mha / aiter arms apply.
        server_args = SimpleNamespace(
            speculative_algorithm=None, speculative_eagle_topk=None
        )
        with (
            override_platform(
                is_sail=False,
                is_hopper_with_cuda_12_3=False,
                is_sm100=False,
                is_hip=False,
                has_flashinfer=has_flashinfer,
            ),
            patch.object(base_module, "is_mps", return_value=False),
        ):
            return get_default_attn_backend(
                server_args, use_mla_backend=False, model_config=model_config
            )

    def test_supported_head_dims_keep_flashinfer(self):
        for head_dim in (64, 128, 256, 512):
            with self.subTest(head_dim=head_dim):
                self.assertEqual(self._default(_model_config(head_dim)), "flashinfer")

    def test_unsupported_head_dims_fall_back_to_triton(self):
        cases = {
            "72 (wrong results upstream)": (72, 72),
            "80 (Phi-2)": (80, 80),
            "96 (Phi-3-mini)": (96, 96),
            "vo not a multiple of 64": (128, 96),
            "vo above 256 not a multiple of 256": (320, 320),
        }
        for label, (qk, vo) in cases.items():
            with self.subTest(label):
                self.assertEqual(self._default(_model_config(qk, vo)), "triton")

    def test_swa_layer_head_dims_are_checked(self):
        self.assertEqual(self._default(_model_config(128, swa_head_dim=96)), "triton")

    def test_without_flashinfer_uses_triton(self):
        self.assertEqual(
            self._default(_model_config(128), has_flashinfer=False), "triton"
        )

    def test_attention_sinks_still_use_triton(self):
        self.assertEqual(
            self._default(_model_config(128, has_attention_sinks=True)), "triton"
        )


if __name__ == "__main__":
    unittest.main()
