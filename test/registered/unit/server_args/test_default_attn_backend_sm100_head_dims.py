"""SM100/SM103 default MHA attention backend vs. the head dims trtllm-gen ships.

On SM100/SM103, `get_default_attn_backend` picks `trtllm_mha`, which runs
trtllm-gen's paged-KV kernels. Those exist only for head dims 64, 128, 256 and
512 with equal K/V dims; any other dims fail with "Missing TRTLLM-GEN kernel" at
CUDA-graph capture or the first request (flashinfer-ai/flashinfer#9900). Those
models default to fa4 where it supports the dims, otherwise to triton.
Asymmetric K/V keeps going to fa4 (#32818).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.model_override_base import get_default_attn_backend
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _model_config(head_dim, v_head_dim=None, *, swa_head_dim=None):
    v_head_dim = head_dim if v_head_dim is None else v_head_dim
    swa_head_dim = head_dim if swa_head_dim is None else swa_head_dim
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        swa_head_dim=swa_head_dim,
        swa_v_head_dim=swa_head_dim,
        has_asymmetric_kv=head_dim != v_head_dim,
        has_attention_sinks=False,
    )


class TestSm100DefaultAttnBackendHeadDims(CustomTestCase):
    def _default(self, model_config):
        server_args = SimpleNamespace(
            speculative_algorithm=None, speculative_eagle_topk=None
        )
        with override_platform(
            is_sail=False,
            is_hopper_with_cuda_12_3=False,
            is_sm100=True,
            is_hip=False,
            has_flashinfer=True,
        ):
            return get_default_attn_backend(
                server_args, use_mla_backend=False, model_config=model_config
            )

    def test_trtllm_gen_head_dims_keep_trtllm_mha(self):
        for head_dim in (64, 128, 256, 512):
            with self.subTest(head_dim=head_dim):
                self.assertEqual(self._default(_model_config(head_dim)), "trtllm_mha")

    def test_other_head_dims_fall_back_to_fa4(self):
        cases = {
            "32 (bge-small, MiniLM)": 32,
            "80 (Phi-2; trtllm-gen has context but no decode kernels)": 80,
            "96 (Phi-3-mini)": 96,
            "112": 112,
        }
        for label, head_dim in cases.items():
            with self.subTest(label):
                self.assertEqual(self._default(_model_config(head_dim)), "fa4")

    def test_head_dims_neither_serves_fall_back_to_triton(self):
        # fa4 on SM100 takes 8..128 in steps of 8 (plus 192/128 and 256/256).
        cases = {"above 128": 160, "not a multiple of 8": 76}
        for label, head_dim in cases.items():
            with self.subTest(label):
                self.assertEqual(self._default(_model_config(head_dim)), "triton")

    def test_swa_layer_head_dims_are_checked(self):
        self.assertEqual(self._default(_model_config(128, swa_head_dim=96)), "fa4")

    def test_asymmetric_kv_still_uses_fa4(self):
        self.assertEqual(self._default(_model_config(192, 128)), "fa4")


if __name__ == "__main__":
    unittest.main()
