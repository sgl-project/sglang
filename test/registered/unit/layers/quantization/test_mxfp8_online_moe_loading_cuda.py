"""GPU parity checks for chunked online MXFP8 MoE weight loading."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.test_utils import CustomTestCase

ROWS = 65
K = 256
TEST_CHUNK_BYTES = 4096


def _backend(*, flashinfer_trtllm: bool):
    backend = MagicMock()
    backend.is_flashinfer_trtllm.return_value = flashinfer_trtllm
    backend.is_flashinfer_trtllm_routed.return_value = False
    return backend


class TestMxfp8OnlineMoeLoadingCuda(CustomTestCase):
    def _run_parity_case(self, *, flashinfer_trtllm: bool):
        from sglang.srt.layers.quantization import fp8 as fp8_quant
        from sglang.srt.layers.quantization import fp8_utils

        torch.manual_seed(42)
        source = torch.randn(ROWS, K, dtype=torch.bfloat16)

        if flashinfer_trtllm:
            expected_q, expected_scale = fp8_utils.flashinfer_mxfp8_quantize(
                source.cuda(), False
            )
        else:
            expected_q, expected_scale = fp8_quant.mxfp8_group_quantize(source.cuda())
        expected_q = expected_q.cpu().view(torch.uint8)
        expected_scale = expected_scale.view(torch.uint8).cpu().view(ROWS, K // 32)

        weight_param = torch.nn.Parameter(
            torch.empty(ROWS, K, dtype=torch.float8_e4m3fn, device="cuda"),
            requires_grad=False,
        )
        scale_param = torch.nn.Parameter(
            torch.empty(ROWS, K // 32, dtype=torch.uint8, device="cuda"),
            requires_grad=False,
        )
        layer = SimpleNamespace(
            w13_weight=weight_param,
            w2_weight=object(),
            w13_weight_scale_inv=scale_param,
            w2_weight_scale_inv=scale_param,
            _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
        )
        calls = []

        def original_weight_loader(
            param, loaded_weight, *, weight_name, shard_id, expert_id
        ):
            calls.append(
                (param, loaded_weight.clone(), weight_name, shard_id, expert_id)
            )

        with (
            patch.object(
                fp8_quant,
                "get_moe_runner_backend",
                return_value=_backend(flashinfer_trtllm=flashinfer_trtllm),
            ),
            patch.object(
                fp8_quant,
                "_ONLINE_MXFP8_MAX_CHUNK_BYTES",
                TEST_CHUNK_BYTES,
            ),
        ):
            loader = fp8_quant.Fp8MoEMethod.get_online_mxfp8_weight_loader(
                layer, original_weight_loader
            )
            if flashinfer_trtllm:
                quantizer = fp8_utils.flashinfer_mxfp8_quantize
                quantizer_patch = patch.object(
                    fp8_utils,
                    "flashinfer_mxfp8_quantize",
                    wraps=quantizer,
                )
            else:
                quantizer = fp8_quant.mxfp8_group_quantize
                quantizer_patch = patch.object(
                    fp8_quant,
                    "mxfp8_group_quantize",
                    wraps=quantizer,
                )
            with quantizer_patch as chunk_quantize:
                loader(
                    weight_param,
                    source,
                    weight_name="experts.0.up_proj.weight",
                    shard_id="w3",
                    expert_id=0,
                )

        self.assertGreater(chunk_quantize.call_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0][1].view(torch.uint8).cpu(), expected_q))
        self.assertTrue(torch.equal(calls[1][1].cpu(), expected_scale))
        self.assertIs(calls[1][0], scale_param)
        self.assertEqual(calls[1][2], "experts.0.up_proj.weight_scale_inv")

    def test_flashinfer_chunking_matches_whole_tensor_quantization(self):
        self._run_parity_case(flashinfer_trtllm=True)

    def test_triton_chunking_matches_whole_tensor_quantization(self):
        self._run_parity_case(flashinfer_trtllm=False)


if __name__ == "__main__":
    unittest.main()
