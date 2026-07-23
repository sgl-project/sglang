import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.model_runner_components.misc_utils import (
    resolve_pp_proxy_residual_num_blocks,
)
from sglang.srt.model_executor.runner_utils.buffers import DecodeInputBuffers
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPProxyResidualShape(unittest.TestCase):
    @staticmethod
    def _model_config(*, architecture="KimiK3ForConditionalGeneration", block_size=6):
        return SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture]),
            hf_text_config=SimpleNamespace(attn_res_block_size=block_size),
        )

    @staticmethod
    def _decode_buffers(*, residual_num_blocks=None):
        return DecodeInputBuffers.create(
            device=torch.device("cpu"),
            max_bs=4,
            max_num_token=8,
            hidden_size=16,
            next_token_logits_buffer=torch.zeros((8, 32)),
            dtype=torch.float32,
            dp_size=1,
            pp_size=2,
            is_encoder_decoder=False,
            require_mlp_tp_gather=False,
            seq_len_fill_value=1,
            encoder_len_fill_value=0,
            num_tokens_per_req=2,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            pp_proxy_residual_num_blocks=residual_num_blocks,
        )

    def test_kimi_k3_later_stage_uses_inherited_bank_width(self):
        num_blocks = resolve_pp_proxy_residual_num_blocks(
            model_config=self._model_config(),
            pp_size=2,
            pp_rank=1,
            start_layer=46,
        )
        self.assertEqual(num_blocks, 8)

    def test_first_stage_keeps_default_residual_shape(self):
        num_blocks = resolve_pp_proxy_residual_num_blocks(
            model_config=self._model_config(),
            pp_size=2,
            pp_rank=0,
            start_layer=0,
        )
        self.assertIsNone(num_blocks)

    def test_other_models_keep_default_residual_shape(self):
        num_blocks = resolve_pp_proxy_residual_num_blocks(
            model_config=self._model_config(architecture="KimiLinearForCausalLM"),
            pp_size=2,
            pp_rank=1,
            start_layer=16,
        )
        self.assertIsNone(num_blocks)

    def test_decode_buffer_allocates_kimi_k3_residual_bank(self):
        buffers = self._decode_buffers(residual_num_blocks=8)
        self.assertEqual(buffers.pp_proxy_tensors["residual"].shape, (8, 8, 16))

    def test_decode_buffer_keeps_other_models_on_batch_sized_residual(self):
        buffers = self._decode_buffers()
        self.assertEqual(buffers.pp_proxy_tensors["residual"].shape, (4, 16))


if __name__ == "__main__":
    unittest.main()
