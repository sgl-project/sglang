"""The Mega backend inherits the base backend's KV layout dispatch state."""

import unittest
from functools import partial
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "the Mega backend fixture needs CUDA SM100/SM103",
)
class TestDeepseekV4MegaLayoutDispatch(CustomTestCase):
    def test_mega_backend_preserves_inherited_layout_dispatch(self):
        """Removing the base constructor's interleaving flag broke both Mega
        entry points although direct FlashMLA kernel tests still passed."""
        from sglang.kernels.ops.attention import flash_mla_interface
        from sglang.srt.layers.attention.deepseek_v4_mega_backend import (
            DeepseekV4MegaAttnBackend,
        )
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.test.kits.attention_unittest.attention_methods import (
            dsv4_attention as fixtures,
        )

        case = fixtures.DSV4AttentionCase(
            name="mega_layout_dispatch",
            backend="flashmla",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=256,
            prefix_lens=(1,),
        )
        # Only the external kernels are replaced. The real constructor and KV
        # pool must supply the inherited state; the test never sets that state.
        outputs = {legacy: (object(), object()) for legacy in (False, True)}

        def decode(*args, use_legacy, **kwargs):
            return *outputs[use_legacy], None

        def prefill(*args, use_legacy, **kwargs):
            return *outputs[use_legacy], None, None

        for layout, legacy in ((KVLayout.V41, True), (KVLayout.V41_NEW, False)):
            with (
                self.subTest(layout=layout),
                # Restore the outer context even if fixture construction fails
                # after installing its own override but before returning.
                get_context().override_server_args(),
                patch.object(
                    fixtures,
                    "DeepSeekV4TokenToKVPool",
                    partial(fixtures.DeepSeekV4TokenToKVPool, kv_layout=layout),
                ),
            ):
                runner = fixtures.MockDSV4ModelRunner(
                    case=case,
                    model_config=fixtures.TinyDSV4ModelConfig(
                        num_heads=64, context_len=256
                    ),
                    dtype=torch.bfloat16,
                    device="cuda",
                    max_context_len=256,
                    swa_size=512,
                )
                try:
                    backend = DeepseekV4MegaAttnBackend(runner)
                    with (
                        patch.object(
                            flash_mla_interface, "flash_mla_mega_decode", decode
                        ),
                        patch.object(
                            flash_mla_interface, "flash_mla_mega_prefill", prefill
                        ),
                    ):
                        self.assertEqual(
                            backend._forward_mega_decode(
                                q=None,
                                swa_k_cache=None,
                                swa_page_indices=None,
                                positions=None,
                                cos_sin_cache=None,
                                attn_sink=None,
                                swa_topk_lengths=None,
                                extra_k_cache=None,
                                extra_indices=None,
                                extra_topk_lengths=None,
                            ),
                            outputs[legacy],
                        )
                        self.assertEqual(
                            backend._forward_mega_prefill(
                                q=None,
                                kv=None,
                                indices=None,
                                positions=None,
                                cos_sin_cache=None,
                                attn_sink=None,
                                topk_length=None,
                            ),
                            outputs[legacy],
                        )
                finally:
                    runner._server_args_override.restore()


if __name__ == "__main__":
    unittest.main()
