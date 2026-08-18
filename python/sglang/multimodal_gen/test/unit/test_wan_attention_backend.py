import unittest
from unittest.mock import patch

from torch import nn

from sglang.multimodal_gen.runtime.layers.attention.roles import AttentionRole
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanSelfAttention
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_WAN = "sglang.multimodal_gen.runtime.models.dits.wanvideo"


class TestWanAttentionBackendRole(unittest.TestCase):
    def _build(self, *, is_cross_attention: bool):
        with (
            patch(f"{_WAN}.ColumnParallelLinear", return_value=nn.Identity()),
            patch(f"{_WAN}.RowParallelLinear", return_value=nn.Identity()),
            patch(f"{_WAN}.get_tp_world_size", return_value=1),
            patch(f"{_WAN}.USPAttention") as usp_attention,
        ):
            WanSelfAttention(
                dim=128,
                num_heads=1,
                qk_norm=False,
                is_cross_attention=is_cross_attention,
                supported_attention_backends={
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.TORCH_SDPA,
                },
            )
        return usp_attention

    def test_cross_attention_role_is_forwarded_to_usp(self):
        usp_attention = self._build(is_cross_attention=True)

        self.assertEqual(
            usp_attention.call_args.kwargs["attention_role"], AttentionRole.CROSS
        )
        self.assertTrue(usp_attention.call_args.kwargs["skip_sequence_parallel"])

    def test_self_attention_role_is_forwarded_to_usp(self):
        usp_attention = self._build(is_cross_attention=False)

        self.assertEqual(
            usp_attention.call_args.kwargs["attention_role"], AttentionRole.SELF
        )
        self.assertFalse(usp_attention.call_args.kwargs["skip_sequence_parallel"])


if __name__ == "__main__":
    unittest.main()
