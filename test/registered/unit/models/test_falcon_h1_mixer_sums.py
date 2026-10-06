import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.models import falcon_h1
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Built(Exception):
    """Stops the layer's construction once its mixers are built."""


class TestFalconH1MixersLeavePartialSums(CustomTestCase):
    """The attention and Mamba mixers read the same input and their outputs are
    summed before the FFN; prepare_mlp completes that sum over attention TP.
    Neither mixer may reduce its own output first, or its share of the sum is
    counted tp times."""

    def test_neither_mixer_reduces_its_output(self):
        config = SimpleNamespace(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rope_parameters={"rope_theta": 10000.0},
            max_position_embeddings=128,
            mamba_d_ssm=None,
            mamba_expand=2,
            mamba2_cache_params=None,
            mamba_conv_bias=True,
            mamba_proj_bias=False,
            mamba_n_groups=1,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            mamba_rms_norm=False,
        )
        linears = []

        def row_parallel(*args, **kwargs):
            linears.append(kwargs)
            return MagicMock()

        mixer = MagicMock(side_effect=_Built)
        for tp in (1, 2):
            with self.subTest(tp=tp):
                linears.clear()
                mixer.reset_mock()
                with (
                    get_parallel().override(
                        attn_tp_rank=0, attn_tp_size=tp, tp_size=tp
                    ),
                    patch.object(falcon_h1, "get_rope", MagicMock()),
                    patch.object(falcon_h1, "QKVParallelLinear", MagicMock()),
                    patch.object(falcon_h1, "RowParallelLinear", row_parallel),
                    patch.object(falcon_h1, "RadixAttention", MagicMock()),
                    patch.object(falcon_h1, "MambaMixer2", mixer),
                ):
                    with self.assertRaises(_Built):
                        falcon_h1.FalconH1HybridAttentionDecoderLayer(
                            config=config, layer_id=0
                        )
                self.assertEqual([k.get("reduce_results") for k in linears], [False])
                self.assertIs(mixer.call_args.kwargs.get("reduce_results"), False)


if __name__ == "__main__":
    unittest.main()
