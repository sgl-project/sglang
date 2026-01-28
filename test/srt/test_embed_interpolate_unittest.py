import unittest

import torch

from sglang.srt.configs.qwen3_vl import Qwen3VLConfig
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.quantization.unquant import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def unpack(tensor, dim_len, pack_len):
    dim_part = dim_len // pack_len
    ret_val = tensor.reshape(dim_part, dim_part, pack_len, pack_len, -1)
    ret_val = ret_val.permute(4, 0, 2, 1, 3).reshape(1, -1, dim_len, dim_len)
    return ret_val


class TestEmbedInterpolate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pDevice = torch.get_default_device()
        torch.set_default_device("npu")

    @classmethod
    def tearDownClass(cls):
        torch.set_default_device(cls.pDevice)

    def test_embed_interpolate(self):
        self.assertTrue(issubclass(UnquantizedLinearMethod, LinearMethodBase))
        t_dim = [16, 32]
        s_dim = [192, 574]
        sarg = ServerArgs(model_path="dummy", device="npu")
        mconf = Qwen3VLConfig(
            hidden_size=64,
            num_heads=1,
            num_position_embeddings=2304,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            deepstack_visual_indexes=[5, 11, 17],
            in_channels=3,
            depth=24,
            intermediate_size=256,
            hidden_act="gelu_pytorch_tanh",
            out_hidden_size=2560,
        )
        set_global_server_args_for_scheduler(sarg)
        init_distributed_environment(
            backend="gloo",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:2646",
        )
        initialize_model_parallel()
        initialize_dp_attention(
            server_args=sarg,
            model_config=mconf,
        )
        model = Qwen3VLMoeVisionModel(
            mconf,
            quant_config=None,
            norm_eps=1e-6,
            prefix="visual",
        )
        embeddings = model.fast_pos_embed_interpolate(
            [(t, s, s) for t, s in zip(t_dim, s_dim)]
        )

        embeddings_s0 = embeddings[: s_dim[0] * s_dim[0], :]
        embeddings_s1 = embeddings[s_dim[0] * s_dim[0] : 2 * s_dim[0] * s_dim[0], :]
        self.assertTrue(torch.allclose(embeddings_s0, embeddings_s1, atol=5e-5))

        embeddings_l = embeddings[
            t_dim[0] * s_dim[0] * s_dim[0] : t_dim[0] * s_dim[0] * s_dim[0]
            + s_dim[1] * s_dim[1],
            :,
        ]
        embeddings_s0 = torch.nn.functional.interpolate(
            unpack(embeddings_s0, s_dim[0], 2),
            size=(48, 48),
            mode="area",
        )
        embeddings_r = torch.nn.functional.interpolate(
            unpack(embeddings_l, s_dim[1], 2),
            size=(48, 48),
            mode="area",
        )
        self.assertTrue(
            torch.allclose(embeddings_s0, embeddings_r, atol=5e-1, rtol=5e-1)
        )


if __name__ == "__main__":
    unittest.main()
