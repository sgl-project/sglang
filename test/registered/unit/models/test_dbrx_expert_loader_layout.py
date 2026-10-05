"""DBRX expert checkpoint shards follow the constructed TP layout."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.dbrx import DbrxExperts
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return get_parallel().override(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


def build_experts(dtype=torch.bfloat16):
    config = SimpleNamespace(
        d_model=32,
        ffn_config=SimpleNamespace(moe_num_experts=4, moe_top_k=2, ffn_hidden_size=128),
    )
    module = DbrxExperts(config, 0, params_dtype=dtype)
    assert isinstance(module.router.layer, ReplicatedLinear)
    assert isinstance(module.router.layer.quant_method, UnquantizedLinearMethod)
    return module


def values(shape, dtype, offset=0):
    count = 1
    for size in shape:
        count *= size
    return (
        (((torch.arange(count, device="cuda") + offset) % 31 - 15) / 128)
        .to(dtype)
        .reshape(shape)
    )


def load_experts(module, *, changed=False, flat=False, offset=0):
    rank = get_parallel().tp_rank
    start = rank * module.intermediate_size
    stop = start + module.intermediate_size
    loaded = {}
    for index, name in enumerate(("w1", "v1", "w2")):
        full = values((4, 128, 32), module.ws.dtype, offset + index * 7)
        param = module.w2s if name == "w2" else module.ws
        with loading_scope(changed):
            param.weight_loader(param, full.flatten() if flat else full, name)
        loaded[name] = full[:, start:stop, :]
    expected_ws = torch.cat((loaded["w1"], loaded["v1"]), dim=1)
    expected_w2 = loaded["w2"].transpose(1, 2)
    torch.testing.assert_close(module.ws, expected_ws, rtol=0, atol=0)
    torch.testing.assert_close(module.w2s, expected_w2, rtol=0, atol=0)
    return expected_ws, expected_w2


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestDbrxExpertLoaderLayout(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def check_loads(self, changed):
        with torch.inference_mode():
            for size in (1, 2, 4):
                for rank in range(size):
                    reset_context()
                    publish(
                        ServerArgs(model_path="dummy", device="cuda", tp_size=size),
                        role="test",
                        ranks=SpawnRanks(world_rank=rank),
                    )
                    for dtype in (torch.float32, torch.bfloat16):
                        module = build_experts(dtype)
                        for flat in (False, True):
                            for offset in (0, 11):
                                load_experts(
                                    module,
                                    changed=changed,
                                    flat=flat,
                                    offset=offset,
                                )

    def test_native_loader_in_the_construction_scope(self):
        self.check_loads(False)

    def test_native_loader_after_scope_exit(self):
        self.check_loads(True)


if __name__ == "__main__":
    unittest.main()
