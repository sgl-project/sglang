"""Restore, reload and finalize of dense Marlin linears (GPTQ, AWQ) must reproduce a fresh load."""

import unittest
from contextlib import contextmanager

import torch

from sglang.srt.layers.quantization.awq.awq import AWQMarlinConfig
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig
from sglang.srt.layers.quantization.marlin_utils import (
    awq_to_marlin_zero_points,
    marlin_permute_scales,
    marlin_sort_g_idx,
    marlin_to_awq_weights,
    marlin_to_awq_zero_points,
    marlin_to_gptq_weights,
    marlin_unpermute_scales,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
# A merged gate/up projection followed by a down projection, as in a Qwen2 MLP.
HIDDEN, INTER, GROUP_SIZE, M = 512, 1024, 128, 16


def _rand_int32(gen, *shape):
    # any int32 packs eight valid int4 values
    return torch.randint(
        -(2**31), 2**31 - 1, shape, dtype=torch.int32, device=DEVICE, generator=gen
    )


def _rand_scales(gen, *shape):
    return (torch.rand(*shape, device=DEVICE, generator=gen) / 100 + 1e-3).to(
        torch.float16
    )


def _gptq_checkpoint(seed, desc_act):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    ckpt = {}
    for name, (k, n) in (
        ("gate_proj", (HIDDEN, INTER)),
        ("up_proj", (HIDDEN, INTER)),
        ("down_proj", (INTER, HIDDEN)),
    ):
        g_idx = torch.arange(k, device=DEVICE, dtype=torch.int32) // GROUP_SIZE
        if desc_act:
            g_idx = g_idx[torch.randperm(k, device=DEVICE, generator=gen)]
        ckpt[name] = {
            "qweight": _rand_int32(gen, k // 8, n),
            "scales": _rand_scales(gen, k // GROUP_SIZE, n),
            "qzeros": _rand_int32(gen, k // GROUP_SIZE, n // 8),
            "g_idx": g_idx,
        }
    return ckpt


def _awq_checkpoint(seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    ckpt = {}
    for name, (k, n) in (
        ("gate_proj", (HIDDEN, INTER)),
        ("up_proj", (HIDDEN, INTER)),
        ("down_proj", (INTER, HIDDEN)),
    ):
        ckpt[name] = {
            "qweight": _rand_int32(gen, k, n // 8),
            "scales": _rand_scales(gen, k // GROUP_SIZE, n),
            "qzeros": _rand_int32(gen, k // GROUP_SIZE, n // 8),
        }
    return ckpt


class _MLP(torch.nn.Module):
    def __init__(self, quant_config):
        super().__init__()
        from sglang.srt.layers.linear import (
            MergedColumnParallelLinear,
            RowParallelLinear,
        )

        self.gate_up_proj = MergedColumnParallelLinear(
            HIDDEN,
            [INTER, INTER],
            bias=False,
            params_dtype=torch.float16,
            quant_config=quant_config,
            prefix="model.layers.0.mlp.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            INTER,
            HIDDEN,
            bias=False,
            params_dtype=torch.float16,
            quant_config=quant_config,
            prefix="model.layers.0.mlp.down_proj",
        )

    def forward(self, x):
        gate, up = self.gate_up_proj(x)[0].chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)[0]


def _load(model, ckpt):
    params = dict(model.named_parameters())
    for name, tensors in ckpt.items():
        for suffix, tensor in tensors.items():
            if name == "down_proj":
                param = params[f"down_proj.{suffix}"]
                param.weight_loader(param, tensor)
            else:
                param = params[f"gate_up_proj.{suffix}"]
                shard_id = 0 if name == "gate_proj" else 1
                param.weight_loader(param, tensor, shard_id)


def _refit(model, ckpt):
    # the component the scheduler drives for each runner in a session
    updater = WeightUpdater(
        tp_rank=0,
        device="cuda",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=None,
        recapture_cuda_graph=None,
        get_model_runner=None,
    )
    updater.begin_weight_update()
    _load(model, ckpt)
    updater.end_weight_update(run_post_load=False)


class TestMarlinInverse(CustomTestCase):
    """The inverse transforms undo the repack kernels bit for bit."""

    def test_round_trips(self):
        from sglang.kernels.ops.quantization.awq_marlin_repack import (
            awq_marlin_repack,
        )
        from sglang.kernels.ops.quantization.gptq_marlin_repack import (
            gptq_marlin_repack,
        )

        gen = torch.Generator(device=DEVICE).manual_seed(0)
        empty = torch.empty(0, dtype=torch.int32, device=DEVICE)
        for bits in (4, 8):
            pack = 32 // bits
            for k, n, g in ((896, 1152, 128), (1024, 512, -1), (2048, 256, 64)):
                groups = 1 if g == -1 else k // g
                with self.subTest(bits=bits, k=k, n=n, group_size=g):
                    qw = _rand_int32(gen, k // pack, n)
                    qm = gptq_marlin_repack(qw, empty, k, n, bits)
                    self.assertTrue(
                        torch.equal(marlin_to_gptq_weights(qm, empty, k, n, bits), qw)
                    )
                    if g != -1:
                        g_idx = (
                            torch.randperm(k, device=DEVICE, generator=gen).int() // g
                        )
                        _, perm = marlin_sort_g_idx(g_idx)
                        qm = gptq_marlin_repack(qw, perm, k, n, bits)
                        self.assertTrue(
                            torch.equal(
                                marlin_to_gptq_weights(qm, perm, k, n, bits), qw
                            )
                        )

                    aw = _rand_int32(gen, k, n // pack)
                    am = awq_marlin_repack(aw, size_k=k, size_n=n, num_bits=bits)
                    self.assertTrue(
                        torch.equal(marlin_to_awq_weights(am, k, n, bits), aw)
                    )

                    az = _rand_int32(gen, groups, n // pack)
                    zm = awq_to_marlin_zero_points(az, groups, n, bits)
                    self.assertTrue(
                        torch.equal(marlin_to_awq_zero_points(zm, groups, n, bits), az)
                    )

                    s = _rand_scales(gen, groups, n)
                    sm = marlin_permute_scales(s, k, n, g)
                    self.assertTrue(
                        torch.equal(marlin_unpermute_scales(sm, k, n, g), s)
                    )


class TestMarlinDenseReload(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29634)
        torch.set_default_device("cuda")

    @classmethod
    def tearDownClass(cls):
        torch.set_default_device("cpu")

    @contextmanager
    def _runtime(self):
        with (
            get_context().override_server_args(model_path="dummy"),
            get_parallel().override(tp_size=1, tp_rank=0),
        ):
            yield

    def _fresh_model(self, quant_config, ckpt):
        model = _MLP(quant_config).cuda()
        _load(model, ckpt)
        DefaultModelLoader.postprocess_weights(model, DEVICE)
        return model

    def _assert_same_weights(self, model, expected):
        expected_params = dict(expected.named_parameters())
        for name, param in model.named_parameters():
            self.assertTrue(
                torch.equal(param, expected_params[name]),
                f"{name} differs from a fresh load",
            )

    def _check_refit(self, quant_config, ckpt_a, ckpt_b):
        fresh_a = self._fresh_model(quant_config, ckpt_a)
        fresh_b = self._fresh_model(quant_config, ckpt_b)
        model = self._fresh_model(quant_config, ckpt_a)
        ptrs = {name: p.data_ptr() for name, p in model.named_parameters()}

        _refit(model, ckpt_b)
        self._assert_same_weights(model, fresh_b)
        _refit(model, ckpt_a)
        self._assert_same_weights(model, fresh_a)
        # captured CUDA graphs keep reading the original buffers
        self.assertEqual(
            {name: p.data_ptr() for name, p in model.named_parameters()}, ptrs
        )

        torch.manual_seed(0)
        x = torch.randn(M, HIDDEN, dtype=torch.float16) / 10
        out_a = fresh_a(x).float()
        out_b = fresh_b(x).float()
        assert_output_close(self, model(x), out_a, cos_threshold=0.9999)
        cos_ab = torch.nn.functional.cosine_similarity(
            out_a.flatten(), out_b.flatten(), dim=0
        ).item()
        self.assertLess(cos_ab, 0.9, "the two checkpoints must give different outputs")

    def test_gptq_refit_reproduces_a_fresh_load(self):
        for desc_act in (False, True):
            with self.subTest(desc_act=desc_act), self._runtime():
                config = GPTQMarlinConfig.from_config(
                    {
                        "bits": 4,
                        "group_size": GROUP_SIZE,
                        "desc_act": desc_act,
                        "sym": True,
                        "lm_head": False,
                    }
                )
                self._check_refit(
                    config,
                    _gptq_checkpoint(1, desc_act),
                    _gptq_checkpoint(2, desc_act),
                )

    def test_awq_refit_reproduces_a_fresh_load(self):
        with self._runtime():
            config = AWQMarlinConfig.from_config(
                {
                    "bits": 4,
                    "group_size": GROUP_SIZE,
                    "zero_point": True,
                    "lm_head": False,
                }
            )
            self._check_refit(config, _awq_checkpoint(1), _awq_checkpoint(2))


if __name__ == "__main__":
    unittest.main()
