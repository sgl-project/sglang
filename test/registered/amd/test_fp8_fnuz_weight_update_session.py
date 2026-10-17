"""gfx94x block-FP8 weights go through real weight-update sessions intact."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

MODULE = "sglang.srt.layers.quantization.fp8"
BLOCK = 128
SHAPES = {
    "dense": {"weight": (512, 256)},
    "moe": {"w13_weight": (2, 512, 256), "w2_weight": (2, 256, 256)},
}
DEVICE = torch.device("cuda")


def _on_fnuz_gpu() -> bool:
    if not (is_hip() and torch.cuda.is_available()):
        return False
    from sglang.srt.layers.quantization import fp8

    return fp8._is_fp8_fnuz


def _aiter_shuffle():
    try:
        from aiter.ops.shuffle import shuffle_weight
    except ImportError:
        return None
    return shuffle_weight


def _fn_checkpoint(seed):
    """Block-FP8 E4M3FN tensors the way a DeepSeek checkpoint or a trainer sends them:
    each block's largest magnitude lands on 448, above FNUZ's 240."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    checkpoint = {}
    for part, shapes in SHAPES.items():
        checkpoint[part] = {}
        for name, shape in shapes.items():
            w = torch.randn(shape, device=DEVICE, generator=generator)
            blocks = w.unflatten(-2, (-1, BLOCK)).unflatten(-1, (-1, BLOCK))
            scale = blocks.abs().amax((-3, -1), keepdim=True) / 448
            checkpoint[part][name] = (
                (blocks / scale).flatten(-2).flatten(-3, -2).to(torch.float8_e4m3fn)
            )
            checkpoint[part][name + "_scale_inv"] = scale.squeeze(-1).squeeze(-2)
    return checkpoint


def _copy_loader(param, loaded):
    # The linear and FusedMoE loaders end in a plain copy_ into the Parameter.
    param.data.copy_(loaded)


def _model():
    from sglang.srt.layers.moe import MoeRunnerBackend
    from sglang.srt.layers.quantization.fp8 import (
        Fp8Config,
        Fp8LinearMethod,
        Fp8MoEMethod,
    )

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[BLOCK, BLOCK]
    )
    model = nn.Module()
    for part, method_cls in (("dense", Fp8LinearMethod), ("moe", Fp8MoEMethod)):
        layer = nn.Module()
        for name, tensor in _fn_checkpoint(seed=0)[part].items():
            param = nn.Parameter(torch.empty_like(tensor), requires_grad=False)
            param.weight_loader = _copy_loader
            layer.register_parameter(name, param)
        layer.quant_method = method_cls(config)
        if method_cls is Fp8MoEMethod:
            # The runner AITER-on configurations pick; the experts are shuffled only for it.
            layer.quant_method.runner = SimpleNamespace(
                runner_backend=MoeRunnerBackend.AITER
            )
        model.add_module(part, layer)
    return model


def _load(model, checkpoint):
    for part, tensors in checkpoint.items():
        for name, tensor in tensors.items():
            param = getattr(getattr(model, part), name)
            param.weight_loader(param, tensor)


def _bytes(model):
    return {
        name: p.detach().view(torch.uint8).clone()
        for name, p in model.named_parameters()
    }


@unittest.skipUnless(_on_fnuz_gpu(), "requires a gfx94x (E4M3FNUZ) GPU")
class TestFp8FnuzWeightUpdateSession(unittest.TestCase):
    """begin_weight_update / end_weight_update run exactly these two loader calls."""

    def _cases(self):
        yield "no shuffle", False, None
        shuffle = _aiter_shuffle()
        if shuffle is not None:
            yield "AITER shuffle", True, shuffle

    def _patched(self, use_aiter, shuffle):
        patches = [patch(f"{MODULE}._use_aiter", use_aiter)]
        if shuffle is not None:
            patches.append(patch(f"{MODULE}.shuffle_weight", shuffle, create=True))
        return patches

    def _fresh(self, seed):
        from sglang.srt.model_loader.loader import DefaultModelLoader

        model = _model()
        _load(model, _fn_checkpoint(seed))
        DefaultModelLoader.postprocess_weights(model, DEVICE)
        return model

    def test_sessions_reproduce_a_fresh_load(self):
        from sglang.srt.model_loader.loader import DefaultModelLoader

        for label, use_aiter, shuffle in self._cases():
            with self.subTest(label):
                patches = self._patched(use_aiter, shuffle)
                for p in patches:
                    p.start()
                try:
                    model = self._fresh(seed=0)
                    identity = {
                        name: (id(p), p.data_ptr())
                        for name, p in model.named_parameters()
                    }
                    for name, param in model.named_parameters():
                        self.assertIs(param.weight_loader, _copy_loader, name)
                    self.assertEqual(model.moe.w13_weight.dtype, torch.float8_e4m3fnuz)
                    self.assertEqual(
                        bool(getattr(model.moe.w13_weight, "is_shuffled", False)),
                        use_aiter,
                    )

                    # An empty session: nothing written, nothing may change.
                    before = _bytes(model)
                    DefaultModelLoader.restore_weights_before_loading(model, DEVICE)
                    DefaultModelLoader.postprocess_weights(model, DEVICE)
                    for name, value in _bytes(model).items():
                        self.assertTrue(torch.equal(value, before[name]), name)

                    # A session that writes new weights must equal loading them fresh.
                    for seed in (1, 2):
                        DefaultModelLoader.restore_weights_before_loading(model, DEVICE)
                        _load(model, _fn_checkpoint(seed))
                        DefaultModelLoader.postprocess_weights(model, DEVICE)
                        fresh = _bytes(self._fresh(seed))
                        for name, param in model.named_parameters():
                            self.assertFalse(torch.isnan(param.float()).any(), name)
                            self.assertTrue(
                                torch.equal(_bytes(model)[name], fresh[name]), name
                            )
                            # CUDA graphs keep reading the same Parameter and storage.
                            self.assertEqual(
                                (id(param), param.data_ptr()), identity[name], name
                            )
                finally:
                    for p in reversed(patches):
                        p.stop()


if __name__ == "__main__":
    unittest.main()
