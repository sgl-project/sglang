"""CPU regressions for checkpoint/runtime KV scales across reload boundaries."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.model_loader.loader import (
    DefaultModelLoader,
    postprocess_weight,
    restore_weight,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils.weight_checker import (
    _build_check_entries,
    _check_tensors,
    _is_skip_weight_check,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MODULE = "sglang.srt.layers.quantization.kv_cache"
CPU = torch.device("cpu")
NAMES = ("k_scale", "v_scale")


def _bytes(tensor):
    return tensor.detach().reshape(-1).view(torch.uint8)


class TestKVCacheScaleReload(unittest.TestCase):
    def _make(self, raw=(-1.0, -1.0), method=None):
        layer = torch.nn.Module()
        layer.quant_method = method or BaseKVCacheMethod(None)
        with torch.device("cpu"):
            layer.quant_method.create_weights(layer)
        layer._test_identity = {}
        for name, value in zip(NAMES, raw):
            param = getattr(layer, name)
            param.weight_loader = default_weight_loader
            param.loader_metadata = object()
            param.weight_loader(param, torch.tensor(value, device=CPU))
            layer._test_identity[name] = (
                param,
                param.data_ptr(),
                param.weight_loader,
                param.loader_metadata,
            )
        return layer

    def _assert_scales(self, layer, expected):
        self.assertEqual(set(dict(layer.named_parameters())), set(NAMES))
        self.assertEqual(dict(layer.named_buffers()), {})
        for name, value in zip(NAMES, expected):
            param = getattr(layer, name)
            identity = layer._test_identity[name]
            self.assertIs(param, identity[0])
            self.assertEqual(param.data_ptr(), identity[1])
            self.assertIs(param.weight_loader, identity[2])
            self.assertIs(param.loader_metadata, identity[3])
            self.assertEqual(param.dtype, torch.float32)
            self.assertEqual(param.device, CPU)
            self.assertFalse(param.requires_grad)
            # k/v scales stay exempt: a checkpoint without them is legitimate, and
            # the synthesized default must not be reported as corruption.
            self.assertTrue(_is_skip_weight_check(name, param, None))
            self.assertEqual(getattr(layer, name + "_float"), value)
            self.assertTrue(
                torch.equal(_bytes(param), _bytes(torch.tensor(value, device=CPU)))
            )

    def _write(self, layer, **values):
        for name, value in values.items():
            param = getattr(layer, name)
            param.weight_loader(param, torch.tensor(value, device=CPU))

    def _process(self, layer, expected):
        for _ in range(3):  # Repeated end/postprocess, not only paired reloads.
            layer.quant_method.process_weights_after_loading(layer)
            self._assert_scales(layer, expected)

    def test_empty_reload_preserves_sentinel_single_and_dual_scales(self):
        for fnuz in (False, True):
            for raw in ((-1.0, -1.0), (0.0, -2.0), (0.25, -1.0), (0.25, 0.75)):
                with (
                    self.subTest(fnuz=fnuz, raw=raw),
                    patch(f"{MODULE}.is_fp8_fnuz", return_value=fnuz),
                ):
                    layer = self._make(raw)
                    method = layer.quant_method
                    method.restore_weights_before_loading(layer)  # Initial no-op.
                    factor = 2 if fnuz else 1
                    expected = (
                        (1.0, 1.0)
                        if raw[0] <= 0
                        else (raw[0] * factor, max(raw) * factor)
                    )
                    self._process(layer, expected)
                    snapshot = {
                        n: _bytes(p).clone() for n, p in layer.named_parameters()
                    }
                    for _ in range(5):
                        method.restore_weights_before_loading(layer)
                        self._assert_scales(layer, raw)
                        method.restore_weights_before_loading(layer)
                        self._assert_scales(layer, raw)
                        self._process(layer, expected)
                        for name, param in layer.named_parameters():
                            self.assertTrue(torch.equal(_bytes(param), snapshot[name]))

    def test_partial_full_and_same_runtime_value_reload_matches_fresh_load(self):
        for fnuz in (False, True):
            with (
                self.subTest(fnuz=fnuz),
                patch(f"{MODULE}.is_fp8_fnuz", return_value=fnuz),
            ):
                layer = self._make()
                self._process(layer, (1.0, 1.0))
                raw = dict(zip(NAMES, (-1.0, -1.0)))
                # Missing -> single -> dual -> partial -> full -> missing again.
                for incoming in (
                    {"k_scale": 1.0},  # Same bytes as the old default, still new.
                    {"k_scale": 2.0},  # Same as the old FNUZ runtime, still new.
                    {"v_scale": 0.5},
                    {"k_scale": 0.125},
                    {"v_scale": 1.0},
                    {"k_scale": 0.75, "v_scale": 0.25},
                    {"k_scale": -1.0, "v_scale": -1.0},
                ):
                    restore_weight(layer, CPU)  # Real model-wide begin hook.
                    self._write(layer, **incoming)
                    restore_weight(layer, CPU)  # Must not discard pending writes.
                    raw.update(incoming)
                    fresh = self._make(tuple(raw.values()))
                    postprocess_weight(fresh, CPU)
                    factor = 2 if fnuz else 1
                    expected = (
                        (1.0, 1.0)
                        if raw["k_scale"] <= 0
                        else (
                            raw["k_scale"] * factor,
                            (raw["v_scale"] if raw["v_scale"] > 0 else raw["k_scale"])
                            * factor,
                        )
                    )
                    self._assert_scales(fresh, expected)
                    postprocess_weight(layer, CPU)  # Real model-wide end hook.
                    self._process(layer, expected)
                    restore_weight(layer, CPU)
                    self._assert_scales(layer, tuple(raw.values()))
                    self._process(layer, expected)

    def test_changed_scales_without_restore_are_not_silently_ignored(self):
        # Unchanged-byte writes require restore; observably changed scales must
        # not be lost to an unconditional `if processed: return` either.
        for fnuz in (False, True):
            with (
                self.subTest(fnuz=fnuz),
                patch(f"{MODULE}.is_fp8_fnuz", return_value=fnuz),
            ):
                layer = self._make((0.25, 0.75))
                factor = 2 if fnuz else 1
                self._process(layer, (0.25 * factor, 0.75 * factor))
                self._write(layer, k_scale=0.125)
                self._process(layer, (0.125 * factor, 0.75 * factor))
                self._write(layer, v_scale=0.375)
                layer.quant_method.restore_weights_before_loading(layer)
                self._assert_scales(layer, (0.125, 0.375))
                self._process(layer, (0.125 * factor, 0.375 * factor))

    def test_deprecated_single_kv_scale_keeps_missing_v_semantics(self):
        with patch(f"{MODULE}.is_fp8_fnuz", return_value=True):
            model = torch.nn.Module()
            model.self_attn = torch.nn.Module()
            model.self_attn.attn = self._make()
            params = dict(model.named_parameters())
            name = maybe_remap_kv_scale_name("self_attn.kv_scale", params)
            self.assertEqual(name, "self_attn.attn.k_scale")
            for value in (0.25, 0.5, 1.0):
                restore_weight(model, CPU)
                param = params[name]
                param.weight_loader(param, torch.tensor([value], device=CPU))
                postprocess_weight(model, CPU)
                self._process(model.self_attn.attn, (value * 2, value * 2))

    def test_shared_method_keeps_per_layer_state(self):
        with patch(f"{MODULE}.is_fp8_fnuz", return_value=True):
            method = BaseKVCacheMethod(None)
            missing = self._make(method=method)
            calibrated = self._make((0.25, 0.75), method=method)
            for _ in range(3):
                self._process(missing, (1.0, 1.0))
                self._process(calibrated, (0.5, 1.5))
                method.restore_weights_before_loading(missing)
                self._assert_scales(missing, (-1.0, -1.0))
                method.restore_weights_before_loading(calibrated)
                self._assert_scales(calibrated, (0.25, 0.75))

    def test_disk_loader_restores_before_consuming_weights(self):
        class Model(torch.nn.Module):
            def load_weights(model, weights):
                # This is called even for an empty iterable, after restore.
                self.assertEqual(
                    (model.attn.k_scale.item(), model.attn.v_scale.item()),
                    model.expected_raw,
                )
                for name, value in weights:
                    self._write(model.attn, **{name: value})

        for fnuz in (False, True):
            with (
                self.subTest(fnuz=fnuz),
                patch(f"{MODULE}.is_fp8_fnuz", return_value=fnuz),
                patch(
                    "sglang.srt.model_loader.loader.is_cuda_alike", return_value=False
                ),
            ):
                model = Model()
                model.attn = self._make()
                model.expected_raw = (-1.0, -1.0)
                # Exercise the real initial load too (restore must be a no-op).
                raw = dict(zip(NAMES, model.expected_raw))
                for incoming in (
                    {},
                    {"k_scale": 1.0},
                    {},
                    {"k_scale": 2.0},
                    {"v_scale": 0.25},
                ):
                    model.expected_raw = tuple(raw.values())
                    DefaultModelLoader.load_weights_and_postprocess(
                        model, iter(incoming.items()), CPU
                    )
                    raw.update(incoming)
                    fresh = self._make(tuple(raw.values()))
                    postprocess_weight(fresh, CPU)
                    self._process(
                        model.attn, (fresh.k_scale_float, fresh.v_scale_float)
                    )

    def test_strict_checker_detects_scale_mutation(self):
        def cpu_transport(tensor, *args, **kwargs):
            # Only replace transport; use the real byte comparator and checker.
            self.assertEqual(tensor.device, CPU)
            return tensor

        with (
            patch(f"{MODULE}.is_fp8_fnuz", return_value=True),
            patch.object(torch.Tensor, "cuda", cpu_transport),
        ):
            layer = self._make()
            self._process(layer, (1.0, 1.0))
            snapshot = {n: p.detach().clone() for n, p in layer.named_parameters()}
            for name in NAMES:
                self._write(layer, **{name: 2.0})
                with self.assertRaisesRegex(Exception, name):
                    _check_tensors(
                        _build_check_entries(snapshot, set()),
                        _build_check_entries(dict(layer.named_parameters()), set()),
                        allow_quant_error=False,
                    )
                self._write(layer, **{name: 1.0})

    def test_all_43_layers_empty_reload_is_byte_exact(self):
        with patch(f"{MODULE}.is_fp8_fnuz", return_value=True):
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleList([self._make() for _ in range(43)])
            postprocess_weight(model, CPU)
            snapshot = {n: _bytes(p).clone() for n, p in model.named_parameters()}
            self.assertEqual(len(snapshot), 86)
            for _ in range(4):
                restore_weight(model, CPU)
                postprocess_weight(model, CPU)
                for name, param in model.named_parameters():
                    self.assertTrue(torch.equal(_bytes(param), snapshot[name]), name)
                for layer in model.layers:
                    self._assert_scales(layer, (1.0, 1.0))

    def test_non_scalar_scales_still_rejected(self):
        layer = self._make()
        layer.k_scale.data = torch.tensor([0.25], device=CPU)
        with self.assertRaisesRegex(ValueError, "per-tensor"):
            layer.quant_method.process_weights_after_loading(layer)

    def test_v_only_scale_still_rejected(self):
        layer = self._make((-1.0, 0.25))
        with self.assertRaises(AssertionError):
            layer.quant_method.process_weights_after_loading(layer)


if __name__ == "__main__":
    unittest.main()
