#!/usr/bin/env python3

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.kernels as K
from sglang.kernels import KernelBackend, PlatformInfo
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4EmbeddingMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GROUP_SIZE = 16

# Written out independently of the implementation: the E2M1 code points in
# magnitude order, so index == the 3-bit magnitude code.
_REFERENCE_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def reference_dequant(
    packed: torch.Tensor, block_scale: torch.Tensor, global_scale: float
) -> torch.Tensor:
    """Comparison oracle. Kept as a plain per-element loop on purpose: a
    vectorized rewrite would mirror the code under test."""
    rows, half = packed.shape
    hidden = half * 2
    out = torch.zeros(rows, hidden, dtype=torch.float32)
    for r in range(rows):
        for c in range(hidden):
            byte = int(packed[r, c // 2])
            code = (byte & 0x0F) if c % 2 == 0 else (byte >> 4)
            magnitude = _REFERENCE_E2M1[code & 0x7]
            value = -magnitude if code & 0x8 else magnitude
            scale = float(block_scale[r, c // GROUP_SIZE]) * global_scale
            out[r, c] = value * scale
    return out


def build_layer(method, vocab_size: int, hidden_size: int) -> torch.nn.Module:
    """Materialize through create_weights, then fill as a checkpoint would."""
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=hidden_size,
        output_partition_sizes=[vocab_size],
        input_size=hidden_size,
        output_size=vocab_size,
        params_dtype=torch.bfloat16,
    )

    generator = torch.Generator().manual_seed(0)
    layer.weight.data.copy_(
        torch.randint(
            0,
            256,
            (vocab_size, hidden_size // 2),
            dtype=torch.uint8,
            generator=generator,
        )
    )
    # Keep the block scales in a range e4m3 represents exactly.
    layer.weight_scale.data.copy_(
        torch.randint(
            1,
            8,
            (vocab_size, hidden_size // GROUP_SIZE),
            dtype=torch.int32,
            generator=generator,
        ).to(torch.float8_e4m3fn)
    )
    layer.weight_scale_2.data.fill_(0.125)
    return layer


class TestNvFp4Embedding(CustomTestCase):
    def setUp(self):
        self.method = ModelOptNvFp4EmbeddingMethod(
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=GROUP_SIZE
            )
        )

    def test_matches_reference_dequant(self):
        vocab_size, hidden_size = 24, 64
        layer = build_layer(self.method, vocab_size, hidden_size)
        self.assertEqual(tuple(layer.weight.shape), (vocab_size, hidden_size // 2))
        self.assertEqual(
            tuple(layer.weight_scale.shape), (vocab_size, hidden_size // GROUP_SIZE)
        )

        ids = torch.tensor([[0, 5, 5], [23, 11, 0]])
        got = self.method.embedding(layer, ids)
        expected = reference_dequant(
            layer.weight[ids.reshape(-1)],
            layer.weight_scale[ids.reshape(-1)].float(),
            float(layer.weight_scale_2),
        )

        self.assertEqual(tuple(got.shape), (2, 3, hidden_size))
        self.assertEqual(got.dtype, torch.bfloat16)
        torch.testing.assert_close(
            got.reshape(-1, hidden_size).float(),
            expected.to(torch.bfloat16).float(),
            rtol=0,
            atol=0,
        )

    def test_hidden_size_must_divide_group_size(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            self.method.create_weights(
                torch.nn.Module(),
                input_size_per_partition=40,
                output_partition_sizes=[8],
                input_size=40,
                output_size=8,
                params_dtype=torch.bfloat16,
            )


def _fake_cuda_tensor(dtype, shape, device="cuda:0", contiguous=True):
    """Just the tensor attributes the fused-kernel dispatch inspects."""
    return SimpleNamespace(
        device=torch.device(device),
        dtype=dtype,
        shape=torch.Size(shape),
        ndim=len(shape),
        is_contiguous=lambda: contiguous,
        numel=lambda: math.prod(shape),
    )


def _fake_cuda_layer(vocab_size=8, hidden_size=32, **overrides):
    attributes = {
        "weight": _fake_cuda_tensor(torch.uint8, (vocab_size, hidden_size // 2)),
        "weight_scale": _fake_cuda_tensor(
            torch.float8_e4m3fn, (vocab_size, hidden_size // GROUP_SIZE)
        ),
        "weight_scale_2": _fake_cuda_tensor(torch.float32, (1,)),
        "use_fused_nvfp4_embedding": True,
    }
    attributes.update(overrides)
    return SimpleNamespace(**attributes)


class TestNvFp4EmbeddingFusedDispatch(CustomTestCase):
    """The fused kernel must only run where it is bit-exact with the fallback."""

    def _supports(
        self,
        layer=None,
        output_dtype=torch.bfloat16,
        capability=(9, 0),
        on_cuda=True,
    ):
        layer = layer or _fake_cuda_layer()
        with (
            mock.patch.object(modelopt_quant, "is_cuda", return_value=on_cuda),
            mock.patch.object(
                modelopt_quant, "get_device_capability", return_value=capability
            ),
        ):
            return modelopt_quant._supports_fused_nvfp4_embedding(
                layer, GROUP_SIZE, output_dtype
            )

    def _can_use(self, layer=None, ids=None):
        layer = layer or _fake_cuda_layer()
        ids = ids or _fake_cuda_tensor(torch.int64, (4,))
        return modelopt_quant._can_use_fused_nvfp4_embedding(layer, ids)

    def test_eligible_table_supports_fused_kernel(self):
        self.assertTrue(self._supports())
        self.assertTrue(self._supports(output_dtype=torch.float16))

    def test_unsupported_platforms_fall_back(self):
        cases = {
            "A100 (SM 8.0) lacks Triton E4M3": dict(capability=(8, 0)),
            "unknown capability": dict(capability=(None, None)),
            "ROCm": dict(on_cuda=False),
            "table on CPU": dict(
                layer=_fake_cuda_layer(
                    weight=_fake_cuda_tensor(torch.uint8, (8, 16), device="cpu")
                )
            ),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertFalse(self._supports(**kwargs))

    def test_unsupported_tables_fall_back(self):
        hidden = 32
        cases = {
            "FP8 output": dict(output_dtype=torch.float8_e4m3fn),
            "scales on another GPU": dict(
                layer=_fake_cuda_layer(
                    weight_scale=_fake_cuda_tensor(
                        torch.float8_e4m3fn,
                        (8, hidden // GROUP_SIZE),
                        device="cuda:1",
                    )
                )
            ),
            "non-contiguous scales": dict(
                layer=_fake_cuda_layer(
                    weight_scale=_fake_cuda_tensor(
                        torch.float8_e4m3fn,
                        (8, hidden // GROUP_SIZE),
                        contiguous=False,
                    )
                )
            ),
            "scale rows do not match table rows": dict(
                layer=_fake_cuda_layer(
                    weight_scale=_fake_cuda_tensor(
                        torch.float8_e4m3fn, (7, hidden // GROUP_SIZE)
                    )
                )
            ),
            "per-row global scale": dict(
                layer=_fake_cuda_layer(
                    weight_scale_2=_fake_cuda_tensor(torch.float32, (8,))
                )
            ),
            "unpacked BF16 table": dict(
                layer=_fake_cuda_layer(
                    weight=_fake_cuda_tensor(torch.bfloat16, (8, hidden // 2))
                )
            ),
            "FP32 block scales": dict(
                layer=_fake_cuda_layer(
                    weight_scale=_fake_cuda_tensor(
                        torch.float32, (8, hidden // GROUP_SIZE)
                    )
                )
            ),
            # 40 // 16 == 2 scale columns, so only the divisibility check catches it.
            "hidden size not a multiple of the group": dict(
                layer=_fake_cuda_layer(hidden_size=40)
            ),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertFalse(self._supports(**kwargs))

    def test_per_lookup_checks(self):
        self.assertTrue(self._can_use())
        self.assertTrue(self._can_use(ids=_fake_cuda_tensor(torch.int32, (2, 3))))
        cases = {
            "table not eligible": dict(
                layer=_fake_cuda_layer(use_fused_nvfp4_embedding=False)
            ),
            "float ids": dict(ids=_fake_cuda_tensor(torch.float32, (4,))),
            "ids on CPU": dict(ids=_fake_cuda_tensor(torch.int64, (4,), device="cpu")),
            "ids on another GPU": dict(
                ids=_fake_cuda_tensor(torch.int64, (4,), device="cuda:1")
            ),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertFalse(self._can_use(**kwargs))

    def test_loading_records_eligibility(self):
        method = ModelOptNvFp4EmbeddingMethod(
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=GROUP_SIZE
            )
        )
        layer = build_layer(method, vocab_size=8, hidden_size=32)
        self.assertFalse(layer.use_fused_nvfp4_embedding)

        with mock.patch.object(
            modelopt_quant, "_supports_fused_nvfp4_embedding", return_value=True
        ) as supports:
            method.process_weights_after_loading(layer)
        supports.assert_called_once_with(layer, GROUP_SIZE, torch.bfloat16)
        self.assertTrue(layer.use_fused_nvfp4_embedding)

    def test_dispatch_matches_registered_kernel_capabilities(self):
        """The registry entry and the srt dispatch each encode the SM floor."""
        spec = K.registry.get_backend(
            "embeddings.nvfp4_embedding", KernelBackend.TRITON
        )
        for capability in [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 1)]:
            with self.subTest(capability=capability):
                platform = PlatformInfo(
                    device_type="cuda",
                    cuda_arch_major=capability[0],
                    cuda_arch_minor=capability[1],
                )
                self.assertEqual(
                    self._supports(capability=capability),
                    K.capabilities_satisfied(spec.capabilities, platform),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
