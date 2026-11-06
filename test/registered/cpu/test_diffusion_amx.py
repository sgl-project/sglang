import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.amx_attn import (
    AMXATTNImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    _cached_get_attn_backend,
    get_attn_backend,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    _split_unquantized_merged_linear,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu-intel")

_SELECTOR = "sglang.multimodal_gen.runtime.layers.attention.selector"


class TestCpuAttentionBackendSelection(CustomTestCase):
    """The selector asks the platform once per candidate backend. The CPU
    platform used to answer amx_attn to every question, so a layer whose
    supported set lacks amx_attn failed at startup, an explicit torch_sdpa
    still ran amx_attn, and fp32 layers reached a kernel that only dispatches
    bf16/fp16."""

    def setUp(self) -> None:
        _cached_get_attn_backend.cache_clear()

    def _resolve(
        self,
        *,
        supported: set[AttentionBackendEnum],
        dtype: torch.dtype = torch.bfloat16,
        selected: AttentionBackendEnum | None = None,
    ) -> AttentionBackendEnum:
        with (
            patch(f"{_SELECTOR}.get_global_forced_attn_backend", return_value=None),
            patch(f"{_SELECTOR}.get_component_forced_attn_backend", return_value=None),
            patch(
                f"{_SELECTOR}.get_global_server_args",
                return_value=SimpleNamespace(attention_backend=None),
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform", CpuPlatform
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.cpu.cpu_has_amx_support",
                return_value=True,
            ),
        ):
            backend_cls = get_attn_backend(
                128,
                dtype,
                supported_attention_backends=supported,
                selected_attention_backend=selected,
            )
        return backend_cls.get_enum()

    def test_layer_without_amx_attn_falls_back_to_sdpa(self):
        backend = self._resolve(
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA}
        )
        self.assertEqual(backend, AttentionBackendEnum.TORCH_SDPA)

    def test_explicit_sdpa_is_honored(self):
        backend = self._resolve(
            supported={AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.AMX_ATTN},
            selected=AttentionBackendEnum.TORCH_SDPA,
        )
        self.assertEqual(backend, AttentionBackendEnum.TORCH_SDPA)

    def test_fp32_layer_uses_sdpa(self):
        backend = self._resolve(
            supported={AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.AMX_ATTN},
            dtype=torch.float32,
        )
        self.assertEqual(backend, AttentionBackendEnum.TORCH_SDPA)

    def test_bf16_layer_that_allows_amx_attn_gets_it(self):
        backend = self._resolve(
            supported={AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.AMX_ATTN}
        )
        self.assertEqual(backend, AttentionBackendEnum.AMX_ATTN)


class TestAMXAttentionBatch(CustomTestCase):
    def test_every_batch_is_attended(self):
        """forward used to attend only query[0], so a batch of two came back
        as a batch of one."""
        torch.manual_seed(0)
        batch, seqlen, heads, head_size = 2, 64, 4, 64
        query, key, value = (
            torch.randn(batch, seqlen, heads, head_size, dtype=torch.bfloat16)
            for _ in range(3)
        )
        impl = AMXATTNImpl(
            num_heads=heads,
            head_size=head_size,
            causal=False,
            softmax_scale=head_size**-0.5,
        )

        out = impl.forward(query, key, value, attn_metadata=None)

        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        tolerance = precision[torch.bfloat16]
        torch.testing.assert_close(out, expected, atol=tolerance, rtol=tolerance)


class TestQwenImagePackedAddedQKV(CustomTestCase):
    def test_split_reads_packed_weight(self):
        """Qwen-Image's lossless path used to slice to_added_qkv.weight by rows
        and run a plain linear per shard. Once AMX packing has run, the weight
        is VNNI-laid-out under an unchanged [N, K] shape and the bias is fp32,
        so that path failed or returned wrong values."""
        torch.manual_seed(0)
        shard_size, in_features = 128, 256
        weight = torch.randn(3 * shard_size, in_features, dtype=torch.bfloat16) * 0.02
        bias = torch.randn(3 * shard_size, dtype=torch.float32) * 0.02
        x = torch.randn(2, 16, in_features, dtype=torch.bfloat16)
        packed_linear = SimpleNamespace(
            weight=torch.ops.sgl_kernel.convert_weight_packed(weight),
            bias=bias,
            output_partition_sizes=[shard_size] * 3,
            use_intel_amx_backend=True,
        )

        shards = _split_unquantized_merged_linear(packed_linear, x)

        expected = F.linear(x.float(), weight.float(), bias).split(shard_size, dim=-1)
        tolerance = precision[torch.bfloat16]
        for shard, expected_shard in zip(shards, expected):
            torch.testing.assert_close(
                shard.float(), expected_shard, atol=tolerance, rtol=tolerance
            )


if __name__ == "__main__":
    unittest.main()
