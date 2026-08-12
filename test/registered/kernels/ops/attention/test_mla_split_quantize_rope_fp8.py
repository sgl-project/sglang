import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large"
)
register_cuda_ci(
    est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200"
)


DEVICE = "cuda"
DTYPE = torch.bfloat16
FP8_DTYPE = torch.float8_e4m3fn

NUM_HEADS = 64
KV_LORA_RANK = 512
ROPE_DIM = 64
Q_PROJ_NOPE_DIM = 192


def _fp8_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _make_cos_sin_cache(max_position: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, ROPE_DIM, 2, device=DEVICE, dtype=torch.float32)
            / ROPE_DIM
        )
    )
    positions = torch.arange(max_position, device=DEVICE, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _make_production_strided_inputs(num_tokens: int):
    generator = torch.Generator(device=DEVICE).manual_seed(20260812 + num_tokens)

    # The absorbed BMM produces contiguous [H, T, 512], which is transposed
    # without a copy before q quantization. Keep its exact production strides.
    q_nope_storage = torch.randn(
        (NUM_HEADS, num_tokens, KV_LORA_RANK),
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )
    q_nope = q_nope_storage.transpose(0, 1)

    # q_rope is a suffix view of the BF16 q projection. GLM uses a 192-wide
    # pre-RoPE projection component, so its head stride is 192 + 64.
    q_proj = torch.randn(
        (num_tokens, NUM_HEADS, Q_PROJ_NOPE_DIM + ROPE_DIM),
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )
    q_rope = q_proj[
        ..., Q_PROJ_NOPE_DIM : Q_PROJ_NOPE_DIM + ROPE_DIM
    ]

    # k_nope and k_rope are sibling views of the latent-cache projection and
    # are squeezed to two dimensions before entering the TRTLLM MLA backend.
    k_storage = torch.randn(
        (num_tokens, KV_LORA_RANK + ROPE_DIM),
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )
    k_nope = k_storage[:, :KV_LORA_RANK]
    k_rope = k_storage[
        :, KV_LORA_RANK : KV_LORA_RANK + ROPE_DIM
    ]

    assert q_nope.stride() == (
        KV_LORA_RANK,
        num_tokens * KV_LORA_RANK,
        1,
    )
    assert q_rope.stride(1) != ROPE_DIM
    assert k_nope.stride(0) != KV_LORA_RANK
    return q_nope, q_rope, k_nope, k_rope


class TestMlaSplitQuantizeRopeFp8(CustomTestCase):
    def assert_fp8_bytes_equal(
        self, actual: torch.Tensor, expected: torch.Tensor, component: str
    ) -> None:
        self.assertEqual(actual.dtype, FP8_DTYPE)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            torch.equal(_fp8_bytes(actual), _fp8_bytes(expected)),
            f"{component} is not byte-identical",
        )

    def test_q_nope_only_cast_honors_input_and_output_strides(self):
        from sglang.kernels.ops.kvcache.cache_ops import cast_q_nope_to_fp8

        for num_tokens in (1, 257):
            with self.subTest(num_tokens=num_tokens):
                q_nope, _, _, _ = _make_production_strided_inputs(num_tokens)

                # Pad the output head/feature storage. Bytes outside the
                # 512-wide q-nope prefix must not be touched; the reduced
                # FlashInfer launch fills the tail.
                output_storage = torch.empty(
                    (num_tokens, NUM_HEADS + 3, KV_LORA_RANK + ROPE_DIM + 16),
                    device=DEVICE,
                    dtype=FP8_DTYPE,
                )
                output_storage.view(torch.uint8).fill_(0xA5)
                expected_storage = output_storage.view(torch.uint8).clone()
                q_fp8 = output_storage[
                    :, :NUM_HEADS, : KV_LORA_RANK + ROPE_DIM
                ]

                expected_nope = q_nope.to(FP8_DTYPE)
                expected_storage[
                    :, :NUM_HEADS, :KV_LORA_RANK
                ].copy_(_fp8_bytes(expected_nope))

                cast_q_nope_to_fp8(q_fp8, q_nope)
                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(output_storage.view(torch.uint8), expected_storage),
                    "cast must write exactly the active q-nope prefix",
                )

    def test_split_helper_is_byte_identical_to_flashinfer_helper(self):
        from sglang.kernels.ops.attention.utils import (
            mla_quantize_and_rope_for_fp8,
            mla_split_quantize_and_rope_for_fp8,
        )

        cos_sin_cache = _make_cos_sin_cache(max_position=1024)
        for num_tokens in (1, 37):
            q_nope, q_rope, k_nope, k_rope = _make_production_strided_inputs(
                num_tokens
            )
            pos_ids = (
                torch.arange(num_tokens, device=DEVICE, dtype=torch.int64) * 17 + 3
            ) % cos_sin_cache.shape[0]

            for is_neox in (False, True):
                with self.subTest(num_tokens=num_tokens, is_neox=is_neox):
                    reference = mla_quantize_and_rope_for_fp8(
                        q_nope,
                        q_rope,
                        k_nope,
                        k_rope,
                        pos_ids,
                        cos_sin_cache,
                        is_neox,
                        KV_LORA_RANK,
                        ROPE_DIM,
                    )
                    split = mla_split_quantize_and_rope_for_fp8(
                        q_nope,
                        q_rope,
                        k_nope,
                        k_rope,
                        pos_ids,
                        cos_sin_cache,
                        is_neox,
                        KV_LORA_RANK,
                        ROPE_DIM,
                    )
                    torch.cuda.synchronize()

                    for component, actual, expected in zip(
                        ("q", "k_nope", "k_rope"), split, reference
                    ):
                        self.assert_fp8_bytes_equal(actual, expected, component)

                    self.assertTrue(
                        torch.equal(
                            _fp8_bytes(split[0][..., :KV_LORA_RANK]),
                            _fp8_bytes(q_nope.to(FP8_DTYPE)),
                        ),
                        "split q-nope prefix must use the exact BF16-to-FP8 cast",
                    )


if __name__ == "__main__":
    unittest.main()
