import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


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
        ** (torch.arange(0, ROPE_DIM, 2, device=DEVICE, dtype=torch.float32) / ROPE_DIM)
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
    q_rope = q_proj[..., Q_PROJ_NOPE_DIM : Q_PROJ_NOPE_DIM + ROPE_DIM]

    # k_nope and k_rope are sibling views of the latent-cache projection and
    # are squeezed to two dimensions before entering the TRTLLM MLA backend.
    k_storage = torch.randn(
        (num_tokens, KV_LORA_RANK + ROPE_DIM),
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )
    k_nope = k_storage[:, :KV_LORA_RANK]
    k_rope = k_storage[:, KV_LORA_RANK : KV_LORA_RANK + ROPE_DIM]

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

    def test_qk_nope_cast_honors_input_and_output_strides(self):
        from sglang.kernels.jit.utils import is_arch_support_pdl
        from sglang.kernels.ops.kvcache.cache_ops import cast_qk_nope_to_fp8

        for num_tokens in (1, 257):
            q_nope, _, k_nope, _ = _make_production_strided_inputs(num_tokens)
            pdl_modes = (False, True) if is_arch_support_pdl() else (False,)
            for enable_pdl in pdl_modes:
                with self.subTest(num_tokens=num_tokens, enable_pdl=enable_pdl):
                    # Pad both destinations. Bytes outside the Q prefix and K
                    # row must not be touched by the combined conversion.
                    q_output_storage = torch.empty(
                        (
                            num_tokens,
                            NUM_HEADS + 3,
                            KV_LORA_RANK + ROPE_DIM + 16,
                        ),
                        device=DEVICE,
                        dtype=FP8_DTYPE,
                    )
                    q_output_storage.view(torch.uint8).fill_(0xA5)
                    expected_q_storage = q_output_storage.view(torch.uint8).clone()
                    q_fp8 = q_output_storage[:, :NUM_HEADS, : KV_LORA_RANK + ROPE_DIM]

                    k_output_storage = torch.empty(
                        (num_tokens, KV_LORA_RANK + 16),
                        device=DEVICE,
                        dtype=FP8_DTYPE,
                    )
                    k_output_storage.view(torch.uint8).fill_(0x5A)
                    expected_k_storage = k_output_storage.view(torch.uint8).clone()
                    k_fp8 = k_output_storage[:, :KV_LORA_RANK]

                    expected_q_storage[:, :NUM_HEADS, :KV_LORA_RANK].copy_(
                        _fp8_bytes(q_nope.to(FP8_DTYPE))
                    )
                    expected_k_storage[:, :KV_LORA_RANK].copy_(
                        _fp8_bytes(k_nope.to(FP8_DTYPE))
                    )

                    cast_qk_nope_to_fp8(
                        q_fp8,
                        q_nope,
                        k_fp8,
                        k_nope,
                        enable_pdl=enable_pdl,
                    )
                    torch.cuda.synchronize()

                    self.assertTrue(
                        torch.equal(
                            q_output_storage.view(torch.uint8), expected_q_storage
                        ),
                        "cast must write exactly the active q-nope prefix",
                    )
                    self.assertTrue(
                        torch.equal(
                            k_output_storage.view(torch.uint8), expected_k_storage
                        ),
                        "cast must write exactly the active k-nope row",
                    )

    def test_qk_nope_cast_matches_all_bf16_bit_patterns(self):
        from sglang.kernels.jit.utils import is_arch_support_pdl
        from sglang.kernels.ops.kvcache.cache_ops import cast_qk_nope_to_fp8

        # Two production-shaped tokens contain exactly all 2^16 BF16 bit
        # patterns, including zeros, subnormals, rounding boundaries, infinities,
        # and NaNs. This locks down the byte-level conversion contract.
        bf16_values = (
            torch.arange(1 << 16, device=DEVICE, dtype=torch.int32)
            .to(torch.uint16)
            .view(DTYPE)
        )
        q_nope = bf16_values.view(NUM_HEADS, 2, KV_LORA_RANK).transpose(0, 1)
        k_nope = bf16_values[: 2 * KV_LORA_RANK].view(2, KV_LORA_RANK)
        q_fp8 = torch.empty(
            (2, NUM_HEADS, KV_LORA_RANK + ROPE_DIM),
            device=DEVICE,
            dtype=FP8_DTYPE,
        )
        k_fp8 = torch.empty_like(k_nope, dtype=FP8_DTYPE)

        cast_qk_nope_to_fp8(
            q_fp8,
            q_nope,
            k_fp8,
            k_nope,
            enable_pdl=is_arch_support_pdl(),
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(
                _fp8_bytes(q_fp8[..., :KV_LORA_RANK]),
                _fp8_bytes(q_nope.to(FP8_DTYPE)),
            )
        )
        self.assertTrue(
            torch.equal(_fp8_bytes(k_fp8), _fp8_bytes(k_nope.to(FP8_DTYPE)))
        )

    def test_split_helper_is_byte_identical_to_flashinfer_helper(self):
        from sglang.kernels.ops.attention.utils import (
            mla_quantize_and_rope_for_fp8,
            mla_split_quantize_and_rope_for_fp8,
        )

        cos_sin_cache = _make_cos_sin_cache(max_position=1024)
        for num_tokens in (1, 37):
            q_nope, q_rope, k_nope, k_rope = _make_production_strided_inputs(num_tokens)
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

    def test_split_helper_pdl_chain_to_kv_writer(self):
        from sglang.kernels.jit.utils import is_arch_support_pdl
        from sglang.kernels.ops.attention.utils import (
            mla_split_quantize_and_rope_for_fp8,
        )
        from sglang.kernels.ops.kvcache.set_mla_kv_buffer import set_mla_kv_buffer

        if not is_arch_support_pdl():
            self.skipTest("PDL requires SM90 or newer")

        num_tokens = 768
        q_nope, q_rope, k_nope, k_rope = _make_production_strided_inputs(num_tokens)
        cos_sin_cache = _make_cos_sin_cache(max_position=1024)
        pos_ids = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)

        _, k_nope_fp8, k_rope_fp8 = mla_split_quantize_and_rope_for_fp8(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            pos_ids,
            cos_sin_cache,
            False,
            KV_LORA_RANK,
            ROPE_DIM,
        )
        kv_buffer = torch.empty(
            (num_tokens, KV_LORA_RANK + ROPE_DIM),
            device=DEVICE,
            dtype=FP8_DTYPE,
        )
        loc = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)

        # Launch the real PDL-aware TMA cache writer immediately, with no
        # synchronization between the split helper and this dependent read.
        set_mla_kv_buffer(kv_buffer, loc, k_nope_fp8, k_rope_fp8)
        torch.cuda.synchronize()

        expected = torch.cat((k_nope_fp8, k_rope_fp8), dim=-1)
        self.assert_fp8_bytes_equal(kv_buffer, expected, "packed_kv")

    def test_split_dispatch_policy(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.device_capability = (10, 3)
        backend.kv_lora_rank = KV_LORA_RANK
        backend.qk_rope_head_dim = ROPE_DIM

        def should_use(
            *,
            num_tokens=4096,
            num_heads=NUM_HEADS,
            nope_dim=KV_LORA_RANK,
            rope_dim=ROPE_DIM,
            mode=ForwardMode.EXTEND,
            is_prefill=True,
        ):
            q = torch.empty(
                (num_tokens, num_heads, nope_dim), device="meta", dtype=DTYPE
            )
            q_rope = torch.empty(
                (num_tokens, num_heads, rope_dim), device="meta", dtype=DTYPE
            )
            return backend._should_use_trtllm_split_rope_quantize(
                q, q_rope, mode, is_prefill
            )

        self.assertFalse(should_use(num_tokens=4095))
        self.assertTrue(should_use(num_tokens=4096))
        self.assertTrue(should_use(mode=ForwardMode.MIXED))

        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.SPLIT_PREFILL,
            ForwardMode.DLLM_EXTEND,
        ):
            with self.subTest(mode=mode):
                self.assertFalse(should_use(mode=mode))

        self.assertFalse(should_use(is_prefill=False))
        self.assertFalse(should_use(num_heads=32))
        self.assertFalse(should_use(nope_dim=256))
        self.assertFalse(should_use(rope_dim=32))

        scalar = torch.empty((), device="meta", dtype=DTYPE)
        valid_q = torch.empty(
            (4096, NUM_HEADS, KV_LORA_RANK), device="meta", dtype=DTYPE
        )
        valid_q_rope = torch.empty(
            (4096, NUM_HEADS, ROPE_DIM), device="meta", dtype=DTYPE
        )
        self.assertFalse(
            backend._should_use_trtllm_split_rope_quantize(
                scalar, valid_q_rope, ForwardMode.EXTEND, True
            )
        )
        mismatched_q_rope = torch.empty(
            (4095, NUM_HEADS, ROPE_DIM), device="meta", dtype=DTYPE
        )
        self.assertFalse(
            backend._should_use_trtllm_split_rope_quantize(
                valid_q, mismatched_q_rope, ForwardMode.EXTEND, True
            )
        )

        backend.kv_lora_rank = 256
        self.assertFalse(should_use())
        backend.kv_lora_rank = KV_LORA_RANK
        backend.qk_rope_head_dim = 32
        self.assertFalse(should_use())
        backend.qk_rope_head_dim = ROPE_DIM

        for capability, num_tokens, expected in (
            ((10, 0), 4096, False),
            ((10, 0), 8192, True),
            ((10, 3), 4096, True),
            ((9, 0), 8192, False),
            ((10, 1), 8192, False),
            ((11, 0), 8192, False),
        ):
            with self.subTest(capability=capability, num_tokens=num_tokens):
                backend.device_capability = capability
                self.assertEqual(should_use(num_tokens=num_tokens), expected)


if __name__ == "__main__":
    unittest.main()
