"""Loads the DeepSeek V4.1 reference implementation for numerical comparison.

Point DSV41_REF_INFERENCE_DIR at the reference `inference/` directory (model.py,
kernel.py, engram.py). Tests skip when it is unset.
"""

import importlib
import os
import sys
import unittest

import torch

REF_DIR = os.getenv("DSV41_REF_INFERENCE_DIR")

requires_ref = unittest.skipUnless(
    REF_DIR, "set DSV41_REF_INFERENCE_DIR to the reference inference directory"
)


def load_ref():
    if REF_DIR not in sys.path:
        sys.path.insert(0, REF_DIR)
    model = importlib.import_module("model")
    kernel = importlib.import_module("kernel")
    engram = importlib.import_module("engram")
    return model, kernel, engram


def small_args(model, **overrides):
    """A small ModelArgs with the released model's scale-independent settings."""
    fields = dict(
        max_batch_size=2,
        max_seq_len=256,
        vocab_size=512,
        dim=256,
        moe_inter_dim=128,
        n_layers=3,
        n_mtp_layers=0,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=128,
        head_dim=128,
        rope_head_dim=32,
        norm_eps=1e-20,
        o_groups=2,
        o_lora_rank=64,
        window_size=16,
        compress_ratios=(0, 2, 1),
        kv_source_layers=(1, 2),
        index_source_layers=(1, 2),
        compress_rope_theta=160000.0,
        original_seq_len=128,
        rope_theta=10000.0,
        rope_factor=16,
        index_n_heads=4,
        index_head_dim=64,
        index_topk=8,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )
    fields.update(overrides)
    return model.ModelArgs(**fields)


def randomize_(module: torch.nn.Module, seed: int = 0) -> None:
    """Fill every parameter with random data of its own dtype (in place)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for name, param in module.named_parameters():
        shape = param.shape
        if param.dtype == torch.float8_e4m3fn:
            value = torch.rand(shape, generator=gen, device="cpu") * 2 - 1
            value = value.to(torch.float8_e4m3fn)
        elif param.dtype == torch.float8_e8m0fnu:
            value = torch.exp2(
                torch.randint(-3, 3, shape, generator=gen, device="cpu").float()
            )
            value = value.to(torch.float8_e8m0fnu)
        elif param.dtype == torch.float4_e2m1fn_x2:
            value = torch.randint(
                0, 256, shape, generator=gen, dtype=torch.uint8, device="cpu"
            )
            value = value.view(torch.float4_e2m1fn_x2)
        elif name.endswith(("norm.weight", "q_weight", "k_weight")):
            value = 1 + 0.1 * torch.randn(shape, generator=gen, device="cpu")
        elif name.endswith(("hc_attn_fn", "hc_ffn_fn")):
            # Mixing logits are a dot product over hc * dim inputs; keep them O(1).
            value = torch.randn(shape, generator=gen, device="cpu") / shape[-1] ** 0.5
        elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
            value = 1 + 0.1 * torch.randn(shape, generator=gen, device="cpu")
        elif name.endswith(
            ("hc_attn_base", "hc_ffn_base", "bias", "bias_vl", "attn_sink")
        ):
            value = torch.randn(shape, generator=gen, device="cpu")
        else:
            value = 0.1 * torch.randn(shape, generator=gen, device="cpu")
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))


class RefTestCase(unittest.TestCase):
    """bf16 default dtype and cuda default device, as the reference generate.py sets."""

    @classmethod
    def setUpClass(cls):
        cls._prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")
        cls.model, cls.kernel, cls.engram = load_ref()

    def setUp(self):
        torch.manual_seed(int(os.getenv("DSV41_TEST_SEED", "0")))

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._prev_dtype)
        torch.set_default_device("cpu")


def assert_equal(actual: torch.Tensor, expected: torch.Tensor, msg: str = "") -> None:
    if actual.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
    if not torch.equal(actual, expected):
        diff = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"{msg} mismatch: {(diff > 0).sum().item()} elements, max_abs={diff.amax().item():.3e}"
        )


def report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().amax().clamp_min(1e-12)
    rel_l2 = (diff.norm() / expected.float().norm().clamp_min(1e-12)).item()
    print(
        f"{name}: max_abs={diff.amax().item():.3e} max_rel={(diff.amax() / scale).item():.3e} rel_l2={rel_l2:.3e}"
    )
