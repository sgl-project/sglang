"""Fused MXFP4 MLA q absorb + RoPE + KV-cache write against the split path.

On gfx950 graph decode, ``forward_absorb_rocm_core`` replaces the MXFP4 branch
of ``rocm_absorb_q_bmm`` followed by ``_fused_rope_cat_and_cache`` with one
``_fused_fp4_bmm_rope_cat_and_cache`` launch. Both paths must hand attention
the same q and leave the same KV-cache bytes, at the out_cache_loc slots only.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K2.5 MLA geometry at TP8 (64 heads / 8).
HEADS = 8
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
NUM_SLOTS = 1024
KIMI_K25_ROPE = dict(
    max_position=262144,
    base=50000,
    rope_scaling={
        "type": "yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 1.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
    is_neox_style=False,
)


class _FakeKVPool:
    def __init__(self, kv_buffer: torch.Tensor):
        self.kv_buffer = kv_buffer
        self.start_layer = 0
        self.layer_transfer_counter = None
        self.layer_router = PlainPoolRouter(self, "full")

    def get_key_buffer(self, _layer_id: int) -> torch.Tensor:
        return self.kv_buffer


def _is_gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"


@unittest.skipUnless(_is_gfx950(), "MXFP4 absorbed BMM fusion is gfx950-only")
class TestFusedFp4AbsorbRopeCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.layers.quantization.quark.utils import quark_post_load_weights
        from sglang.srt.layers.rotary_embedding import get_rope_wrapper
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            forward_mla_rocm,
        )

        cls.fp8_dtype = fp8_dtype
        cls.mla = forward_mla_rocm
        torch.manual_seed(0)
        with torch.device("cuda"):
            cls.rotary_emb = get_rope_wrapper(
                QK_ROPE_HEAD_DIM,
                rotary_dim=QK_ROPE_HEAD_DIM,
                device="cuda",
                **KIMI_K25_ROPE,
            )
            # The weight loader's kv_b_proj -> (w_kc, w_scale_k) path,
            # as for a BF16 kv_b_proj on gfx950 Quark checkpoints.
            kv_b = torch.randn(
                HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
                KV_LORA_RANK,
                dtype=torch.bfloat16,
            ) * (KV_LORA_RANK**-0.5)
            w_kc, cls.w_scale_k, _, _ = quark_post_load_weights(
                SimpleNamespace(
                    qk_nope_head_dim=QK_NOPE_HEAD_DIM, v_head_dim=V_HEAD_DIM
                ),
                kv_b,
                "mxfp4",
            )
        cls.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)

    def _attn(self, kv_cache_dtype: str):
        k_scale = (
            torch.tensor(0.5, dtype=torch.float32, device="cuda")
            if kv_cache_dtype == "fp8_e4m3"
            else None
        )
        return SimpleNamespace(
            w_kc=self.w_kc,
            w_scale_k=self.w_scale_k,
            kv_cache_dtype=kv_cache_dtype,
            current_attention_backend="aiter",
            rotary_emb=self.rotary_emb,
            attn_mqa=SimpleNamespace(layer_id=0, k_scale=k_scale),
        )

    def _inputs(self, batch: int):
        g = torch.Generator(device="cuda").manual_seed(1000 + batch)
        # Same strided views prepare hands core: q split, latent split.
        q = torch.randn(
            batch,
            HEADS,
            QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=g,
        )
        latent = torch.randn(
            batch,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=g,
        )
        q_nope, q_pe = q.split([QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)
        k_nope = latent[..., :KV_LORA_RANK].unsqueeze(1)
        k_pe = latent[..., KV_LORA_RANK:].unsqueeze(1)
        positions = torch.randint(
            0,
            KIMI_K25_ROPE["max_position"],
            (batch,),
            dtype=torch.int64,
            device="cuda",
            generator=g,
        )
        out_cache_loc = torch.randperm(NUM_SLOTS, device="cuda", generator=g)[:batch]
        return q_nope, q_pe, k_nope, k_pe, positions, out_cache_loc

    def _kv_buffer(self, kv_cache_dtype: str):
        # Random bytes, so a write to a wrong slot or a missed write both show.
        dtype = self.fp8_dtype if kv_cache_dtype == "fp8_e4m3" else torch.bfloat16
        width = KV_LORA_RANK + QK_ROPE_HEAD_DIM
        raw = torch.randint(
            0, 64, (NUM_SLOTS, 1, width * dtype.itemsize), dtype=torch.uint8
        )
        return raw.cuda().view(dtype)

    def _run(self, fn, kv_buffer):
        backend = SimpleNamespace(token_to_kv_pool=_FakeKVPool(kv_buffer))
        with forward_context(ForwardContext(attn_backend=backend)):
            return fn()

    def test_matches_split_absorb_then_rope_cache(self):
        for kv_cache_dtype in ("fp8_e4m3", "auto"):
            for batch in (1, 7, 64, 200):
                with self.subTest(kv_cache_dtype=kv_cache_dtype, batch=batch):
                    self._check(kv_cache_dtype, batch)

    def _check(self, kv_cache_dtype: str, batch: int):
        attn = self._attn(kv_cache_dtype)
        q_nope, q_pe, k_nope, k_pe, positions, loc = self._inputs(batch)
        split_kv = self._kv_buffer(kv_cache_dtype)
        fused_kv = split_kv.clone()

        rope_cache_args = dict(
            attn=attn,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            positions=positions,
            out_cache_loc=loc,
        )

        def split():
            q_nope_out = self.mla.rocm_absorb_q_bmm(
                attn=attn, q_nope=q_nope, is_capture_mode=True
            ).transpose(0, 1)
            return self.mla._fused_rope_cat_and_cache(
                q_nope_out=q_nope_out, **rope_cache_args
            )[0]

        def fused():
            return self.mla._fused_fp4_bmm_rope_cat_and_cache(
                q_nope=q_nope, **rope_cache_args
            )[0]

        q_split = self._run(split, split_kv)
        q_fused = self._run(fused, fused_kv)
        torch.cuda.synchronize()

        self.assertEqual(q_fused.dtype, q_split.dtype)
        self.assertEqual(q_fused.shape, (batch, HEADS, KV_LORA_RANK + QK_ROPE_HEAD_DIM))
        self.assertTrue(
            torch.equal(fused_kv.view(torch.uint8), split_kv.view(torch.uint8)),
            "KV-cache bytes differ",
        )
        if kv_cache_dtype != "fp8_e4m3":
            self.assertTrue(torch.equal(q_fused, q_split), "q differs")
            return
        # The rope half shares no arithmetic with the BMM, so it matches bit for bit.
        self.assertTrue(
            torch.equal(
                q_fused[..., KV_LORA_RANK:].view(torch.uint8),
                q_split[..., KV_LORA_RANK:].view(torch.uint8),
            ),
            "q rope half differs",
        )
        # Split rounds the BMM to BF16 before the FP8 cast, fused casts FP32 to FP8;
        # the double rounding moves about 1% of values by one FP8 step.
        torch.testing.assert_close(
            q_fused[..., :KV_LORA_RANK].float(),
            q_split[..., :KV_LORA_RANK].float(),
            rtol=0.125,
            atol=2**-9,
        )


if __name__ == "__main__":
    unittest.main()
