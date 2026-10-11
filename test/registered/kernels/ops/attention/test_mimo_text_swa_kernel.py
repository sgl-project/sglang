"""MiMo text windows include the current token, as in the release HF mask."""

import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers.masking_utils import sliding_window_causal_mask_function

from sglang.kernels.ops.attention.flash_attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    resolve_sliding_window_size,
)
from sglang.srt.models import mimo_v2, mimo_v2_nextn
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def config():
    return SimpleNamespace(
        hidden_size=128,
        num_hidden_layers=2,
        hybrid_layer_pattern=[1, 0],
        num_attention_heads=16,
        num_key_value_heads=2,
        swa_num_attention_heads=16,
        swa_num_key_value_heads=2,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        sliding_window_size=128,
        attention_bias=False,
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        intermediate_size=256,
        hidden_act="silu",
        layernorm_epsilon=1e-6,
    )


def make_attention(cfg, layer_id=0, mtp=False):
    """Keep the real decoder/attention constructors; stub unrelated weights/TP."""
    module = mimo_v2_nextn if mtp else mimo_v2
    cls = module.MiMoV2MTPLayer if mtp else module.MiMoV2DecoderLayer
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                mimo_v2,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
            )
        )
        for name in ("QKVParallelLinear", "RowParallelLinear", "get_rope"):
            stack.enter_context(patch.object(mimo_v2, name, return_value=nn.Identity()))
        stack.enter_context(
            patch.object(module, "MiMoV2MLP", return_value=nn.Identity())
        )
        stack.enter_context(
            patch.object(module, "is_dense_ffn_fully_dp", return_value=False)
        )
        stack.enter_context(patch.object(module, "declare_attn"))
        stack.enter_context(patch.object(module, "declare_ffn"))
        stack.enter_context(
            patch.object(
                module, "append_stages", return_value=(nn.Identity(), nn.Identity())
            )
        )
        return cls(cfg, layer_id=layer_id).self_attn


def runner_window(cfg, mtp=False):
    cls = mimo_v2_nextn.MiMoV2MTP if mtp else mimo_v2.MiMoV2ForCausalLM
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = cfg
    return resolve_sliding_window_size(
        model,
        SimpleNamespace(
            is_hybrid_swa=True, sliding_window_size=cfg.sliding_window_size
        ),
    )


def reference(q, k, v, sinks, window=128):
    """Release text semantics: HF causal window, then a virtual sink column."""
    ratio = q.shape[1] // k.shape[1]
    qq = q.double().transpose(0, 1)
    kk = k.double().repeat_interleave(ratio, 1).transpose(0, 1)
    vv = v.double().repeat_interleave(ratio, 1).transpose(0, 1)
    scores = (qq @ kk.transpose(-1, -2)) * q.shape[-1] ** -0.5
    qi = torch.arange(k.shape[0] - q.shape[0], k.shape[0], device=q.device)[:, None]
    ki = torch.arange(k.shape[0], device=q.device)[None, :]
    allowed = ki <= qi
    if window is not None:
        allowed = sliding_window_causal_mask_function(window)(0, 0, qi, ki)
    scores.masked_fill_(~allowed, -math.inf)
    if sinks is not None:
        scores = torch.cat(
            (scores, sinks.double()[:, None, None].expand(-1, q.shape[0], 1)), -1
        )
    probs = scores.softmax(-1)[..., : k.shape[0]]
    return (probs @ vv).transpose(0, 1)


def inputs(length):
    q = torch.randn(length, 16, 192, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(length, 2, 192, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(length, 2, 128, device="cuda", dtype=torch.bfloat16)
    sinks = torch.linspace(-0.9, 1.4, 16, device="cuda", dtype=torch.bfloat16)
    return q, k, v, sinks


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "FA4 requires SM90+",
)
class TestTextSWAFA4(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1011)
        self.left = make_attention(config()).attn.sliding_window_size

    def assert_reference(self, actual, expected):
        torch.testing.assert_close(actual.double(), expected, atol=3e-3, rtol=2e-2)

    def varlen(self, q, k, v, sinks, lengths, left):
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu,
            cu,
            max(lengths),
            max(lengths),
            causal=True,
            window_size=(left, 0),
            sinks=sinks,
            ver=4,
        )

    def test_packed_prefill_and_full_attention(self):
        lengths = [127, 128, 129, 257]
        q, k, v, sinks = inputs(sum(lengths))
        for left, count in ((self.left, 128), (-1, None)):
            expected = torch.cat(
                [
                    reference(qq, kk, vv, sinks, count)
                    for qq, kk, vv in zip(
                        q.split(lengths), k.split(lengths), v.split(lengths)
                    )
                ]
            )
            actual = self.varlen(q, k, v, sinks, lengths, left)
            self.assert_reference(actual, expected)

    def test_excluded_key_cannot_contribute(self):
        q, k, v, sinks = inputs(129)
        q.zero_()
        k.zero_()
        v.zero_()
        v[0].fill_(128)
        expected = reference(q, k, v, sinks)
        actual = self.varlen(q, k, v, sinks, [129], self.left)
        self.assert_reference(actual, expected)
        self.assertEqual(actual[-1].count_nonzero().item(), 0)
        self.assertGreater(actual[-2].abs().min().item(), 0.9)
        legacy = self.varlen(q, k, v, sinks, [129], 128)
        self.assertGreater(legacy[-1].abs().min().item(), 0.9)

    def test_paged_decode_and_chunked_prefill(self):
        page_size = 256
        for length in (127, 128, 129, 257):
            q, k, v, sinks = inputs(length)
            pages = math.ceil(length / page_size)
            # Reverse physical pages to exercise the page table rather than
            # relying on a contiguous identity mapping.
            table = torch.arange(pages - 1, -1, -1, device="cuda", dtype=torch.int32)[
                None
            ]
            kc = torch.zeros(pages, page_size, 2, 192, device="cuda", dtype=k.dtype)
            vc = torch.zeros(pages, page_size, 2, 128, device="cuda", dtype=v.dtype)
            for page in range(pages):
                start = page * page_size
                n = min(page_size, length - start)
                kc[pages - 1 - page, :n] = k[start : start + n]
                vc[pages - 1 - page, :n] = v[start : start + n]
            for q_len in (1, 5):
                with self.subTest(length=length, q_len=q_len):
                    qq = q[-q_len:]
                    actual = flash_attn_with_kvcache(
                        qq,
                        kc,
                        vc,
                        page_table=table,
                        cache_seqlens=torch.tensor(
                            [length], device="cuda", dtype=torch.int32
                        ),
                        cu_seqlens_q=torch.tensor(
                            [0, q_len], device="cuda", dtype=torch.int32
                        ),
                        max_seqlen_q=q_len,
                        causal=True,
                        window_size=(self.left, 0),
                        sinks=sinks,
                        ver=4,
                    )
                    self.assert_reference(actual, reference(qq, k, v, sinks))


if __name__ == "__main__":
    unittest.main()
