"""DSpark SM70 packs CSA2 rings before sparse."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_v4_dspark import (
    DeepseekV4ForCausalLMDSpark,
    _dspark_core_attention,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsv41DsparkSm70Csa2(CustomTestCase):
    def test_sm70_packs_then_sparse_and_strips_head_pad(self):
        calls = []
        layer = SimpleNamespace(
            n_local_heads=8,
            attn=SimpleNamespace(layer_id=37),
        )
        q = torch.zeros(5, 64, 4)
        kv = torch.zeros(5, 4)
        hidden = torch.zeros(5, 4)
        positions = torch.arange(5)
        backend = object()
        sink = torch.zeros(64)

        def pack(layer_arg, x, q_lora, positions_arg, forward_batch):
            calls.append(("pack", layer_arg is layer, tuple(x.shape)))

        def attn(q_attn, kv_arg, attn_layer, ratio, sink_arg, forward_batch, save):
            calls.append(("attn", tuple(q_attn.shape), ratio, save))
            return torch.zeros(q_attn.shape)

        plat = SimpleNamespace(is_sm70=True)
        with (
            patch(
                "sglang.srt.models.deepseek_v4_dspark.get_platform",
                return_value=plat,
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.bcg_csa2_low_ratio_sources",
                side_effect=pack,
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.bcg_csa2_attention_with_output",
                side_effect=attn,
            ),
        ):
            out = _dspark_core_attention(
                layer,
                hidden,
                q,
                kv,
                positions,
                object(),
                backend,
                sink,
            )

        self.assertEqual(calls[0][0], "pack")
        self.assertTrue(calls[0][1])
        self.assertEqual(calls[1][0], "attn")
        self.assertEqual(calls[1][1], (5, 8, 4))
        self.assertEqual(calls[1][2], 0)
        self.assertFalse(calls[1][3])
        self.assertEqual(tuple(out.shape), (5, 8, 4))

    def test_none_layer_idx_skips_scatter(self):
        from sglang.srt.eplb.expert_distribution import (
            _SelectExpertsSinglePassGatherer,
        )

        g = SimpleNamespace(
            _data=torch.zeros(40, 384, dtype=torch.int),
            _expert_location_metadata=SimpleNamespace(num_physical_experts=384),
        )
        ids = torch.tensor([[1, 2, 3]])
        _SelectExpertsSinglePassGatherer.on_select_experts(g, None, ids)
        self.assertEqual(int(g._data.sum()), 0)
        _SelectExpertsSinglePassGatherer.on_select_experts(g, 37, ids)
        self.assertEqual(int(g._data[37, 1:4].sum()), 3)

    def test_non_sm70_uses_pool_backend(self):
        seen = []
        layer = SimpleNamespace(
            n_local_heads=8,
            attn=SimpleNamespace(layer_id=37),
        )
        q = torch.zeros(5, 64, 4)
        kv = torch.zeros(5, 4)

        class Backend:
            def forward(self, **kwargs):
                seen.append(kwargs)
                return torch.zeros(5, 64, 4)

        plat = SimpleNamespace(is_sm70=False)
        with patch(
            "sglang.srt.models.deepseek_v4_dspark.get_platform",
            return_value=plat,
        ):
            out = _dspark_core_attention(
                layer,
                torch.zeros(5, 4),
                q,
                kv,
                torch.arange(5),
                object(),
                Backend(),
                torch.zeros(64),
            )
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["compress_ratio"], 0)
        self.assertIs(seen[0]["q"], q)
        self.assertFalse(seen[0]["save_kv_cache"])
        self.assertEqual(tuple(out.shape), (5, 64, 4))

    def test_write_target_hidden_kv_packs_draft_csa2_on_sm70(self):
        attn = SimpleNamespace(
            layer_id=1,
            wkv=None,
            kv_norm=SimpleNamespace(weight=torch.ones(4)),
            eps=1e-5,
            freqs_cis=torch.zeros(32, 4),
        )
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        model.stages = [SimpleNamespace(self_attn=attn)]
        model.project_target_hidden = lambda h: h
        stored = []
        packed = []
        kv = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        backend = object()
        pool = SimpleNamespace(
            set_swa_key_buffer_radix_fused_norm_rope=lambda **kwargs: stored.append(
                kwargs
            )
        )
        swa_loc = torch.tensor([-1, 0, 1, 2, -1], dtype=torch.int32)
        positions = torch.arange(5)

        with (
            patch(
                "sglang.srt.models.deepseek_v4_dspark.get_platform",
                return_value=SimpleNamespace(is_sm70=True),
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.is_unified_kv_triton",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.CommitKvProj.execute",
                return_value=[kv],
            ),
            patch(
                "sglang.srt.layers.attention.dsv4.sm70_csa2.pack_swa_window_from_kv",
                side_effect=lambda *args, **kwargs: packed.append((args, kwargs)),
            ),
        ):
            DeepseekV4ForCausalLMDSpark.write_target_hidden_kv(
                model,
                main_hidden=torch.zeros(5, 4),
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
                attn_backend=backend,
            )

        self.assertEqual(len(stored), 0)
        self.assertEqual(len(packed), 1)
        args, kwargs = packed[0]
        self.assertIs(args[0], backend)
        self.assertIs(args[1], attn)
        self.assertTrue(torch.equal(args[2], kv))
        self.assertTrue(torch.equal(args[3], positions))
        self.assertTrue(torch.equal(kwargs["mask"], swa_loc >= 0))

    def test_write_target_hidden_kv_stores_pool_off_sm70(self):
        attn = SimpleNamespace(
            layer_id=1,
            wkv=None,
            kv_norm=SimpleNamespace(weight=torch.ones(4)),
            eps=1e-5,
            freqs_cis=torch.zeros(32, 4),
        )
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        model.stages = [SimpleNamespace(self_attn=attn)]
        model.project_target_hidden = lambda h: h
        stored = []
        kv = torch.zeros(2, 4)
        pool = SimpleNamespace(
            set_swa_key_buffer_radix_fused_norm_rope=lambda **kwargs: stored.append(
                kwargs
            )
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v4_dspark.get_platform",
                return_value=SimpleNamespace(is_sm70=False),
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.is_unified_kv_triton",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.CommitKvProj.execute",
                return_value=[kv],
            ),
        ):
            DeepseekV4ForCausalLMDSpark.write_target_hidden_kv(
                model,
                main_hidden=torch.zeros(2, 4),
                swa_loc=torch.arange(2, dtype=torch.int32),
                positions=torch.arange(2),
                pool=pool,
            )
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["layer_id"], 1)
        self.assertTrue(torch.equal(stored[0]["kv"], kv))

    def test_write_target_hidden_kv_sm70_requires_draft_backend(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        model.stages = [SimpleNamespace(self_attn=SimpleNamespace(wkv=None))]
        model.project_target_hidden = lambda h: h
        pool = SimpleNamespace(set_swa_key_buffer_radix_fused_norm_rope=lambda **k: None)
        with (
            patch(
                "sglang.srt.models.deepseek_v4_dspark.get_platform",
                return_value=SimpleNamespace(is_sm70=True),
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.is_unified_kv_triton",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.CommitKvProj.execute",
                return_value=[torch.zeros(1, 4)],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "draft attn backend"):
                DeepseekV4ForCausalLMDSpark.write_target_hidden_kv(
                    model,
                    main_hidden=torch.zeros(1, 4),
                    swa_loc=torch.zeros(1, dtype=torch.int32),
                    positions=torch.zeros(1, dtype=torch.int64),
                    pool=pool,
                )


if __name__ == "__main__":
    unittest.main()
