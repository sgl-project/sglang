import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.mimo_v2 import (
    _collect_mimo_qkv_adaptations,
    maybe_adapt_mimo_v2_fused_qkv_for_cp,
    repack_mimo_v2_fused_qkv_block_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _block_quantize_fixture(weight, block_size):
    block_n, block_k = block_size
    n, k = weight.shape
    padded = torch.zeros(
        (
            (n + block_n - 1) // block_n * block_n,
            (k + block_k - 1) // block_k * block_k,
        ),
        dtype=torch.float32,
    )
    padded[:n, :k] = weight
    blocks = padded.view(
        padded.shape[0] // block_n,
        block_n,
        padded.shape[1] // block_k,
        block_k,
    )
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-12)
    quantized = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return quantized.view_as(padded)[:n, :k].contiguous(), scale.contiguous()


_RELOAD_BLOCK_SIZE = [128, 128]
_RELOAD_Q_ROWS = 512
_RELOAD_K_ROWS = 384
_RELOAD_V_ROWS = 256
_RELOAD_WEIGHT_COLS = 256
_RELOAD_WEIGHT_ROWS = _RELOAD_Q_ROWS + _RELOAD_K_ROWS + _RELOAD_V_ROWS


def _reload_checkpoint_fixture():
    checkpoint_groups = []
    for rank in range(4):
        checkpoint_groups.append(
            torch.cat(
                [
                    torch.full((128, _RELOAD_WEIGHT_COLS), float(rank + 1)),
                    torch.full((96, _RELOAD_WEIGHT_COLS), float(16 * (rank + 1))),
                    torch.full((64, _RELOAD_WEIGHT_COLS), float(-(rank + 1))),
                ]
            )
        )
    quantized_groups = [
        _block_quantize_fixture(group, _RELOAD_BLOCK_SIZE)
        for group in checkpoint_groups
    ]
    return (
        torch.cat([item[0] for item in quantized_groups]),
        torch.cat([item[1] for item in quantized_groups]),
    )


def _packed_runtime_scale(fill_value):
    # DeepGEMM repeats each block-row scale across 128 weight rows, packs four
    # UE8M0 bytes into int32, and leaves this runtime-only shape behind.
    return torch.full(
        (_RELOAD_WEIGHT_ROWS, 1), fill_value, dtype=torch.int32
    ).contiguous()


class _FakeQKVParallelLinear:
    def __init__(self, runtime_scale, *, use_mxfp8=False):
        self.q_proj_shard_size = _RELOAD_Q_ROWS
        self.kv_proj_shard_size = _RELOAD_K_ROWS
        self.v_proj_shard_size = _RELOAD_V_ROWS
        self.weight = torch.nn.Parameter(
            torch.empty(
                (_RELOAD_WEIGHT_ROWS, _RELOAD_WEIGHT_COLS),
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = torch.nn.Parameter(runtime_scale, requires_grad=False)
        self.weight_scale_inv.format_ue8m0 = True
        self.weight_scale_inv.weight_loader = object()
        self.quant_method = SimpleNamespace(
            use_mxfp8=use_mxfp8,
            is_checkpoint_fp8_serialized=True,
            quant_config=SimpleNamespace(
                weight_block_size=_RELOAD_BLOCK_SIZE,
                use_mxfp8=use_mxfp8,
            ),
        )


def _reload_model(*qkv_projections):
    return SimpleNamespace(
        config=SimpleNamespace(
            architectures=["MiMoV2ForCausalLM"],
            attention_projection_layout="fused_qkv",
            num_key_value_heads=4,
        ),
        named_modules=lambda: [
            (f"model.layers.{index}.self_attn.qkv_proj", qkv_proj)
            for index, qkv_proj in enumerate(qkv_projections)
        ],
    )


class TestMiMoV2CPWeightAdapter(CustomTestCase):
    def test_reload_from_packed_ue8m0_is_reentrant_and_body_failure_restores(self):
        checkpoint_weight, checkpoint_scale = _reload_checkpoint_fixture()
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(17))
        model = _reload_model(qkv_proj)
        scale_param = qkv_proj.weight_scale_inv
        scale_loader = scale_param.weight_loader

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                self.assertIs(qkv_proj.weight_scale_inv, scale_param)
                self.assertIs(scale_param.weight_loader, scale_loader)
                self.assertEqual(tuple(scale_param.shape), (12, 2))
                self.assertEqual(scale_param.dtype, torch.float32)
                self.assertFalse(scale_param.format_ue8m0)
                qkv_proj.weight.data.copy_(checkpoint_weight)
                scale_param.data.copy_(checkpoint_scale)

            self.assertIs(qkv_proj.weight_scale_inv, scale_param)
            self.assertIs(scale_param.weight_loader, scale_loader)
            self.assertEqual(tuple(scale_param.shape), (9, 2))
            self.assertEqual(scale_param.dtype, torch.float32)
            self.assertFalse(scale_param.format_ue8m0)

            packed_after_postprocess = _packed_runtime_scale(29)
            packed_after_postprocess_copy = packed_after_postprocess.clone()
            packed_data_ptr = packed_after_postprocess.data_ptr()
            scale_param.data = packed_after_postprocess
            scale_param.format_ue8m0 = True

            with self.assertRaisesRegex(RuntimeError, "reload body failed"):
                with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                    self.assertIs(qkv_proj.weight_scale_inv, scale_param)
                    self.assertIs(scale_param.weight_loader, scale_loader)
                    self.assertEqual(tuple(scale_param.shape), (12, 2))
                    self.assertEqual(scale_param.dtype, torch.float32)
                    self.assertFalse(scale_param.format_ue8m0)
                    raise RuntimeError("reload body failed")

        self.assertIs(qkv_proj.weight_scale_inv, scale_param)
        self.assertIs(scale_param.weight_loader, scale_loader)
        self.assertEqual(scale_param.data_ptr(), packed_data_ptr)
        self.assertTrue(torch.equal(scale_param, packed_after_postprocess_copy))
        self.assertEqual(tuple(scale_param.shape), (_RELOAD_WEIGHT_ROWS, 1))
        self.assertEqual(scale_param.dtype, torch.int32)
        self.assertTrue(scale_param.format_ue8m0)

    def test_finish_failure_restores_every_packed_ue8m0_scale(self):
        checkpoint_weight, checkpoint_scale = _reload_checkpoint_fixture()
        qkv_projections = [
            _FakeQKVParallelLinear(_packed_runtime_scale(41)),
            _FakeQKVParallelLinear(_packed_runtime_scale(53)),
        ]
        model = _reload_model(*qkv_projections)
        original_scales = [
            qkv.weight_scale_inv.detach().clone() for qkv in qkv_projections
        ]
        original_data_ptrs = [
            qkv.weight_scale_inv.data_ptr() for qkv in qkv_projections
        ]
        repack_calls = 0

        def fail_second_finish(*args, **kwargs):
            nonlocal repack_calls
            repack_calls += 1
            if repack_calls == 2:
                raise RuntimeError("adapter finish failed")
            return repack_mimo_v2_fused_qkv_block_fp8(*args, **kwargs)

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.repack_mimo_v2_fused_qkv_block_fp8",
                side_effect=fail_second_finish,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "adapter finish failed"):
                with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                    for qkv_proj in qkv_projections:
                        qkv_proj.weight.data.copy_(checkpoint_weight)
                        qkv_proj.weight_scale_inv.data.copy_(checkpoint_scale)

        self.assertEqual(repack_calls, 2)
        for qkv_proj, original_scale, original_data_ptr in zip(
            qkv_projections, original_scales, original_data_ptrs
        ):
            scale = qkv_proj.weight_scale_inv
            self.assertEqual(scale.data_ptr(), original_data_ptr)
            self.assertTrue(torch.equal(scale, original_scale))
            self.assertEqual(tuple(scale.shape), (_RELOAD_WEIGHT_ROWS, 1))
            self.assertEqual(scale.dtype, torch.int32)
            self.assertTrue(scale.format_ue8m0)

    def test_serialized_mxfp8_qkv_is_rejected_by_quant_method_semantics(self):
        qkv_proj = _FakeQKVParallelLinear(
            torch.ones((9, 2), dtype=torch.float32), use_mxfp8=True
        )
        model = _reload_model(qkv_proj)

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "serialized MXFP8 QKV"):
                _collect_mimo_qkv_adaptations(model)

    def test_mtp_draft_input_embedding_clamps_sentinel_ids(self):
        from sglang.srt.model_executor.runner.eager_runner import (
            _get_cp_v2_input_embeds,
        )

        embedding = torch.nn.Embedding.from_pretrained(
            torch.arange(12, dtype=torch.float32).reshape(4, 3)
        )

        class DraftModel:
            config = SimpleNamespace(
                architectures=["MiMoV2MTP"],
                vocab_size=4,
            )

            def get_input_embedding(self, input_ids):
                raise AssertionError(
                    "MiMoV2MTP inherits an unavailable singular accessor"
                )

            def get_input_embeddings(self):
                return embedding

        input_ids = torch.tensor([-100, 1, 999])
        expected = embedding(torch.tensor([0, 1, 3]))

        self.assertTrue(
            torch.equal(_get_cp_v2_input_embeds(DraftModel(), input_ids), expected)
        )

    def test_mtp_draft_qkv_is_collected_for_cp_repacking(self):
        class FakeQKVParallelLinear:
            def __init__(self):
                self.q_proj_shard_size = 32
                self.kv_proj_shard_size = 24
                self.v_proj_shard_size = 16
                self.weight = torch.empty((72, 4), dtype=torch.float8_e4m3fn)
                self.weight_scale_inv = torch.empty((18, 2), dtype=torch.float32)
                self.quant_method = SimpleNamespace(
                    quant_config=SimpleNamespace(weight_block_size=[4, 2])
                )

        qkv_proj = FakeQKVParallelLinear()
        model = SimpleNamespace(
            config=SimpleNamespace(
                architectures=["MiMoV2MTP"],
                attention_projection_layout="fused_qkv",
                num_key_value_heads=4,
            ),
            named_modules=lambda: [("model.mtp_block.self_attn.qkv_proj", qkv_proj)],
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            adaptations = _collect_mimo_qkv_adaptations(model)

        self.assertEqual(len(adaptations), 1)
        self.assertEqual(
            adaptations[0].module_name,
            "model.mtp_block.self_attn.qkv_proj",
        )

    def test_repack_fused_tp4_qkv_requantizes_across_block_boundaries(self):
        block_size = [4, 2]
        q_values = [1.0, 2.0, 4.0, 8.0]
        k_values = [16.0, 32.0, 64.0, 128.0]
        v_values = [-1.0, -2.0, -4.0, -8.0]

        checkpoint_groups = []
        for rank in range(4):
            checkpoint_groups.append(
                torch.cat(
                    [
                        torch.full((8, 4), q_values[rank]),
                        torch.full((6, 4), k_values[rank]),
                        torch.full((4, 4), v_values[rank]),
                    ]
                )
            )

        quantized_groups = [
            _block_quantize_fixture(group, block_size) for group in checkpoint_groups
        ]
        checkpoint_weight = torch.cat([item[0] for item in quantized_groups])
        checkpoint_scale = torch.cat([item[1] for item in quantized_groups])

        repacked_weight, repacked_scale = repack_mimo_v2_fused_qkv_block_fp8(
            checkpoint_weight,
            checkpoint_scale,
            q_rows=32,
            k_rows=24,
            v_rows=16,
            checkpoint_tp_size=4,
            block_size=block_size,
            output_dtype=torch.float32,
        )
        actual = block_quant_dequant(
            repacked_weight,
            repacked_scale,
            block_size,
            torch.float32,
        )
        expected = torch.cat(
            [
                *(torch.full((8, 4), value) for value in q_values),
                *(torch.full((6, 4), value) for value in k_values),
                *(torch.full((4, 4), value) for value in v_values),
            ]
        )

        self.assertEqual(tuple(checkpoint_weight.shape), (72, 4))
        self.assertEqual(tuple(checkpoint_scale.shape), (20, 2))
        self.assertEqual(tuple(repacked_weight.shape), (72, 4))
        self.assertEqual(tuple(repacked_scale.shape), (18, 2))
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
