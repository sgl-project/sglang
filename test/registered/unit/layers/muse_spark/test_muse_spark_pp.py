import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.muse_spark import MuseSpark1xModel, MuseSparkTextForCausalLM
from sglang.srt.models.muse_spark_omni import MuseSparkForConditionalGeneration
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ResidualLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.self_attn_input_norm = nn.Identity()

    def forward(self, positions, stream, attn_input, batch):
        stream = stream + (self.index + 1) / 1024
        return stream, stream


def _stage(pp, rank, taps):
    model = MuseSpark1xModel.__new__(MuseSpark1xModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    model.residual_dtype = torch.float32
    model.operator_dtype = torch.bfloat16
    model.pp_group = SimpleNamespace(
        world_size=pp, is_first_rank=rank == 0, is_last_rank=rank == pp - 1
    )
    model.start_layer, model.end_layer = get_pp_indices(62, rank, pp)
    model.layers = nn.ModuleList(
        _ResidualLayer(index)
        if model.start_layer <= index < model.end_layer
        else PPMissingLayer()
        for index in range(62)
    )
    model.embed_norm = None
    model.norm = nn.Identity()
    model.layers_to_capture = taps
    return model


class _CarryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn_input_norm = nn.Identity()

    def forward(self, positions, stream, attn_input, batch):
        return stream, stream


class _RecordNorm(nn.Identity):
    def forward(self, value):
        self.observed = value.clone()
        return value


class _WeightLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Linear(4, 4, bias=False)
        self.post_attention_residual_gate = nn.Identity()
        self.post_feedforward_residual_gate = nn.Identity()


def _weight_model(pp, rank):
    model = MuseSparkTextForCausalLM.__new__(MuseSparkTextForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(tie_word_embeddings=False, num_hidden_layers=4)
    model.nvfp4_checkpoint = model.int4_checkpoint = False
    model.model = nn.Module()
    text = model.model.language_model = nn.Module()
    text.start_layer, text.end_layer = get_pp_indices(4, rank, pp)
    text.layers = nn.ModuleList(
        _WeightLayer()
        if text.start_layer <= index < text.end_layer
        else PPMissingLayer()
        for index in range(4)
    )
    return model


class TestMuseSparkPP(CustomTestCase):
    def test_dummy_runner_preserves_non_muse_aux_allocation(self):
        from sglang.srt.model_executor.runner.base_runner import BaseRunner

        with get_context().override_server_args(model_path="dummy"):
            for model_type in ("muse_spark_text", "kimi_k2", None):
                with (
                    self.subTest(model_type=model_type),
                    get_parallel().override(pp_size=2),
                ):
                    mr = SimpleNamespace(
                        device=torch.device("cpu"),
                        model_config=SimpleNamespace(
                            hidden_size=4,
                            vocab_size=8,
                            dtype=torch.bfloat16,
                            is_encoder_decoder=False,
                            hf_text_config=SimpleNamespace(model_type=model_type),
                        ),
                        attn_backend=SimpleNamespace(
                            get_cuda_graph_seq_len_fill_value=lambda: 16
                        ),
                        ngram_embedding_manager=SimpleNamespace(enabled=False),
                        get_pp_proxy_topk_size=lambda: None,
                        get_pp_proxy_residual_num_blocks=lambda: None,
                        get_pp_proxy_dspark_hidden_size=lambda: 12,
                    )
                    buffers = BaseRunner._alloc_dummy_decode_buffers(
                        SimpleNamespace(model_runner=mr),
                        max_bs=2,
                        num_tokens_per_req=8,
                    )
                    self.assertEqual(
                        "dspark_hidden_states" in buffers.pp_proxy_tensors,
                        model_type == "muse_spark_text",
                    )
                    if model_type == "muse_spark_text":
                        self.assertEqual(
                            tuple(
                                buffers.pp_proxy_tensors["dspark_hidden_states"].shape
                            ),
                            (16, 12),
                        )

    def test_partitioned_forward_preserves_residual_and_global_tap_order(self):
        embeddings = torch.tensor(
            [[1, -2, 3, -4], [2, -1, 4, -3]], dtype=torch.bfloat16
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with get_flags().sp.override(enabled=False):
            for pp in (1, 2, 4, 8):
                for taps in ([1, 15, 31, 47, 61], [0, 31]):
                    with self.subTest(pp=pp, taps=taps):
                        proxy = None
                        for rank in range(pp):
                            result = _stage(pp, rank, taps).forward(
                                None,
                                torch.arange(2),
                                batch,
                                input_embeds=embeddings if rank == 0 else None,
                                pp_proxy_tensors=proxy,
                            )
                            if rank < pp - 1:
                                self.assertIsInstance(result, PPProxyTensors)
                                proxy = result
                        hidden, captured = result
                        expected = embeddings.float() + sum(range(1, 63)) / 1024
                        torch.testing.assert_close(
                            hidden, expected.bfloat16(), rtol=0, atol=0
                        )
                        self.assertEqual(len(captured), len(taps))
                        for tap, actual in zip(taps, captured):
                            expected_tap = (
                                embeddings.float() + sum(range(1, tap + 2)) / 1024
                            )
                            torch.testing.assert_close(
                                actual, expected_tap.bfloat16(), rtol=0, atol=0
                            )

    def test_fp32_proxy_residual_obeys_rounding_bound_and_uses_tail(self):
        generator = torch.Generator().manual_seed(42)
        values = torch.randn(4096, 4, generator=generator) * torch.exp2(
            torch.randint(-30, 31, (4096, 4), generator=generator).float()
        )
        first, second = _stage(2, 0, []), _stage(2, 1, [])
        first.layers = nn.ModuleList(_CarryLayer() for _ in range(62))
        second.layers = nn.ModuleList(_CarryLayer() for _ in range(62))
        record = _RecordNorm()
        second.layers[second.start_layer].self_attn_input_norm = record
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with get_flags().sp.override(enabled=False):
            proxy = first.forward(None, torch.arange(4096), batch, input_embeds=values)
            second.forward(None, torch.arange(4096), batch, pp_proxy_tensors=proxy)
        error = (record.observed.double() - values.double()).abs()
        bound = values.double().abs() * (2**-16)
        self.assertTrue(torch.all(error <= bound).item())
        self.assertTrue(torch.any(error > 0).item())
        head_only_error = (proxy["hidden_states"].double() - values.double()).abs()
        self.assertLess(error.mean().item(), head_only_error.mean().item() / 100)

    def test_checkpoint_loading_filters_only_other_pipeline_stages(self):
        weights = [
            (
                f"model.language_model.layers.{i}.input_layernorm.weight",
                torch.full((4, 4), i + 1.0),
            )
            for i in range(4)
        ]
        with get_context().override_server_args(model_path="dummy"):
            for pp, rank in ((1, 0), (2, 0), (2, 1)):
                with (
                    self.subTest(pp=pp, rank=rank),
                    get_parallel().override(pp_size=pp),
                ):
                    model = _weight_model(pp, rank)
                    model.load_weights(weights)
                    start, end = get_pp_indices(4, rank, pp)
                    params = dict(model.named_parameters())
                    self.assertEqual(len(params), end - start)
                    for index in range(start, end):
                        key = f"model.language_model.layers.{index}.input_layernorm.weight"
                        self.assertTrue(torch.equal(params[key], weights[index][1]))
            for pp in (1, 2):
                with self.subTest(pp=pp), get_parallel().override(pp_size=pp):
                    model = _weight_model(pp, 0)
                    with self.assertRaisesRegex(
                        ValueError, "outside the Muse Spark text model"
                    ):
                        model.load_weights(
                            [
                                (
                                    "model.language_model.layers.9.input_layernorm.weight",
                                    torch.ones(4, 4),
                                )
                            ]
                        )

    def test_omni_pp1_does_not_require_a_backend_model_pp_group(self):
        model = MuseSparkForConditionalGeneration.__new__(
            MuseSparkForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.language_model = nn.Identity()
        embeddings = torch.ones(2, 4)
        expected = torch.full((2, 4), 7.0)
        with (
            get_context().override_server_args(model_path="dummy"),
            get_parallel().override(
                pp_size=1, pp_group=SimpleNamespace(is_first_rank=True)
            ),
            patch.object(MuseSparkTextForCausalLM, "forward", return_value=expected),
        ):
            actual = model.forward(
                None, torch.arange(2), SimpleNamespace(), input_embeds=embeddings
            )
        self.assertIs(actual, expected)


if __name__ == "__main__":
    unittest.main()
