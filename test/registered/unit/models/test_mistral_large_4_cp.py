import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.arg_groups.model_overrides.deepseek_v2 import _dsa_dcp_overrides
from sglang.srt.environ import envs
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.model_executor.runner import eager_runner
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models import mistral_large_4
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
from sglang.srt.models.mistral_large_4 import (
    MistralLarge4AttentionMLA,
    MistralLarge4ForCausalLM,
    MistralLarge4Model,
)
from sglang.srt.models.pixtral import PixtralForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _parallel(**overrides):
    values = dict(
        attn_cp_size=1,
        dcp_enabled=False,
        cp_strategy=None,
        enable_dsa_cache_layer_split=False,
        pp_size=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _build(parallel):
    def fake_init(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.logits_processor = SimpleNamespace()

    config = SimpleNamespace(index_kv_dtype="fp8", softmax_tanh_gating=30.0)
    with (
        mock.patch.object(mistral_large_4, "get_parallel", return_value=parallel),
        mock.patch.object(DeepseekV3ForCausalLM, "__init__", fake_init),
    ):
        return MistralLarge4ForCausalLM(config)


class TestContextParallelLayouts(CustomTestCase):
    def test_supported_layouts_are_accepted(self):
        _build(_parallel())
        _build(_parallel(attn_cp_size=2, cp_strategy="interleave"))
        _build(_parallel(dcp_enabled=True))

    def test_unsupported_layouts_are_rejected(self):
        cases = {
            "zigzag": _parallel(attn_cp_size=2, cp_strategy="zigzag"),
            "decode CP with pipeline": _parallel(dcp_enabled=True, pp_size=2),
            "layer split": _parallel(
                attn_cp_size=2,
                cp_strategy="interleave",
                enable_dsa_cache_layer_split=True,
            ),
            "pipeline": _parallel(attn_cp_size=2, cp_strategy="interleave", pp_size=2),
        }
        for name, parallel in cases.items():
            with self.subTest(name), self.assertRaises(NotImplementedError):
                _build(parallel)


class TestDecodeContextParallelOverrides(CustomTestCase):
    def test_dspark_under_dsa_dcp_requires_static_verify(self):
        def overrides(ragged_mode):
            cfg = SimpleNamespace(
                dcp_size=2,
                enable_hisparse=False,
                enable_hierarchical_cache=False,
                disaggregation_mode="null",
                speculative_eagle_topk=None,
                speculative_algorithm="DSPARK",
                dsa_prefill_backend=None,
                dsa_decode_backend=None,
                dcp_replicate_q_proj=None,
                dcp_comm_backend=None,
            )
            with envs.SGLANG_RAGGED_VERIFY_MODE.override(ragged_mode):
                return _dsa_dcp_overrides(cfg)

        self.assertEqual(
            overrides("static"),
            {"dsa_prefill_backend": "trtllm", "dsa_decode_backend": "trtllm"},
        )
        with self.assertRaisesRegex(ValueError, "SGLANG_RAGGED_VERIFY_MODE=static"):
            overrides("compact")


class TestKVSharingConsumerUnderPrefillCP(CustomTestCase):
    def test_only_kv_owning_layers_gather_kv(self):
        k_nope = torch.zeros(3, 1, 4)
        k_pe = torch.zeros(3, 1, 2)
        gathered = (torch.ones(6, 1, 4), torch.ones(6, 1, 2))
        strategy = mock.Mock()
        strategy.materialize_full_mla_kv.return_value = gathered
        provider = SimpleNamespace(save_kv_cache=True, attn_mqa=object())
        consumer = SimpleNamespace(save_kv_cache=False, attn_mqa=provider.attn_mqa)

        with (
            mock.patch.object(dsa_backend, "dsa_use_prefill_cp", return_value=True),
            mock.patch.object(dsa_backend, "is_cp_active", return_value=True),
            mock.patch.object(dsa_backend, "get_cp_strategy", return_value=strategy),
        ):
            consumer_kv = dsa_backend.prepare_kv_for_attention(
                consumer, object(), k_nope, k_pe, defer_materialization=False
            )
            strategy.materialize_full_mla_kv.assert_not_called()
            provider_kv = dsa_backend.prepare_kv_for_attention(
                provider, object(), k_nope, k_pe, defer_materialization=False
            )

        self.assertIs(consumer_kv[0], k_nope)
        self.assertIs(consumer_kv[1], k_pe)
        self.assertIs(provider_kv, gathered)
        strategy.materialize_full_mla_kv.assert_called_once()


class _Attention(nn.Module):
    def __init__(self, layer_id, kv_source_layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.kv_source_layer_id = kv_source_layer_id
        self.attn_mqa = nn.Identity()
        self.attn_mha = nn.Identity()
        self.attn_mqa_for_dcp_decode = nn.Identity()


class TestAttentionUnderDecodeCP(CustomTestCase):
    def test_consumer_attends_through_the_provider_dcp_module(self):
        kv_sources = [0, 1, 1]
        model = MistralLarge4Model.__new__(MistralLarge4Model)
        nn.Module.__init__(model)
        model.layers = [
            SimpleNamespace(self_attn=_Attention(layer_id, source))
            for layer_id, source in enumerate(kv_sources)
        ]
        model.start_layer, model.end_layer = 0, len(kv_sources)

        model._share_provider_attention()

        provider, consumer = model.layers[1].self_attn, model.layers[2].self_attn
        self.assertIs(
            consumer.attn_mqa_for_dcp_decode, provider.attn_mqa_for_dcp_decode
        )
        self.assertIsNot(
            model.layers[0].self_attn.attn_mqa_for_dcp_decode,
            provider.attn_mqa_for_dcp_decode,
        )

    def test_query_is_gathered_not_replicated(self):
        parallel = SimpleNamespace(dcp_replicate_q_proj=True)
        attn = SimpleNamespace(dcp_size=2)
        decode = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: True, is_target_verify=lambda: False
            )
        )
        with mock.patch.object(forward_mla, "get_parallel", return_value=parallel):
            self.assertFalse(
                MistralLarge4AttentionMLA.is_dcp_q_replicate_phase(attn, decode)
            )

    def test_nope_layer_dcp_module_uses_the_nope_scaling(self):
        attn = MistralLarge4AttentionMLA.__new__(MistralLarge4AttentionMLA)
        nn.Module.__init__(attn)
        attn.qk_nope_head_dim, attn.hidden_size, attn.num_heads = 128, 8, 2
        attn.kv_lora_rank = 4
        attn.attn_mqa = SimpleNamespace(scaling=192**-0.5)
        attn.attn_mha = SimpleNamespace(scaling=192**-0.5)
        attn.attn_mqa_for_dcp_decode = SimpleNamespace(scaling=192**-0.5)

        with (
            mock.patch.object(mistral_large_4, "ColumnParallelLinear"),
            mock.patch.object(mistral_large_4, "ReplicatedLinear"),
        ):
            attn._build_nope_projections(quant_config=None, prefix="")

        self.assertEqual(attn.attn_mqa_for_dcp_decode.scaling, 128**-0.5)
        self.assertEqual(attn.attn_mqa.scaling, 128**-0.5)


class _Body(nn.Module):
    def __init__(self, capture_aux_hidden_states: bool):
        super().__init__()
        self.capture_aux_hidden_states = capture_aux_hidden_states

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if self.capture_aux_hidden_states:
            return input_embeds * 2, input_embeds * 3
        return input_embeds * 2


class _LanguageModel(nn.Module):
    def __init__(self, capture_aux_hidden_states: bool):
        super().__init__()
        self.model = _Body(capture_aux_hidden_states)
        self.capture_aux_hidden_states = capture_aux_hidden_states
        self.lm_head = nn.Identity()
        self.pp_group = SimpleNamespace(is_last_rank=True)
        self.logits_processor = mock.Mock(return_value="logits")


class TestPixtralUnderPrefillCP(CustomTestCase):
    def _run_extend_cp(self, capture_aux_hidden_states: bool):
        wrapper = PixtralForConditionalGeneration.__new__(
            PixtralForConditionalGeneration
        )
        nn.Module.__init__(wrapper)
        wrapper.language_model = _LanguageModel(capture_aux_hidden_states)
        runner = EagerRunner.__new__(EagerRunner)
        runner.model_runner = SimpleNamespace(model=wrapper)
        input_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
        forward_batch = SimpleNamespace(
            input_ids=torch.tensor([1, 2, 3]),
            positions=torch.tensor([0, 1, 2]),
            contains_mm_inputs=lambda: False,
        )

        @contextlib.contextmanager
        def no_shard(embeds, positions, batch, input_ids):
            yield embeds, positions, input_ids

        with (
            mock.patch.object(eager_runner, "cp_shard_model_inputs", no_shard),
            mock.patch.object(eager_runner, "cp_gather_after_forward", lambda h, *_: h),
            mock.patch("torch.cuda.current_stream", return_value=None),
        ):
            out = runner._execute_extend_cp(
                forward_batch, {"input_embeds": input_embeds}
            )

        self.assertEqual(out, "logits")
        return wrapper.language_model.logits_processor.call_args, input_embeds

    def test_cp_runner_reaches_the_language_model_body(self):
        call, input_embeds = self._run_extend_cp(capture_aux_hidden_states=False)

        self.assertTrue(torch.equal(call.args[1], input_embeds * 2))

    def test_cp_runner_passes_captured_aux_hidden_states(self):
        call, input_embeds = self._run_extend_cp(capture_aux_hidden_states=True)

        self.assertTrue(torch.equal(call.args[1], input_embeds * 2))
        self.assertTrue(torch.equal(call.args[4], input_embeds * 3))
        self.assertNotIn("hidden_states_before_norm", call.kwargs)


class TestSharedKVRaggedPrefill(CustomTestCase):
    def _run_ragged(self, implementation, *, shared, q8_fusion=False):
        provider_cache = torch.full((2, 1, 4), 7.0)
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return kwargs["q_all"] if "q_all" in kwargs else kwargs["q_nope"]

        backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
            dsa_backend.DeepseekSparseAttnBackend
        )
        backend.__dict__.update(
            dsa_prefill_impl=implementation,
            dsa_decode_impl="fa3",
            use_mha=False,
            use_fused_topk=True,
            hisparse_coordinator=None,
            has_shared_kv_consumers=shared,
            forward_metadata=SimpleNamespace(
                page_table_1_flattened=None,
                page_table_1=torch.tensor([[0, 1]]),
                indexer_seq_lens_cpu=torch.tensor([2]),
                dsa_cache_seqlens_int32=torch.tensor([1, 2]),
            ),
            token_to_kv_pool=SimpleNamespace(
                layer_router=SimpleNamespace(
                    read=lambda _: SimpleNamespace(key=lambda: provider_cache)
                )
            ),
            _resolve_kpool_tail_backend=lambda indices, impl: impl,
            _check_kpool_tail_backend=lambda *args: None,
            get_topk_transform_method=lambda _: dsa_backend.TopkTransformMethod.RAGGED,
            _pad_topk_indices=lambda indices, n: indices,
            _get_fused_topk_page_table=lambda indices: indices,
            _q8kv8_kv_cat_fusion=q8_fusion,
            _forward_flashmla_sparse=capture,
            _forward_flashmla_sparse_q8kv8=capture,
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=35,
            tp_q_head_num=1,
            v_head_dim=2,
            head_dim=4,
            scaling=1.0,
        )
        mode = SimpleNamespace(
            is_target_verify=lambda: False, is_draft_extend_v2=lambda: False
        )
        batch = SimpleNamespace(forward_mode=mode, extend_prefix_lens_cpu=[0])
        zeros = torch.zeros((2, 1, 2))
        with (
            mock.patch.object(
                dsa_backend,
                "dequantize_k_cache_paged_dsa",
                side_effect=lambda cache, _: cache,
            ),
            mock.patch.object(
                dsa_backend,
                "concat_mla_absorb_q_general",
                side_effect=lambda q, rope: torch.cat((q, rope), dim=-1),
            ),
        ):
            dsa_backend.DeepseekSparseAttnBackend.forward_extend(
                backend,
                zeros,
                zeros.clone(),
                zeros.clone(),
                layer,
                batch,
                save_kv_cache=False,
                q_rope=zeros.clone(),
                k_rope=zeros.clone(),
                topk_indices=torch.tensor([[0, -1], [0, 1]], dtype=torch.int32),
            )
        return provider_cache, captured

    def test_fresh_ragged_prefill_reads_expected_kv(self):
        """Shared consumers use provider KV; ordinary models keep fresh inputs."""
        provider, captured = self._run_ragged("flashmla_sparse", shared=True)
        self.assertIs(captured["kv_cache"], provider)

        for fusion in (False, True):
            with self.subTest(fusion=fusion):
                provider, captured = self._run_ragged(
                    "flashmla_sparse_q8", shared=True, q8_fusion=fusion
                )
                self.assertIs(captured["paged_kv_cache"], provider)
                self.assertIsNone(captured["kv_bf16"])
                self.assertIsNotNone(captured["page_table_1_flattened"])

        provider, captured = self._run_ragged("flashmla_sparse", shared=False)
        self.assertIsNot(captured["kv_cache"], provider)
        self.assertEqual(captured["kv_cache"].abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
