import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.layernorm import RMSNorm  # noqa: E402
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod  # noqa: E402
from sglang.srt.mem_cache.kv_cache_builder import get_draft_kv_pool  # noqa: E402
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig  # noqa: E402
from sglang.srt.model_executor.runner.base_runner import (  # noqa: E402
    _allocate_decode_buffers,
)
from sglang.srt.model_executor.runner_utils.buffers import (  # noqa: E402
    DecodeInputBuffers,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: E402
from sglang.srt.models.deepseek_v4_dspark import (  # noqa: E402
    DeepseekV4ForCausalLMDSpark,
    _BlockFp8LinearSlice,
)
from sglang.srt.models.dflash import DFlashDraftModel  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_config import (  # noqa: E402
    resolve_single_owner_pp_rank,
    use_lifecycle_only_draft_model,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (  # noqa: E402
    DSparkWorkerV2,
    _is_context_only_pp_prefill_rank,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _TupleLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        return self.linear(hidden_states), None


def _make_deepseek_v4_dspark_projection_model(
    *, hidden_size: int, num_target_features: int
) -> DeepseekV4ForCausalLMDSpark:
    model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=hidden_size)
    model.num_target_features = num_target_features
    stage = torch.nn.Module()
    stage.main_proj = _TupleLinear(hidden_size * num_target_features, hidden_size)
    stage.main_norm = RMSNorm(hidden_size, eps=1e-6)
    model.stages = torch.nn.ModuleList([stage])
    model.markov_head = torch.nn.Identity()
    model.confidence_head = torch.nn.Identity()
    model.embed_tokens = None
    model.lm_head = None
    model._partial_feature_indices = ()
    model._partial_main_proj = None
    return model


class TestDSparkPPContext(CustomTestCase):
    def test_full_projection_fast_path_requires_final_pp_owner(self):
        """Only final-rank ownership can bypass the ctx_acc handoff."""
        with patch.dict(
            os.environ,
            {"SGLANG_PP_LAYER_PARTITION": "6,5,6,5,6,5,5,5"},
        ):
            self.assertEqual(
                resolve_single_owner_pp_rank(
                    target_layer_ids=[40, 41, 42],
                    num_hidden_layers=43,
                    pp_size=8,
                ),
                7,
            )
            self.assertIsNone(
                resolve_single_owner_pp_rank(
                    target_layer_ids=[35, 40],
                    num_hidden_layers=43,
                    pp_size=8,
                )
            )

    def test_lifecycle_only_draft_model_is_limited_to_non_owner_ranks(self):
        common = dict(
            disaggregation_mode="prefill",
            pp_size=8,
            target_layer_ids=[40, 41, 42],
            num_hidden_layers=43,
        )
        for pp_rank in range(7):
            self.assertTrue(use_lifecycle_only_draft_model(pp_rank=pp_rank, **common))
        self.assertFalse(use_lifecycle_only_draft_model(pp_rank=7, **common))
        self.assertFalse(
            use_lifecycle_only_draft_model(
                pp_rank=0,
                **{**common, "target_layer_ids": [35, 40]},
            )
        )

    def test_lifecycle_only_model_does_not_consume_checkpoint_weights(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        torch.nn.Module.__init__(model)
        model.is_lifecycle_only = True

        def weights():
            raise AssertionError("lifecycle-only model consumed checkpoint weights")
            yield

        model.load_weights(weights())

    def test_block_fp8_projection_slice_selects_matching_weight_and_scale_blocks(self):
        feature_width = 128
        output_size = 2
        quant_method = Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            )
        )
        weight = torch.arange(
            output_size * feature_width * 3, dtype=torch.float32
        ).reshape(output_size, feature_width * 3)
        weight_scale = torch.tensor([[11.0, 22.0, 33.0]])
        source = SimpleNamespace(
            quant_method=quant_method,
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale_inv=torch.nn.Parameter(weight_scale, requires_grad=False),
        )
        source.weight_scale_inv.format_ue8m0 = False

        projection_slice = _BlockFp8LinearSlice(
            source=source,
            feature_indices=[0, 2],
            feature_width=feature_width,
        )

        expected_weight = torch.cat(
            [weight[:, :feature_width], weight[:, 2 * feature_width :]], dim=1
        )
        self.assertTrue(torch.equal(projection_slice.weight, expected_weight))
        self.assertTrue(
            torch.equal(
                projection_slice.weight_scale_inv,
                torch.tensor([[11.0, 33.0]]),
            )
        )

    def test_partial_projection_uses_prepared_quantized_slice(self):
        model = _make_deepseek_v4_dspark_projection_model(
            hidden_size=4, num_target_features=3
        )
        expected = torch.randn(2, 4)
        projection_slice = Mock(return_value=expected)
        model._partial_feature_indices = (1,)
        model._partial_main_proj = projection_slice
        local_hidden = torch.randn(2, 4)

        actual = model.project_target_hidden_partial(local_hidden, [1])

        projection_slice.assert_called_once_with(local_hidden)
        self.assertIs(actual, expected)

    def test_pp_spec_verify_buffers_use_token_axis(self):
        """PP verify buffers must cover bs times speculative token width."""
        max_bs = 64
        num_tokens_per_req = 6
        max_num_token = max_bs * num_tokens_per_req
        common_kwargs = dict(
            device=torch.device("cpu"),
            max_bs=max_bs,
            max_num_token=max_num_token,
            hidden_size=4,
            dtype=torch.float32,
            dp_size=1,
            pp_size=8,
            is_encoder_decoder=False,
            require_mlp_tp_gather=False,
            seq_len_fill_value=1,
            encoder_len_fill_value=0,
            num_tokens_per_req=num_tokens_per_req,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            hc_hidden_size=16,
        )
        eager_buffers = _allocate_decode_buffers(vocab_size=8, **common_kwargs)
        graph_buffers = DecodeInputBuffers.create(
            next_token_logits_buffer=torch.zeros((max_num_token, 8)),
            **common_kwargs,
        )

        self.assertEqual(
            eager_buffers.pp_proxy_tensors["hidden_states"].shape,
            (max_num_token, 16),
        )
        self.assertEqual(
            graph_buffers.pp_proxy_tensors["hidden_states"].shape,
            (max_num_token, 16),
        )

    def test_partial_projection_sum_matches_full_projection(self):
        """PP partial projections must preserve the full pre-norm FC result."""
        torch.manual_seed(0)
        hidden_size = 4
        model = DFlashDraftModel.__new__(DFlashDraftModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=hidden_size)
        model.num_context_features = 3
        model.fc = torch.nn.Linear(3 * hidden_size, hidden_size, bias=False)
        model.hidden_norm = RMSNorm(hidden_size, eps=1e-6)

        feature_hidden = [
            torch.randn(5, hidden_size, dtype=torch.float32) for _ in range(3)
        ]
        full_hidden = torch.cat(feature_hidden, dim=-1)
        full_projected = model.project_target_hidden(full_hidden)

        stage_0 = model.project_target_hidden_partial(
            torch.cat([feature_hidden[0], feature_hidden[2]], dim=-1),
            [0, 2],
        )
        stage_1 = model.project_target_hidden_partial(feature_hidden[1], [1])
        pp_projected = model.hidden_norm(stage_0 + stage_1)

        torch.testing.assert_close(pp_projected, full_projected)

    def test_deepseek_v4_partial_projection_survives_context_only_pruning(self):
        """Cross-rank contributions must retain full projection math after pruning."""
        torch.manual_seed(1)
        hidden_size = 4
        model = _make_deepseek_v4_dspark_projection_model(
            hidden_size=hidden_size, num_target_features=3
        )
        features = [torch.randn(5, hidden_size, dtype=torch.float32) for _ in range(3)]
        full_projected = model.project_target_hidden(torch.cat(features, dim=-1))

        stage_0 = model.project_target_hidden_partial(
            torch.cat([features[0], features[2]], dim=-1),
            [0, 2],
        )
        model.prune_to_ctx_projection()
        stage_1 = model.project_target_hidden_partial(features[1], [1])
        pp_projected = model.stages[0].main_norm(stage_0 + stage_1)
        write_context_hidden_kv = Mock()
        model._write_context_hidden_kv = write_context_hidden_kv
        model.write_projected_context_kv(
            projected_context=stage_0 + stage_1,
            swa_loc=torch.arange(5),
            positions=torch.arange(5),
            pool=object(),
        )

        self.assertEqual(list(model.stages[0]._modules), ["main_proj", "main_norm"])
        torch.testing.assert_close(pp_projected, full_projected)
        torch.testing.assert_close(
            write_context_hidden_kv.call_args.kwargs["main_x"],
            full_projected,
        )

    def test_deepseek_v4_capture_is_local_to_each_pp_rank(self):
        """A target feature on a non-last rank must not disappear from ctx_acc."""
        model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_last_rank=False)
        model.model = SimpleNamespace(
            start_layer=10,
            end_layer=20,
            dspark_layers_to_capture=None,
        )
        model.capture_aux_hidden_states = False

        model.set_dspark_layers_to_capture([5, 12, 18, 25])

        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.dspark_layers_to_capture, [12, 18])

    def test_non_last_pp_prefill_does_not_require_target_lm_head(self):
        """PP0-PP(N-2) must initialize without the last rank's lm_head."""
        self.assertTrue(
            _is_context_only_pp_prefill_rank(
                disaggregation_mode="prefill",
                pp_rank=0,
                pp_size=8,
            )
        )
        self.assertFalse(
            _is_context_only_pp_prefill_rank(
                disaggregation_mode="prefill",
                pp_rank=7,
                pp_size=8,
            )
        )

    def test_context_only_rank_does_not_require_draft_attention_backend(self):
        """Projection-only ranks must initialize overlap state without draft attention."""
        target_backend = object()
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._is_context_only_pp_prefill_rank = True
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=target_backend)
        )
        worker.draft_model_runner = SimpleNamespace()

        self.assertEqual(worker.spec_v2_attn_backends, (target_backend,))

    def test_lifecycle_only_rank_does_not_allocate_draft_pool(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_worker = Mock()
        worker._is_lifecycle_only_pp_prefill_rank = True

        worker.alloc_memory_pool(memory_pool_config=Mock())

        worker._draft_worker.alloc_memory_pool.assert_not_called()

    def test_lifecycle_only_rank_does_not_publish_draft_pool(self):
        worker = SimpleNamespace(is_lifecycle_only_pp_prefill_rank=True)
        spec_algorithm = SimpleNamespace(
            is_ngram=lambda: False,
            is_dspark=lambda: True,
        )

        self.assertIsNone(
            get_draft_kv_pool(
                draft_worker=worker,
                spec_algorithm=spec_algorithm,
                server_args=SimpleNamespace(enable_multi_layer_eagle=False),
            )
        )

    def test_non_last_pp_prefill_uses_minimal_draft_kv_pool(self):
        """A context-only PP rank must not reserve the full draft KV capacity."""
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_worker = Mock()
        worker._is_pd_prefill = True
        worker._draft_is_moe = True
        worker._is_context_only_pp_prefill_rank = True
        worker._is_lifecycle_only_pp_prefill_rank = False
        worker.ps = SimpleNamespace(pp_rank=0, pp_size=2)
        worker.page_size = 64
        full_config = MemoryPoolConfig(
            max_total_num_tokens=4096,
            max_running_requests=32,
        )

        worker.alloc_memory_pool(memory_pool_config=full_config)

        passed_config = worker._draft_worker.alloc_memory_pool.call_args.kwargs[
            "memory_pool_config"
        ]
        self.assertEqual(passed_config.max_total_num_tokens, 64)
        self.assertEqual(passed_config.max_running_requests, 32)
        self.assertEqual(full_config.max_total_num_tokens, 4096)

    def test_last_pp_prefill_keeps_full_draft_kv_pool(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_worker = Mock()
        worker._is_pd_prefill = True
        worker._draft_is_moe = True
        worker._is_context_only_pp_prefill_rank = False
        worker._is_lifecycle_only_pp_prefill_rank = False
        worker.ps = SimpleNamespace(pp_rank=1, pp_size=2)
        worker.page_size = 64
        full_config = MemoryPoolConfig(
            max_total_num_tokens=4096,
            max_running_requests=32,
        )

        worker.alloc_memory_pool(memory_pool_config=full_config)

        passed_config = worker._draft_worker.alloc_memory_pool.call_args.kwargs[
            "memory_pool_config"
        ]
        self.assertIs(passed_config, full_config)


if __name__ == "__main__":
    unittest.main()
