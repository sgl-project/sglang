"""Execute worker startup ordering and draft sampler warmup on CPU.

Checkpoint I/O, process-group setup, memory allocation and actual NPU graph
capture are boundaries here. The worker constructor, planner, injector,
proposer, observers, epilogue allocation and sampler are production code.
This is not a substitute for an NPU model-serving smoke test.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import PretrainedConfig, Qwen3Config

from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    scatter_compact_to_strided,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu import dspark_quarot
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.models.dspark import DSparkConfidenceHead, VanillaMarkov
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dspark_components import (
    dspark_planner,
    dspark_verify,
    dspark_worker_v2,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cpu_scatter_into(*, compact, verify_lens, out, stride, fill_value):
    # The production in-place entry point launches Triton unconditionally;
    # replace only that device boundary with its production torch reference.
    out.copy_(
        scatter_compact_to_strided(
            compact=compact,
            layout=SimpleNamespace(
                verify_lens=verify_lens, graph_num_tokens=compact.shape[0]
            ),
            fill_value=fill_value,
            verify_num_draft_tokens=stride,
        )
    )
    return out


@pytest.mark.parametrize("attn_dp_size", [1, 2])
@pytest.mark.parametrize("mode", ["static", "compact"])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("quarot", [False, True])
def test_dense_worker_constructor_to_draft_sampler_warmup(
    mode, graph, profile, quarot, attn_dp_size
):
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    target_hf = PretrainedConfig(architectures=["GlmMoeDsaForCausalLM"])
    target_cfg = SimpleNamespace(
        hf_config=target_hf,
        hf_text_config=target_hf,
        vocab_size=32,
        linear_attn_registry_result=None,
    )
    target_runner = SimpleNamespace(
        model_config=target_cfg,
        model=object(),
        capture_tail_hooks=[],
        spec_aux_config=SimpleNamespace(draft_model_config=None),
        spec_algorithm=SpeculativeAlgorithm.DSPARK,
    )
    target = SimpleNamespace(model_runner=target_runner, device="cpu", random_seed=19)
    draft_hf = Qwen3Config(
        hidden_size=8,
        num_hidden_layers=1,
        block_size=3,
        mask_token_id=31,
        target_layer_ids=[0],
        markov_rank=4,
        markov_head_type="vanilla",
    )
    head = torch.nn.Linear(8, 32, bias=False)
    head.org_vocab_size = 32
    confidence = (
        DSparkConfidenceHead(hidden_size=8, markov_rank=4)
        if mode == "compact"
        else None
    )
    markov = VanillaMarkov(vocab_size=32, markov_rank=4)
    draft = SimpleNamespace(
        sample_from_anchor=True,
        uses_own_vocab_modules=True,
        supports_pre_gather_target_hidden_projection=False,
        markov_head=markov,
        confidence_head=confidence,
        lm_head=head,
        compute_base_logits=lambda hidden: (head(hidden), hidden),
        compute_confidence=lambda **kwargs: confidence.predict(
            kwargs["confidence_features"],
            anchor_tokens=kwargs["anchor_tokens"],
            draft_tokens=kwargs["draft_tokens"],
            markov_head=markov,
        ),
    )
    draft_runner = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=draft_hf),
        device=torch.device("cpu"),
        model=draft,
        capture_tail_hooks=[],
    )
    draft_worker = SimpleNamespace(
        alloc_memory_pool=Mock(),
        init_attention_backends=Mock(),
        init_cuda_graphs=Mock(),
    )
    bundle = SimpleNamespace(
        draft_worker=draft_worker,
        draft_model_runner=draft_runner,
        draft_model=draft,
        resolved_attention_backend="ascend",
    )
    plan = (
        dspark_quarot.GlmDSparkQuaRotConfig("unused", 8, "unused") if quarot else None
    )

    def build(**kwargs):
        # Real eligibility and loading arithmetic have separate loader tests.
        # Check that worker startup keeps the plan scoped to draft building.
        assert dspark_quarot.get_glm_dspark_quarot_config() is plan
        draft._glm_dspark_quarot_config = plan
        return bundle

    config = CudaGraphConfig(
        decode=PhaseConfig(
            backend=Backend.FULL if graph else Backend.DISABLED, bs=[1, 2, 4]
        ),
        prefill=PhaseConfig(backend=Backend.DISABLED),
    )
    with (
        get_context().override_server_args(
            device="cpu",
            speculative_algorithm="DSpark",
            page_size=128,
            speculative_num_draft_tokens=4,
            max_running_requests=4,
            tp_size=attn_dp_size,
            attn_dp_size=attn_dp_size,
            enable_dp_lm_head=attn_dp_size > 1,
            cuda_graph_config=config,
        ) as args,
        get_parallel().override(
            tp_group=SimpleNamespace(world_size=attn_dp_size, rank_in_group=0),
            attn_tp_group=group,
            pp_group=SimpleNamespace(is_last_rank=True),
            tp_rank=0,
            attn_tp_rank=0,
            attn_dp_enabled=attn_dp_size > 1,
        ),
        envs.SGLANG_RAGGED_VERIFY_MODE.override(mode),
        envs.SGLANG_NPU_GLM_DSPARK_APPLY_QUAROT_TO_DRAFT.override(quarot),
        envs.SGLANG_DSPARK_ENABLE_SPS_RECORD.override(profile),
        envs.SGLANG_SIMULATE_ACC_LEN.override(1.0 if profile else 0.0),
        patch.object(dspark_worker_v2, "_is_npu", True),
        patch.object(dspark_planner, "_is_npu", True),
        patch.object(
            dspark_worker_v2, "draft_uses_compressed_kv_attention", return_value=False
        ),
        patch.object(dspark_worker_v2, "draft_pp_context", side_effect=nullcontext),
        patch.object(dspark_worker_v2, "get_pp_stage_load_group", return_value=group),
        patch.object(
            dspark_worker_v2, "draft_tp_context", side_effect=lambda *_: nullcontext()
        ),
        patch.object(dspark_worker_v2, "get_dp_tp_group", return_value=group),
        patch.object(dspark_worker_v2, "build_draft_tp_worker", side_effect=build),
        patch.object(
            dspark_quarot, "build_glm_dspark_quarot_config", return_value=plan
        ),
        patch.object(SpecTpSync, "available_memory_gb", return_value=8.0),
    ):
        worker = dspark_worker_v2.DSparkWorkerV2(args, 0, 0, target)
        assert dspark_quarot.get_glm_dspark_quarot_config() is None
        assert worker.gamma == 3 and worker.verify_num_draft_tokens == 4
        assert not hasattr(target_runner, "attn_backend")
        assert not worker._verify_planner._npu_glm_compact
        assert (worker._verify_epilogue is not None) == (graph and mode == "compact")
        assert (worker._simulated_correct_len is not None) == profile
        # Match the scheduler: memory pools precede backend initialization.
        worker.alloc_memory_pool()
        draft_worker.alloc_memory_pool.assert_called_once()
        draft_runner.token_to_kv_pool = SimpleNamespace()
        target_runner.attn_backend = SimpleNamespace(_dspark_dsa_ragged_verify=True)
        draft_runner.attn_backend = SimpleNamespace(_dspark_dsa_ragged_verify=False)
        worker.init_attention_backends()
        assert worker._verify_planner._npu_glm_compact
        worker.init_cuda_graphs()
        draft_worker.init_cuda_graphs.assert_called_once_with(
            capture_decode_cuda_graph=graph
        )
        if graph:
            assert worker._draft_sampler is not None
            # Real folded proposal warmup, without CUDA aliases or a model
            # backbone. This allocates the step logits before graph capture.
            hidden = torch.randn(6, 8)
            ids = torch.tensor([1, 31, 31, 2, 31, 31])
            with patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                side_effect=AssertionError("CUDA-only API"),
            ):
                worker._draft_sampler(hidden, ids)
            assert worker._draft_sampler.out[:6].shape == (6,)
            if mode == "compact":
                assert worker._draft_sampler.confidence_out[:2].shape == (2, 3)
                assert torch.isfinite(worker._draft_sampler.confidence_out[:2]).all()
                # Exercise real compact scatter/accept/finalize with two
                # different windows after the draft sampler has filled the
                # epilogue's shared draft-token buffer.
                candidates = torch.cat(
                    [ids.view(2, 3)[:, :1], worker._draft_sampler.out[:6].view(2, 3)],
                    dim=1,
                )
                packed = torch.cat([candidates[0, :1], candidates[1, :3]])
                logits = torch.full((4, 32), -10.0)
                predicted = torch.tensor([5, candidates[1, 1], candidates[1, 2], 6])
                logits.scatter_(1, predicted[:, None], 10.0)
                tail = worker._verify_epilogue
                tail.begin_step(torch.tensor([1, 3]), armed=False)
                prefix = torch.tensor([15, 31])
                with patch.object(
                    dspark_verify,
                    "scatter_compact_to_strided_into",
                    side_effect=_cpu_scatter_into,
                ):
                    tail(
                        compact_logits=logits,
                        compact_hidden=torch.randn(4, 8),
                        input_ids=packed,
                        seq_lens=prefix,
                        req_pool_indices=torch.tensor([1, 2]),
                        bs=2,
                    )
                expected = [1, 1] if profile else [1, 3]
                assert tail.read_accept(2).commit_lens.tolist() == expected
                assert (
                    tail.read_accept(2).new_seq_lens.tolist()
                    == (prefix + torch.tensor(expected)).tolist()
                )
        else:
            assert worker._draft_sampler is None
