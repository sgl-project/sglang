from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dflash,
    _handle_dspark,
    _handle_eagle_family,
    _resolve_dflash_draft_attention_backend,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
    ModeConfig,
)
from sglang.srt.layers.attention import flash_attn_v100_backend
from sglang.srt.layers.attention.flash_attn_v100_backend import (
    FlashAttnV100Backend,
    _get_native_paged_attention_params,
    _grouped_decode_requested,
    _is_dflash_draft_native_shape_supported,
    _should_skip_triton_prefill,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.dflash import (
    DFlashDraftModel,
    _get_dflash_layer_attention_params,
    _resolve_dflash_rope_config,
)
from sglang.srt.models.dspark import DSparkDraftModel, VanillaMarkov
from sglang.srt.models.qwen3_5_mtp import _is_mtp_dynamically_unquantized
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    get_dflash_attention_sliding_window_size,
    parse_dflash_draft_config,
    resolve_dflash_verify_mask_policy,
    scale_kv_cell_size_per_token_for_dflash,
    synchronize_dflash_sampling_results,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.dspark_config import parse_dspark_draft_config
from sglang.srt.speculative.dspark_worker_v2 import (
    DSparkWorkerV2,
    _DSparkGraphSampler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class _ForwardMode:
    def __init__(self, *, target_verify=False, draft_extend=False):
        self._target_verify = target_verify
        self._draft_extend = draft_extend

    def is_target_verify(self):
        return self._target_verify

    def is_decode_or_idle(self):
        return False

    def is_draft_extend(self, include_v2=False):
        return self._draft_extend


def test_grouped_tilelang_decode_defaults_on(monkeypatch):
    monkeypatch.delenv("SGLANG_V100_GROUPED_DECODE", raising=False)
    monkeypatch.delenv("SGLANG_V100_LONG_DECODE_XQA", raising=False)

    assert _grouped_decode_requested() is True


@pytest.mark.parametrize(("value", "expected"), [("yes", True), ("off", False)])
def test_grouped_tilelang_decode_explicit_override(monkeypatch, value, expected):
    monkeypatch.setenv("SGLANG_V100_GROUPED_DECODE", value)

    assert _grouped_decode_requested() is expected


def test_grouped_tilelang_decode_rejects_invalid_override(monkeypatch):
    monkeypatch.setenv("SGLANG_V100_GROUPED_DECODE", "sometimes")

    with pytest.raises(ValueError, match="must be a boolean value"):
        _grouped_decode_requested()


def test_dflash_normalizes_target_tensors_to_loaded_weight_dtype():
    model = object.__new__(DFlashDraftModel)
    torch.nn.Module.__init__(model)
    model.fc = torch.nn.Linear(4, 4, bias=False, dtype=torch.float16)
    bf16_target_embedding = torch.randn(2, 4, dtype=torch.bfloat16)

    normalized = model._to_runtime_dtype(bf16_target_embedding)

    assert normalized.dtype == torch.float16
    already_normalized = torch.randn(2, 4, dtype=torch.float16)
    assert model._to_runtime_dtype(already_normalized) is already_normalized


def test_dflash_sampling_results_are_synchronized_across_tp():
    tp_group = SimpleNamespace(world_size=4, broadcast=Mock())
    correct_len = torch.tensor([0, 3], dtype=torch.int32)
    bonus = torch.tensor([42, 99], dtype=torch.int64)

    synchronized = synchronize_dflash_sampling_results(
        correct_len=correct_len,
        bonus=bonus,
        tp_group=tp_group,
    )

    assert synchronized[0] is correct_len
    assert synchronized[1] is bonus
    assert tp_group.broadcast.call_args_list == [
        ((correct_len,), {"src": 0}),
        ((bonus,), {"src": 0}),
    ]


def test_dflash_sampling_sync_is_noop_for_single_tp_rank():
    tp_group = SimpleNamespace(world_size=1, broadcast=Mock())
    correct_len = torch.tensor([2], dtype=torch.int32)
    bonus = torch.tensor([7], dtype=torch.int64)

    synchronize_dflash_sampling_results(
        correct_len=correct_len,
        bonus=bonus,
        tp_group=tp_group,
    )

    tp_group.broadcast.assert_not_called()


def test_dflash_sampling_rejects_invalid_kernel_acceptance(monkeypatch):
    from sglang.srt.speculative import dflash_utils

    def fake_target_only_kernel(**kwargs):
        kwargs["accept_token_num"].fill_(-1)
        kwargs["accept_index"].copy_(kwargs["retrive_index"].to(torch.int32))
        kwargs["predicts"].copy_(
            torch.arange(
                kwargs["predicts"].numel(),
                dtype=torch.int32,
                device=kwargs["predicts"].device,
            )
            + 10
        )

    monkeypatch.setattr(dflash_utils, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True)
    monkeypatch.setattr(
        dflash_utils,
        "tree_speculative_sampling_target_only",
        fake_target_only_kernel,
    )
    candidates = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    logits = torch.zeros((6, 8), dtype=torch.float32)
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((2, 1), dtype=torch.float32),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
    )

    with pytest.raises(
        RuntimeError,
        match="invalid accepted-token count",
    ):
        dflash_utils.compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=logits,
            sampling_info=sampling_info,
            threshold_single=1.0,
            threshold_acc=1.0,
            uniform_samples=torch.zeros((2, 3), dtype=torch.float32),
            uniform_samples_for_final_sampling=torch.zeros((2,), dtype=torch.float32),
            use_sparse_topk=False,
        )


def test_dflash_verify_applies_overlap_additive_and_scaling_penalties():
    logits = torch.tensor(
        [[4.0, -4.0, 1.0], [2.0, -2.0, -1.0]],
        dtype=torch.float32,
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), dtype=torch.float32),
        has_custom_logit_processor=False,
        penalizer_orchestrator=None,
        acc_additive_penalties=torch.tensor([[1.0, 2.0, 3.0]]),
        acc_scaling_penalties=torch.tensor([[2.0, 2.0, 1.0]]),
        vocab_mask=None,
        logit_bias=torch.tensor([[0.5, 0.0, -0.5]]),
    )

    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=2,
    )

    assert torch.equal(
        logits,
        torch.tensor(
            [[3.0, -4.0, 3.5], [2.0, 0.0, 1.5]],
            dtype=torch.float32,
        ),
    )


def test_dflash_verify_applies_live_penalizer_to_every_block_row():
    penalizer = SimpleNamespace(is_required=True, apply=Mock())
    logits = torch.zeros((6, 4), dtype=torch.float32)
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((2, 1), dtype=torch.float32),
        has_custom_logit_processor=False,
        penalizer_orchestrator=penalizer,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        vocab_mask=None,
        logit_bias=None,
    )

    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=3,
    )

    penalizer.apply.assert_called_once_with(logits, repeat=3)


def test_dflash_min_p_filters_and_renormalizes_target_distribution(monkeypatch):
    from sglang.srt.speculative import dflash_utils

    captured = {}

    def fake_target_only_kernel(**kwargs):
        captured["target_probs"] = kwargs["target_probs"].clone()
        kwargs["accept_token_num"].zero_()
        kwargs["accept_index"].copy_(kwargs["retrive_index"].to(torch.int32))
        kwargs["predicts"].zero_()

    monkeypatch.setattr(dflash_utils, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True)
    monkeypatch.setattr(
        dflash_utils,
        "tree_speculative_sampling_target_only",
        fake_target_only_kernel,
    )
    logits = torch.log(
        torch.tensor(
            [[0.60, 0.30, 0.10], [0.50, 0.40, 0.10]],
            dtype=torch.float32,
        )
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), dtype=torch.float32),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=True,
        min_ps=torch.tensor([0.75], dtype=torch.float32),
        sampling_seed=None,
    )

    dflash_utils.compute_dflash_sampling_correct_drafts_and_bonus(
        candidates=torch.tensor([[1, 2]], dtype=torch.int64),
        next_token_logits=logits,
        sampling_info=sampling_info,
        threshold_single=1.0,
        threshold_acc=1.0,
        uniform_samples=torch.zeros((1, 2), dtype=torch.float32),
        uniform_samples_for_final_sampling=torch.zeros((1,), dtype=torch.float32),
        use_sparse_topk=False,
    )

    assert torch.allclose(
        captured["target_probs"],
        torch.tensor([[[1.0, 0.0, 0.0], [5.0 / 9.0, 4.0 / 9.0, 0.0]]]),
    )


def test_dflash_sampling_uses_request_seed_and_position(monkeypatch):
    from sglang.srt.speculative import dflash_utils

    captured = {}
    seeded_coins = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    def fake_seeded_uniform_samples(**kwargs):
        assert kwargs["sampling_seed"].tolist() == [123]
        assert kwargs["sampling_positions"].tolist() == [17]
        assert kwargs["num_samples"] == 4
        return seeded_coins

    def fake_target_only_kernel(**kwargs):
        captured["coins"] = kwargs["uniform_samples"].clone()
        captured["final_coins"] = kwargs["uniform_samples_for_final_sampling"].clone()
        kwargs["accept_token_num"].zero_()
        kwargs["accept_index"].copy_(kwargs["retrive_index"].to(torch.int32))
        kwargs["predicts"].zero_()

    monkeypatch.setattr(dflash_utils, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True)
    monkeypatch.setattr(
        dflash_utils,
        "_dflash_seeded_uniform_samples",
        fake_seeded_uniform_samples,
    )
    monkeypatch.setattr(
        dflash_utils,
        "tree_speculative_sampling_target_only",
        fake_target_only_kernel,
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), dtype=torch.float32),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        sampling_seed=torch.tensor([123], dtype=torch.int64),
    )

    dflash_utils.compute_dflash_sampling_correct_drafts_and_bonus(
        candidates=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        next_token_logits=torch.zeros((3, 4), dtype=torch.float32),
        sampling_info=sampling_info,
        sampling_positions=torch.tensor([17], dtype=torch.int64),
        threshold_single=1.0,
        threshold_acc=1.0,
        use_sparse_topk=False,
    )

    assert torch.equal(captured["coins"], seeded_coins[:, :3])
    assert torch.equal(captured["final_coins"], seeded_coins[:, 3])


@pytest.mark.parametrize(
    "model_arch",
    [
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
def test_qwen35_uses_builtin_mtp_checkpoint(model_arch):
    server_args = SimpleNamespace(
        speculative_algorithm="EAGLE",
        enable_dp_attention=False,
        max_running_requests=4,
        speculative_eagle_topk=1,
        disable_overlap_schedule=True,
        enable_mixed_chunk=False,
        get_model_config=Mock(
            return_value=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[model_arch])
            )
        ),
        model_path="Qwen/model",
        revision=None,
        speculative_draft_model_path=None,
        speculative_draft_model_revision=None,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        attention_backend="flash_attn_v100",
        decode_attention_backend=None,
        prefill_attention_backend=None,
        page_size=16,
    )

    _handle_eagle_family(server_args)

    assert server_args.speculative_draft_model_path == server_args.model_path


def test_qwen35_factory_gptq_keeps_mtp_unquantized():
    quant_config = SimpleNamespace(dynamic={"-:.*mtp.*": {}})

    assert _is_mtp_dynamically_unquantized(quant_config, "mtp")
    assert _is_mtp_dynamically_unquantized(quant_config, "draft.mtp.layers.0")
    assert not _is_mtp_dynamically_unquantized(quant_config, "model.layers.0")


def test_dflash_does_not_inherit_gptq_target_quantization(monkeypatch):
    from sglang.srt.utils import hf_transformers_utils

    monkeypatch.setattr(
        hf_transformers_utils,
        "get_config",
        lambda *args, **kwargs: SimpleNamespace(quantization_config=None),
    )
    server_args = SimpleNamespace(
        enable_dp_attention=False,
        pp_size=1,
        speculative_draft_model_path="z-lab/dflash",
        trust_remote_code=False,
        speculative_draft_model_revision="main",
        json_model_override_args="{}",
        speculative_draft_model_quantization="gptq_marlin",
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_dflash_block_size=16,
        speculative_num_draft_tokens=None,
        speculative_draft_window_size=None,
        max_running_requests=4,
        disable_overlap_schedule=False,
        enable_mixed_chunk=False,
        attention_backend="flash_attn_v100",
    )

    _handle_dflash(server_args)

    assert server_args.speculative_draft_model_quantization is None
    assert server_args.speculative_num_draft_tokens == 16


@pytest.mark.parametrize(
    ("attention_backend", "explicit_block_size", "expected_block_size"),
    [
        ("tilelang_fa_v100", None, 8),
        ("flash_attn_v100", None, 8),
        ("triton", None, 16),
        ("tilelang_fa_v100", 4, 4),
        ("tilelang_fa_v100", 16, 16),
    ],
)
def test_laguna_dflash_uses_v100_tuned_default_block_size(
    monkeypatch, attention_backend, explicit_block_size, expected_block_size
):
    from sglang.srt.utils import hf_transformers_utils

    monkeypatch.setattr(
        hf_transformers_utils,
        "get_config",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["DFlashLagunaForCausalLM"],
            quantization_config={"quant_method": "compressed-tensors"},
            num_hidden_layers=6,
            dflash_config={"block_size": 16},
        ),
    )
    server_args = SimpleNamespace(
        enable_dp_attention=False,
        pp_size=1,
        speculative_draft_model_path="poolside/Laguna-S-2.1-DFlash-INT4",
        trust_remote_code=False,
        speculative_draft_model_revision="main",
        json_model_override_args="{}",
        speculative_draft_model_quantization=None,
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_dflash_block_size=explicit_block_size,
        speculative_num_draft_tokens=None,
        speculative_draft_window_size=None,
        max_running_requests=4,
        disable_overlap_schedule=False,
        enable_mixed_chunk=False,
        attention_backend=attention_backend,
    )

    _handle_dflash(server_args)

    assert server_args.speculative_num_draft_tokens == expected_block_size


@pytest.mark.parametrize("kind", ["decode", "extend"])
@pytest.mark.parametrize("backend", ["tilelang_fa_v100", "flash_attn_v100"])
def test_mtp_maps_v100_tilelang_backend_to_triton(kind, backend):
    server_args = SimpleNamespace(
        speculative_draft_attention_backend=None,
        attention_backend=backend,
        decode_attention_backend=None,
        prefill_attention_backend=None,
        speculative_attention_mode="prefill",
    )
    factory = DraftBackendFactory(
        server_args,
        draft_model_runner=object(),
        topk=1,
        speculative_num_steps=3,
    )
    sentinel = object()
    if kind == "decode":
        factory._create_triton_decode_backend = Mock(return_value=sentinel)
        result = factory.create_decode_backend()
        factory._create_triton_decode_backend.assert_called_once_with()
    else:
        factory._create_triton_prefill_backend = Mock(return_value=sentinel)
        result = factory.create_draft_extend_backend()
        factory._create_triton_prefill_backend.assert_called_once_with()
    assert result is sentinel


@pytest.mark.parametrize("backend", ["tilelang_fa_v100", "flash_attn_v100"])
def test_dflash_uses_native_draft_attention_on_v100(backend):
    assert _resolve_dflash_draft_attention_backend(backend) == backend


@pytest.mark.parametrize("enable_spec_v2", [True, False])
def test_dflash_worker_selection_tracks_spec_v2_env(enable_spec_v2):
    from sglang.srt.environ import envs

    with envs.SGLANG_ENABLE_SPEC_V2.override(enable_spec_v2):
        worker_cls = SpeculativeAlgorithm.DFLASH.create_worker(SimpleNamespace())

        assert SpeculativeAlgorithm.DFLASH.supports_spec_v2() is enable_spec_v2
        if enable_spec_v2:
            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

            assert worker_cls is DFlashWorkerV2
        else:
            # V2 drives both overlap and non-overlap now.
            assert worker_cls is DFlashWorkerV2


def test_dspark_worker_is_always_spec_v2():
    assert SpeculativeAlgorithm.DSPARK.supports_spec_v2()
    assert SpeculativeAlgorithm.DSPARK.is_dflash_family()
    assert (
        SpeculativeAlgorithm.DSPARK.get_num_tokens_per_bs_for_target_verify(8, True)
        == 7
    )
    assert (
        SpeculativeAlgorithm.DSPARK.get_num_tokens_per_bs_for_target_verify(8, False)
        == 8
    )
    assert (
        SpeculativeAlgorithm.DSPARK.create_worker(SimpleNamespace()) is DSparkWorkerV2
    )


def test_dspark_config_and_verify_width_contract(monkeypatch):
    draft_config = SimpleNamespace(
        architectures=["DSparkDraftModel"],
        num_hidden_layers=5,
        block_size=7,
        markov_rank=256,
        markov_head_type="vanilla",
        quantization_config=None,
        dflash_config={"mask_token_id": 248077},
    )
    monkeypatch.setattr(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        lambda *args, **kwargs: draft_config,
    )
    args = SimpleNamespace(
        enable_dp_attention=False,
        pp_size=1,
        speculative_draft_model_path="RadixArk/Qwen3.8-27B-DSpark",
        trust_remote_code=True,
        speculative_draft_model_revision="main",
        json_model_override_args="{}",
        speculative_draft_model_quantization="unquant",
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_dspark_block_size=7,
        speculative_num_draft_tokens=None,
        speculative_draft_window_size=None,
        max_running_requests=1,
        enable_mixed_chunk=True,
    )

    _handle_dspark(args)
    parsed = parse_dspark_draft_config(draft_config)

    assert parsed.gamma == 7
    assert parsed.markov_rank == 256
    assert parsed.mask_token_id == 248077
    assert args.speculative_num_draft_tokens == 8
    assert args.speculative_num_steps == 1
    assert args.speculative_eagle_topk == 1
    assert args.speculative_draft_model_quantization is None
    assert args.enable_mixed_chunk is False


def test_dspark_sm70_draft_cache_stays_fp16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)

    assert (
        worker._draft_kv_cache_dtype(SimpleNamespace(kv_cache_dtype="fp8_e4m3"))
        == "auto"
    )
    assert (
        scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=8192,
            target_num_layers=16,
            draft_num_layers=5,
            draft_cell_size_per_token=5120,
        )
        == 13312
    )


def test_dspark_markov_head_is_autoregressive(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.models.dspark.tensor_model_parallel_all_gather",
        lambda logits, dim: logits,
    )
    model = DSparkDraftModel.__new__(DSparkDraftModel)
    torch.nn.Module.__init__(model)
    model.gamma = 2
    model.config = SimpleNamespace(vocab_size=4)
    model.markov_head = VanillaMarkov(vocab_size=4, rank=4)
    lm_head = torch.nn.Linear(2, 4, bias=False)

    with torch.no_grad():
        lm_head.weight.zero_()
        model.markov_head.markov_w1.weight.copy_(torch.eye(4))
        model.markov_head.markov_w2.weight.zero_()
        model.markov_head.markov_w2.weight[2, 1] = 10
        model.markov_head.markov_w2.weight[3, 2] = 10

    proposed = model.sample_greedy_block(
        hidden_states=torch.zeros(1, 2, 2),
        anchor_tokens=torch.tensor([1]),
        lm_head=lm_head,
    )

    assert proposed.tolist() == [[2, 3]]


def test_dspark_graph_sampler_and_eager_fallback_match():
    class DraftModel:
        def __init__(self):
            self.calls = 0

        def sample_greedy_block(self, *, hidden_states, anchor_tokens, lm_head):
            del hidden_states, lm_head
            self.calls += 1
            offsets = torch.arange(1, 4, dtype=torch.int64)
            return anchor_tokens[:, None] + offsets[None, :]

    model = DraftModel()
    sampler = _DSparkGraphSampler(
        model=model,
        lm_head=object(),
        gamma=3,
        max_bs=2,
        device="cpu",
    )
    block_ids = torch.tensor([[10, 0, 0], [20, 0, 0]], dtype=torch.int64)
    hidden_states = torch.zeros((6, 4))
    sampler(hidden_states, block_ids.flatten())

    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker._graph_sampler = sampler
    worker.draft_model = model
    graph_tokens = torch.empty((2, 4), dtype=torch.int64)
    worker._populate_draft_tokens(
        draft_hidden=hidden_states.view(2, 3, 4),
        block_ids=block_ids,
        lm_head=object(),
        draft_tokens=graph_tokens,
        can_run_graph=True,
    )
    eager_tokens = torch.empty_like(graph_tokens)
    worker._populate_draft_tokens(
        draft_hidden=hidden_states.view(2, 3, 4),
        block_ids=block_ids,
        lm_head=object(),
        draft_tokens=eager_tokens,
        can_run_graph=False,
    )

    assert graph_tokens.tolist() == [[10, 11, 12, 13], [20, 21, 22, 23]]
    assert torch.equal(graph_tokens, eager_tokens)
    assert model.calls == 2


def test_dflash_v2_reports_target_and_draft_attention_capabilities():
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    target_backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    draft_backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=target_backend)
    )
    worker.draft_model_runner = SimpleNamespace(attn_backend=draft_backend)
    server_args = SimpleNamespace(
        enable_two_batch_overlap=False,
        disable_piecewise_cuda_graph=True,
    )

    assert worker.spec_v2_attn_backends == (target_backend, draft_backend)
    assert not decide_needs_cpu_seq_lens(server_args, worker.spec_v2_attn_backends)

    worker.draft_model_runner.attn_backend = SimpleNamespace(needs_cpu_seq_lens=True)
    assert decide_needs_cpu_seq_lens(server_args, worker.spec_v2_attn_backends)


def test_v100_one_stage_override_reaches_default_custom_allreduce_v2(monkeypatch):
    communicator = CustomAllReduceV2.__new__(CustomAllReduceV2)
    communicator.disabled = True
    communicator.override_algo = None
    # CUDA TP4 defaults: small buffers use push, medium buffers use pull.
    communicator.config = ModeConfig(
        one_shot_push_threshold=384 * 1024,
        one_shot_pull_threshold=256 * 1024,
    )
    qwen27_verify = torch.empty(64 * 5120, dtype=torch.float16)
    qwen35_verify = torch.empty(64 * 2048, dtype=torch.float16)

    monkeypatch.delenv("SGLANG_CUSTOM_ALLREDUCE_ALGO", raising=False)
    assert communicator._determine_algo(qwen27_verify) == AllReduceAlgo.TWO_SHOT_PULL

    monkeypatch.setenv("SGLANG_CUSTOM_ALLREDUCE_ALGO", "1stage")
    assert communicator._determine_algo(qwen27_verify) == AllReduceAlgo.ONE_SHOT_PULL
    assert communicator._determine_algo(qwen35_verify) == AllReduceAlgo.ONE_SHOT_PUSH


def test_custom_allreduce_v2_rejects_invalid_algorithm_override(monkeypatch):
    communicator = CustomAllReduceV2.__new__(CustomAllReduceV2)
    communicator.disabled = True
    communicator.override_algo = None
    communicator.config = ModeConfig(1024, 1024)
    monkeypatch.setenv("SGLANG_CUSTOM_ALLREDUCE_ALGO", "one-ish")

    with pytest.raises(ValueError, match="Valid values"):
        communicator._determine_algo(torch.empty(8, dtype=torch.float16))


def test_dflash_draft_skips_irrelevant_sm70_prefill_warmup(monkeypatch):
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cuda"
    runner.is_generation = True
    runner.is_draft_worker = True
    runner.spec_algorithm = SimpleNamespace(
        is_dflash=lambda: True,
        is_dflash_family=lambda: True,
        is_speculative=lambda: True,
    )
    runner._should_run_flashinfer_autotune = Mock(return_value=False)
    runner._flashinfer_autotune = Mock()
    runner._warmup_sm70_flashinfer_sampling = Mock()
    runner._warmup_prefill_kernels_extends = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))

    runner.kernel_warmup()

    runner._warmup_sm70_flashinfer_sampling.assert_called_once_with()
    runner._warmup_prefill_kernels_extends.assert_not_called()


def test_dflash_compact_length_preserves_cpu_host_mirror():
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.draft_window_size = 4096
    worker.page_size = 16
    worker.device = "cuda"
    seq_lens_cpu = torch.tensor([100, 10_005], dtype=torch.int32)

    compact = worker._compute_compact_draft_seq_lens(seq_lens_cpu)

    assert compact.device.type == "cpu"
    assert compact.tolist() == [100, 4101]


def test_dflash_mamba_tracking_uses_post_commit_sequence_lengths():
    update_mamba_state = Mock()
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.server_args = SimpleNamespace(mamba_track_interval=256)
    worker.target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(
                update_mamba_state_after_mtp_verify=update_mamba_state
            ),
            model=object(),
        )
    )
    batch = SimpleNamespace(
        # Spec-v2 deliberately has not published the new sequence lengths yet.
        seq_lens=torch.tensor([250, 300], dtype=torch.int32),
        mamba_track_indices=torch.tensor([3, 7], dtype=torch.int32),
    )

    worker._update_target_mamba_state_after_verify(
        batch=batch,
        seq_lens_pre_verify=batch.seq_lens.clone(),
        commit_lens=torch.tensor([10, 4], dtype=torch.int32),
    )

    call = update_mamba_state.call_args.kwargs
    assert call["last_correct_step_indices"].tolist() == [9, 3]
    assert call["mamba_steps_to_track"].tolist() == [5, -1]
    assert call["mamba_track_indices"] is batch.mamba_track_indices


@pytest.mark.parametrize(
    ("attn_type", "window", "expected"),
    [
        (AttentionType.DECODER, 2047, (True, 2047)),
        (AttentionType.DECODER, -1, (True, -1)),
        (AttentionType.ENCODER_ONLY, -1, (False, -1)),
    ],
)
def test_v100_native_attention_uses_per_layer_dflash_mask(attn_type, window, expected):
    layer = SimpleNamespace(
        is_cross_attention=False,
        attn_type=attn_type,
        sliding_window_size=window,
    )

    assert _get_native_paged_attention_params(layer, True) == expected


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "head_dim", "kv_dtype", "expected"),
    [
        (8, 2, 128, torch.float16, True),  # TP4
        (16, 4, 128, torch.float16, True),  # TP2
        (8, 2, 128, torch.float8_e4m3fn, True),  # TP4 E4M3
        (16, 4, 128, torch.float8_e4m3fn, False),  # TP2 E4M3 guard
        (10, 2, 128, torch.float16, True),  # Qwen3.8 DSpark GQA-5
        (10, 2, 128, torch.float8_e5m2, True),  # compact DSpark KV
        (16, 2, 128, torch.float16, True),
        (16, 4, 256, torch.float16, False),
        (0, 0, 128, torch.float16, False),
    ],
)
def test_v100_native_dflash_draft_shape_support(
    q_heads, kv_heads, head_dim, kv_dtype, expected
):
    layer = SimpleNamespace(
        tp_q_head_num=q_heads,
        tp_k_head_num=kv_heads,
        head_dim=head_dim,
    )

    assert _is_dflash_draft_native_shape_supported(layer, kv_dtype) is expected


def test_v100_e5m2_target_verify_uses_native_tilelang(monkeypatch):
    monkeypatch.delenv("SGLANG_V100_NATIVE_LINEAR_VERIFY", raising=False)
    captured = {}

    def fake_tilelang(q, k_cache, v_cache, page_table, seq_lens, *args, **kwargs):
        captured.update(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            seq_lens=seq_lens,
            kwargs=kwargs,
        )
        kwargs["out"].zero_()

    monkeypatch.setattr(
        flash_attn_v100_backend,
        "_load_paged_forward",
        lambda: fake_tilelang,
    )
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend.page_size = 16
    backend.max_context_len = 262144
    backend._uses_sm70_fp8_kv = True
    backend._uses_sm70_e4m3_kv = False
    backend._uses_sm70_e5m2_kv = True
    backend._fp8_prefill_scratch_enabled = False
    backend.model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(
            is_dflash_family=lambda: True,
            is_eagle=lambda: False,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
    k_cache = torch.zeros(4, 16, 1, 256, dtype=torch.float8_e5m2)
    v_cache = torch.zeros_like(k_cache)
    backend.token_to_kv_pool = SimpleNamespace(
        get_kv_buffer=lambda _layer_id: (k_cache, v_cache)
    )
    rows = 7
    backend.forward_metadata = SimpleNamespace(
        page_table=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        seq_lens=torch.tensor([64], dtype=torch.int32),
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([57], dtype=torch.int32),
        causal=True,
        swa_page_table=None,
        smallq_page_table=torch.zeros(rows, 4, dtype=torch.int32),
        smallq_swa_page_table=None,
        smallq_seq_lens=torch.arange(58, 65, dtype=torch.int32),
        smallq_active_num_partitions=torch.tensor([1], dtype=torch.int32),
        smallq_max_seq_len=64,
    )
    layer = SimpleNamespace(
        layer_id=0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        tp_q_head_num=6,
        tp_k_head_num=1,
        head_dim=256,
        scaling=256**-0.5,
        k_scale=None,
        v_scale=None,
        sliding_window_size=-1,
    )
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(target_verify=True),
        out_cache_loc=None,
    )

    output = backend.forward_extend(
        torch.randn(rows, 6 * 256, dtype=torch.float16),
        None,
        None,
        layer,
        forward_batch,
        save_kv_cache=False,
    )

    assert output.shape == (rows, 6 * 256)
    assert captured["k_cache"].dtype == torch.float8_e5m2
    assert captured["v_cache"].dtype == torch.float8_e5m2
    assert captured["kwargs"]["linear_verify"] is True
    assert captured["kwargs"]["max_seq_len_hint"] == 64
    assert captured["page_table"].tolist() == [[0, 1, 2, 3]]


def test_v100_e5m2_target_metadata_counts_native_partitions():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend.device = "cpu"
    backend.page_size = 16
    backend._max_pages = 5000
    backend.model_runner = SimpleNamespace(is_draft_worker=False)
    backend.req_to_token = torch.arange(80000, dtype=torch.int64).view(1, -1)
    backend.token_to_kv_pool = SimpleNamespace()

    metadata = backend._build_extend_metadata(
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([65543], dtype=torch.int32),
        extend_seq_lens=torch.tensor([7], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([65536], dtype=torch.int32),
        causal=True,
        build_smallq=True,
    )

    assert metadata.smallq_active_num_partitions.item() == 513
    assert metadata.smallq_seq_lens.tolist() == list(range(65537, 65544))


def test_v100_e5m2_prefill_also_builds_partial_chunk_triton_metadata():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._uses_sm70_fp8_kv = True
    backend._uses_sm70_e4m3_kv = False
    backend._uses_sm70_e5m2_kv = True
    backend._fp8_prefill_scratch_enabled = True
    backend.model_runner = SimpleNamespace(is_draft_worker=False)
    backend._triton = Mock()
    backend._build_extend_metadata = Mock(return_value="native-metadata")
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices="req-pool-indices",
        seq_lens=torch.tensor([32769], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([32768], dtype=torch.int32),
        extend_seq_lens=torch.tensor([1], dtype=torch.int32),
    )

    backend.init_forward_metadata(forward_batch)

    assert backend.forward_metadata == "native-metadata"
    backend._triton.init_forward_metadata.assert_called_once_with(forward_batch)


def test_v100_native_extend_builds_distinct_swa_page_table():
    class _SwaPool:
        def __init__(self):
            self.full_to_swa_index_mapping = torch.arange(256, dtype=torch.int64)
            self.full_to_swa_index_mapping[32] = 80
            self.full_to_swa_index_mapping[48] = 96

        def translate_loc_from_full_to_swa(self, indices):
            return self.full_to_swa_index_mapping[indices].to(torch.int32)

    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend.device = "cpu"
    backend.page_size = 16
    backend._max_pages = 8
    backend.req_to_token = torch.arange(256, dtype=torch.int64).reshape(1, 256)
    backend.token_to_kv_pool = _SwaPool()

    metadata = backend._build_extend_metadata(
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([32], dtype=torch.int32),
        extend_seq_lens=torch.tensor([32], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        causal=True,
    )

    assert metadata.page_table[0, :2].tolist() == [0, 1]
    assert metadata.swa_page_table is not None
    assert metadata.swa_page_table[0, :2].tolist() == [0, 1]

    backend.req_to_token[0, 0] = 32
    backend.req_to_token[0, 16] = 48
    metadata = backend._build_extend_metadata(
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([32], dtype=torch.int32),
        extend_seq_lens=torch.tensor([32], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        causal=True,
    )
    assert metadata.page_table[0, :2].tolist() == [2, 3]
    assert metadata.swa_page_table[0, :2].tolist() == [5, 6]


def test_dflash_v100_triton_verify_skips_redundant_custom_mask():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)

    assert resolve_dflash_verify_mask_policy(backend) == (
        "FlashAttnV100Backend",
        False,
    )


def test_dflash_reads_transformers_v5_rope_parameters():
    config = SimpleNamespace(
        rope_theta=None,
        rope_scaling={"rope_type": "default", "rope_theta": 10_000_000},
    )

    rope_theta, rope_scaling = _resolve_dflash_rope_config(config)

    assert rope_theta == 10_000_000
    assert rope_scaling == config.rope_scaling


@pytest.mark.parametrize(
    (
        "config",
        "expected_block_size",
        "expected_target_layers",
        "expected_mask_token_id",
        "expected_window",
    ),
    [
        (
            SimpleNamespace(
                num_hidden_layers=5,
                num_target_layers=64,
                block_size=16,
                dflash_config={
                    "mask_token_id": 248070,
                    "target_layer_ids": [1, 16, 31, 46, 61],
                },
                layer_types=["sliding_attention"] * 4 + ["full_attention"],
                sliding_window=2048,
            ),
            16,
            [1, 16, 31, 46, 61],
            248070,
            2047,
        ),
        (
            SimpleNamespace(
                num_hidden_layers=6,
                num_target_layers=48,
                dflash_config={
                    "block_size": 16,
                    "mask_token_id": 248077,
                    "target_layer_ids": [1, 7, 14, 20, 26, 32, 39, 45],
                },
                layer_types=["sliding_attention"] * 5 + ["full_attention"],
                sliding_window=4096,
            ),
            16,
            [1, 7, 14, 20, 26, 32, 39, 45],
            248077,
            4095,
        ),
        (
            SimpleNamespace(
                num_hidden_layers=6,
                num_target_layers=40,
                dflash_config={
                    "block_size": 16,
                    "mask_token_id": 248077,
                    "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
                },
                layer_types=["sliding_attention"] * 5 + ["full_attention"],
                sliding_window=4096,
            ),
            16,
            [1, 6, 11, 16, 22, 27, 32, 37],
            248077,
            4095,
        ),
    ],
)
def test_dflash_checkpoint_config_layouts(
    config,
    expected_block_size,
    expected_target_layers,
    expected_mask_token_id,
    expected_window,
):
    parsed = parse_dflash_draft_config(draft_hf_config=config)

    assert parsed.resolve_block_size() == expected_block_size
    assert (
        parsed.resolve_target_layer_ids(target_num_layers=config.num_target_layers)
        == expected_target_layers
    )
    assert parsed.mask_token_id == expected_mask_token_id
    assert get_dflash_attention_sliding_window_size(config) == expected_window
    assert _get_dflash_layer_attention_params(config, 0) == (
        expected_window,
        AttentionType.DECODER,
    )
    assert _get_dflash_layer_attention_params(config, config.num_hidden_layers - 1) == (
        -1,
        AttentionType.ENCODER_ONLY,
    )


def test_dflash_interleaved_sliding_window_layers():
    config = SimpleNamespace(
        num_hidden_layers=3,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=2048,
    )

    assert get_dflash_attention_sliding_window_size(config) == 2047
    assert _get_dflash_layer_attention_params(config, 0) == (
        2047,
        AttentionType.DECODER,
    )
    assert _get_dflash_layer_attention_params(config, 2) == (
        -1,
        AttentionType.ENCODER_ONLY,
    )


@pytest.mark.parametrize(
    ("is_speculative", "expected_skip_prefill"),
    [(False, True), (True, False)],
)
def test_v100_triton_allocates_prefill_metadata_for_speculation(
    is_speculative, expected_skip_prefill
):
    model_runner = SimpleNamespace(
        kv_cache_dtype=torch.float16,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: is_speculative)
    )

    assert _should_skip_triton_prefill(model_runner) is expected_skip_prefill


@pytest.mark.parametrize("mode", ["target_verify", "draft_extend"])
def test_v100_speculative_extend_delegates_to_triton(mode):
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._uses_sm70_fp8_kv = False
    backend._uses_sm70_e4m3_kv = False
    backend._uses_sm70_e5m2_kv = False
    backend._fp8_prefill_scratch_enabled = False
    backend.model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: False,
            is_dflash_family=lambda: False,
            is_eagle=lambda: False,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
    backend._triton = Mock()
    backend._triton.forward_extend.return_value = "triton-output"
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(
            target_verify=mode == "target_verify",
            draft_extend=mode == "draft_extend",
        )
    )

    output = backend.forward_extend(
        q="q",
        k="k",
        v="v",
        layer="layer",
        forward_batch=forward_batch,
        save_kv_cache=False,
    )

    assert output == "triton-output"
    backend._triton.forward_extend.assert_called_once_with(
        "q",
        "k",
        "v",
        "layer",
        forward_batch,
        save_kv_cache=False,
    )


def test_v100_mtp_linear_verify_builds_native_causal_metadata(monkeypatch):
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._uses_sm70_fp8_kv = False
    backend._uses_sm70_e4m3_kv = False
    backend._uses_sm70_e5m2_kv = False
    backend.model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: False,
            is_dflash_family=lambda: False,
            is_eagle=lambda: True,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
    backend._triton = Mock()
    backend._build_extend_metadata = Mock(return_value="native-metadata")
    prefix_lens = torch.tensor([17, 33], dtype=torch.int32)
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(target_verify=True),
        req_pool_indices="req-pool-indices",
        seq_lens=prefix_lens,
        spec_info=SimpleNamespace(draft_token_num=8),
    )

    backend.init_forward_metadata(forward_batch)

    args = backend._build_extend_metadata.call_args.args
    assert args[0] == "req-pool-indices"
    assert args[1].tolist() == [25, 41]
    assert args[2].tolist() == [8, 8]
    assert args[3] is prefix_lens
    assert backend._build_extend_metadata.call_args.kwargs == {
        "causal": True,
        "build_smallq": True,
    }
    assert backend.forward_metadata == "native-metadata"
    backend._triton.init_forward_metadata.assert_not_called()


def test_v100_dflash_verify_builds_native_causal_metadata(monkeypatch):
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._uses_sm70_fp8_kv = False
    backend._uses_sm70_e4m3_kv = False
    backend._uses_sm70_e5m2_kv = False
    backend.model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: True,
            is_dflash_family=lambda: True,
            is_eagle=lambda: False,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
    backend._triton = Mock()
    backend._build_extend_metadata = Mock(return_value="native-metadata")
    prefix_lens = torch.tensor([9000], dtype=torch.int32)
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(target_verify=True),
        req_pool_indices="req-pool-indices",
        seq_lens=prefix_lens,
        spec_info=SimpleNamespace(draft_token_num=16),
    )

    backend.init_forward_metadata(forward_batch)

    args = backend._build_extend_metadata.call_args.args
    assert args[0] == "req-pool-indices"
    assert args[1].tolist() == [9016]
    assert args[2].tolist() == [16]
    assert args[3] is prefix_lens
    assert backend._build_extend_metadata.call_args.kwargs == {
        "causal": True,
        "build_smallq": True,
    }
    assert backend.forward_metadata == "native-metadata"
    backend._triton.init_forward_metadata.assert_not_called()


def test_v100_dflash_verify_uses_triton_when_native_verify_is_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_V100_NATIVE_LINEAR_VERIFY", "0")
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._uses_sm70_fp8_kv = False
    backend._uses_sm70_e4m3_kv = False
    backend._uses_sm70_e5m2_kv = False
    backend.model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: True,
            is_dflash_family=lambda: True,
            is_eagle=lambda: False,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
    backend._triton = Mock()
    forward_batch = SimpleNamespace(forward_mode=_ForwardMode(target_verify=True))

    backend.init_forward_metadata(forward_batch)

    backend._triton.init_forward_metadata.assert_called_once_with(forward_batch)
    assert backend.forward_metadata is None


def test_v100_tilelang_uses_supported_full_attention_signature(monkeypatch):
    captured = {}

    def fake_tilelang_paged(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        query_start_loc,
        prefix_kv_lens,
        *,
        out,
        block_size,
        softmax_scale,
        causal,
        num_kv_heads,
        **kwargs,
    ):
        captured["shapes"] = (q.shape, k_cache.shape, v_cache.shape)
        captured["options"] = (
            block_size,
            softmax_scale,
            causal,
            num_kv_heads,
        )
        out.zero_()

    monkeypatch.setattr(
        flash_attn_v100_backend,
        "_load_paged_forward",
        lambda: fake_tilelang_paged,
    )
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._uses_sm70_fp8_kv = False
    backend._fp8_prefill_scratch_enabled = False
    backend.page_size = 16
    backend.model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: False,
            is_eagle=lambda: False,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
    k_cache = torch.randn(32, 1, 2)
    v_cache = torch.randn_like(k_cache)
    backend.token_to_kv_pool = SimpleNamespace(
        get_kv_buffer=lambda layer_id: (k_cache, v_cache)
    )
    backend.forward_metadata = SimpleNamespace(
        page_table=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([0], dtype=torch.int32),
        causal=True,
    )
    layer = SimpleNamespace(
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=-1,
        tp_q_head_num=2,
        tp_k_head_num=1,
        head_dim=2,
        scaling=0.5,
        k_scale=None,
        v_scale=None,
        layer_id=0,
    )
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(),
        out_cache_loc=None,
    )

    output = backend.forward_extend(
        q=torch.randn(2, 4),
        k=None,
        v=None,
        layer=layer,
        forward_batch=forward_batch,
        save_kv_cache=False,
    )

    assert output.shape == (2, 4)
    assert captured == {
        "shapes": (
            torch.Size([2, 2, 2]),
            torch.Size([2, 16, 1, 2]),
            torch.Size([2, 16, 1, 2]),
        ),
        "options": (16, 0.5, True, 1),
    }


def test_v100_native_extend_selects_swa_page_table(monkeypatch):
    captured = {}

    def fake_tilelang_paged(q, k_cache, v_cache, block_table, seq_lens,
                            query_start_loc, prefix_kv_lens, *, out, **kwargs):
        captured["block_table"] = block_table.clone()
        captured["sliding_window_size"] = kwargs["sliding_window_size"]
        out.zero_()

    monkeypatch.setattr(
        flash_attn_v100_backend,
        "_load_paged_forward",
        lambda: fake_tilelang_paged,
    )
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._uses_sm70_fp8_kv = False
    backend._fp8_prefill_scratch_enabled = False
    backend.page_size = 16
    backend.model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: False,
            is_eagle=lambda: False,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
    k_cache = torch.randn(32, 1, 2)
    v_cache = torch.randn_like(k_cache)
    backend.token_to_kv_pool = SimpleNamespace(
        get_kv_buffer=lambda layer_id: (k_cache, v_cache),
    )
    backend.forward_metadata = SimpleNamespace(
        page_table=torch.tensor([[2, 3]], dtype=torch.int32),
        swa_page_table=torch.tensor([[5, 6]], dtype=torch.int32),
        seq_lens=torch.tensor([32], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([0], dtype=torch.int32),
        causal=True,
    )
    layer = SimpleNamespace(
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=511,
        tp_q_head_num=2,
        tp_k_head_num=1,
        head_dim=2,
        scaling=0.5,
        k_scale=None,
        v_scale=None,
        layer_id=0,
    )
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(),
        out_cache_loc=None,
    )

    output = backend.forward_extend(
        q=torch.randn(2, 4),
        k=None,
        v=None,
        layer=layer,
        forward_batch=forward_batch,
        save_kv_cache=False,
    )

    assert output.shape == (2, 4)
    assert captured["block_table"].tolist() == [[5, 6]]
    assert captured["sliding_window_size"] == 511


def test_v100_spec_v2_metadata_delegates_to_triton():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._uses_sm70_fp8_kv = False
    backend._triton = Mock()
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)

    backend.init_forward_metadata(forward_batch)

    backend._triton.init_forward_metadata.assert_called_once_with(forward_batch)
    assert backend.forward_metadata is None


def test_v100_verify_cuda_graph_buffers_delegate_to_triton():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._triton = Mock()
    backend._triton.get_verify_buffers_to_fill_after_draft.return_value = [
        "mask",
        None,
    ]

    assert backend.get_verify_buffers_to_fill_after_draft() == ["mask", None]
    backend.update_verify_buffers_to_fill_after_draft("spec-info", 4)

    backend._triton.get_verify_buffers_to_fill_after_draft.assert_called_once_with()
    backend._triton.update_verify_buffers_to_fill_after_draft.assert_called_once_with(
        "spec-info", 4
    )
