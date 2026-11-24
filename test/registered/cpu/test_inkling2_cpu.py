"""CPU unit tests for Inkling2 runtime overrides, sconv backend, vocab_gather,
Inkling2MTP ops, and DSpark speculative decoding components."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="extra-test-cpu-intel")
register_cpu_ci(est_time=15, suite="extra-test-cpu-arm64")


# ---------------------------------------------------------------------------
# 1. Runtime overrides & CPU hooks
# ---------------------------------------------------------------------------


def test_inkling2_loop_cpu_overrides_and_hooks():
    from sglang.srt.arg_groups.choices import DRAFT_ATTENTION_BACKEND_CHOICES
    from sglang.srt.arg_groups.cuda_graph_hook import (
        apply_inkling_prefill_cuda_graph_default,
    )
    from sglang.srt.arg_groups.mamba_hook import (
        _platform_supports_mamba_cache_extra_buffer,
    )
    from sglang.srt.arg_groups.model_overrides.inkling import _inkling_overrides
    from sglang.srt.mem_cache.allocation import get_last_loc, get_last_loc_torch
    from sglang.srt.runtime_context import get_platform, override_platform
    from sglang.srt.speculative.draft_worker_common import (
        resolve_draft_worker_attention_backend,
    )

    assert "intel_amx" in DRAFT_ATTENTION_BACKEND_CHOICES
    assert "torch_native" in DRAFT_ATTENTION_BACKEND_CHOICES

    text = SimpleNamespace(hidden_size=4608, intermediate_size=2304)
    hf = SimpleNamespace(
        architectures=["Inkling2LoopForConditionalGeneration"],
        get_text_config=lambda: text,
    )
    args = SimpleNamespace(
        device="cpu",
        tool_call_parser=None,
        reasoning_parser=None,
        tp_size=1,
        moe_dp_size=1,
        ep_size=1,
        attn_dp_size=1,
        moe_a2a_backend="none",
        encoder_only=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        swa_full_tokens_ratio=None,
        mamba_full_memory_ratio=None,
        mamba_radix_cache_strategy="auto",
        disable_radix_cache=False,
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        _model_config=SimpleNamespace(quantization="inkling2"),
    )

    with (
        override_platform(
            is_cuda=False,
            is_cpu=True,
            has_amx=True,
            is_sm90=False,
            is_sm100=False,
            is_sm120=False,
        ),
        patch("sglang.srt.utils.device.is_host_cpu_arm64", return_value=False),
    ):
        assert get_platform().is_cpu is True
        out = _inkling_overrides(args, hf)
        assert out["attention_backend"] == "intel_amx"
        assert out["tool_call_parser"] == "inkling2"
        assert out["reasoning_parser"] == "inkling"
        assert out["swa_full_tokens_ratio"] == 0.1
        assert out["mamba_full_memory_ratio"] == 0.1
        assert out["mamba_radix_cache_strategy"] == "extra_buffer"
        assert out["uses_mamba_radix_cache"] is True
        assert "moe_a2a_backend" not in out
        assert _platform_supports_mamba_cache_extra_buffer() is True

        sa = SimpleNamespace(
            device="cpu",
            disable_cuda_graph=False,
            cuda_graph_backend_prefill=None,
            disable_prefill_cuda_graph=False,
            model_path="dummy",
            _model_config=SimpleNamespace(hf_config=hf),
            _declared={},
            _resolution_records=[],
        )
        apply_inkling_prefill_cuda_graph_default(sa)
        assert "cuda_graph_backend_prefill" not in sa._declared

        with (
            patch(
                "sglang.srt.speculative.draft_worker_common.get_spec",
                return_value=SimpleNamespace(speculative_draft_attention_backend=None),
            ),
            patch(
                "sglang.srt.speculative.draft_worker_common.attention_backends",
                return_value=("intel_amx", "intel_amx"),
            ),
        ):
            assert (
                resolve_draft_worker_attention_backend(algo_label="DSPARK")
                == "intel_amx"
            )

        with (
            patch(
                "sglang.srt.speculative.draft_worker_common.get_spec",
                return_value=SimpleNamespace(speculative_draft_attention_backend="fa4"),
            ),
            patch(
                "sglang.srt.speculative.draft_worker_common.attention_backends",
                return_value=("fa4", "fa4"),
            ),
        ):
            # CPU falls back to intel_amx when draft backend is unset.
            with patch(
                "sglang.srt.speculative.draft_worker_common.get_spec",
                return_value=SimpleNamespace(speculative_draft_attention_backend=None),
            ):
                with patch(
                    "sglang.srt.speculative.draft_worker_common.attention_backends",
                    return_value=(None, None),
                ):
                    assert resolve_draft_worker_attention_backend() == "intel_amx"

    # get_last_loc on CPU takes the torch fallback without Triton.
    req_to_token = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
    prefix_lens = torch.tensor([2, 0], dtype=torch.int64)
    with (
        patch("sglang.srt.mem_cache.allocation._is_cpu", True),
        patch(
            "sglang.srt.mem_cache.allocation.attention_backends",
            return_value=("intel_amx", "intel_amx"),
        ),
    ):
        locs = get_last_loc(req_to_token, req_pool_indices, prefix_lens)
        expected = get_last_loc_torch(req_to_token, req_pool_indices, prefix_lens)
        assert torch.equal(locs, expected)
        assert locs.tolist() == [11, -1]


# ---------------------------------------------------------------------------
# 2. InklingSConvAttnBackend forward metadata + speculative rollback on CPU
# ---------------------------------------------------------------------------


def _make_cpu_sconv_backend(*, slot_count=8, draft_token_num=4):
    from sglang.srt.layers.attention.linear.inkling_sconv_backend import (
        InklingShortConvAttnBackend,
    )
    from sglang.srt.mem_cache.kv.alloc.band import Band
    from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
    from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec, StateView
    from sglang.srt.mem_cache.kv.requests import RequestTable
    from sglang.test.unified_allocator_fixtures import arena_of

    mamba = MambaSubPoolSpec(
        name="mamba",
        layer_num=2,
        grow_direction="up",
        conv_state_shapes=((3, 16), (3, 16)),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(0,),
        temporal_dtype=torch.float32,
    )
    full = MHASubPoolSpec(
        name="full",
        layer_num=1,
        head_num=1,
        head_dim=8,
        store_dtype=torch.bfloat16,
        grow_direction="down",
    )
    shared = arena_of(
        total_bytes=64 * mamba.entry_bytes(),
        sub_pool_specs=[mamba, full],
        device="cpu",
        enable_memory_saver=False,
    )
    shared.raw.zero_()
    pool = StateView(
        arena=shared,
        spec=shared.spec("mamba"),
        mamba_layer_ids=[0, 1],
        intermediate_ssm=torch.empty((0,), dtype=torch.float32),
        intermediate_conv_window=[
            torch.zeros((2, 4, 4, 3, 16), dtype=torch.bfloat16),
            torch.zeros((2, 4, 4, 3, 16), dtype=torch.bfloat16),
        ],
    )
    allocator = Band(
        arena=shared,
        sub_pool_name="mamba",
        device="cpu",
        is_id_owner=True,
        lazy_compaction=True,
    )
    slots = allocator.alloc(slot_count)
    assert slots is not None

    req_pool = RequestTable.__new__(RequestTable)
    req_pool.size = 4
    req_pool.state_sub_pool = ("mamba", None)
    req_pool.req_index_to_mamba_index_mapping = slots[:4].to(torch.int32)

    router = SimpleNamespace(
        part_view=lambda part: pool,
        part_indices=lambda part: [0, 1],
    )
    token_to_kv_pool = SimpleNamespace(layer_router=router)
    runner_ids = SimpleNamespace(
        state_ids=lambda sub_pool: allocator.ids.slot_space(
            (allocator.sub_pool_name, None)
        ),
        request_ids=lambda: None,
    )
    model_runner = SimpleNamespace(
        server_args=SimpleNamespace(enable_mis=False),
        device="cpu",
        is_draft_worker=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool=token_to_kv_pool,
        runner_ids=runner_ids,
    )

    cg_cfg = SimpleNamespace(
        decode=SimpleNamespace(bs=[1, 2, 4], max_bs=4),
        prefill=SimpleNamespace(bs=[8]),
    )
    mod = "sglang.srt.layers.attention.linear.inkling_sconv_backend"
    base_mod = "sglang.srt.layers.attention.hybrid_linear_attn_backend"
    with (
        patch(
            f"{mod}.get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(cuda_graph_config=cg_cfg)
            ),
        ),
        patch(
            f"{mod}.get_spec",
            return_value=SimpleNamespace(speculative_num_draft_tokens=draft_token_num),
        ),
        patch(f"{mod}.mamba_cache_chunk_size", return_value=4),
        patch(
            f"{base_mod}.get_spec",
            return_value=SimpleNamespace(
                speculative_eagle_topk=None,
                speculative_num_draft_tokens=draft_token_num,
            ),
        ),
        patch(
            f"{base_mod}.get_memory",
            return_value=SimpleNamespace(enable_unified_memory=True),
        ),
        patch(
            f"{base_mod}.get_exec",
            return_value=SimpleNamespace(
                mamba=SimpleNamespace(
                    enable_linear_replayssm=False, enable_linear_replayssm_spec=False
                )
            ),
        ),
    ):
        backend = InklingShortConvAttnBackend(model_runner)
        backend.init_cpu_graph_state(4, 16)

    return backend, pool, slots


def test_inkling_sconv_backend_cpu_modes_and_rollback():
    from sglang.srt.layers.attention.linear.inkling_sconv_backend import (
        InklingShortConvHybridAttnBackend,
    )
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

    backend, pool, slots = _make_cpu_sconv_backend(slot_count=8, draft_token_num=4)

    # 1. EXTEND with tracking metadata
    extend_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        extend_num_tokens=7,
        extend_seq_lens=torch.tensor([3, 4], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([3, 6], dtype=torch.int32),
        spec_info=None,
        mamba_track_mask=torch.tensor([False, True], dtype=torch.bool),
        mamba_track_indices=torch.tensor([4, 5], dtype=torch.int64),
        mamba_track_seqlens=torch.tensor([3, 6], dtype=torch.int64),
    )
    backend.init_forward_metadata(extend_batch)
    md = backend.conv_state_metadata(0, extend_batch)
    assert md.query_start_loc.tolist() == [0, 3, 7]
    assert md.has_initial_state.tolist() == [False, True]
    assert md.precomputed["cu"].tolist() == [0, 3, 7]
    assert md.precomputed["si"].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert md.track_conv_indices is not None
    assert md.track_conv_indices.shape == (2, 3)

    # 2. DECODE
    decode_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=torch.tensor([4, 7], dtype=torch.int32),
        spec_info=None,
        mamba_track_mask=None,
        mamba_track_indices=None,
        mamba_track_seqlens=None,
    )
    backend.init_forward_metadata(decode_batch)
    md = backend.conv_state_metadata(0, decode_batch)
    assert md.query_start_loc.tolist() == [0, 1, 2]
    assert md.has_initial_state.tolist() == [True, True]
    assert md.precomputed["si"].tolist() == [0, 1]

    # 3. CPU graph capture hook for DECODE
    backend.init_forward_metadata_capture_cpu_graph(
        bs=2,
        num_tokens=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=torch.tensor([4, 7], dtype=torch.int32),
        out_cache_loc=torch.tensor([0, 1], dtype=torch.int64),
        encoder_lens=None,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
    )
    assert backend.sconv_metadata.query_start_loc.tolist() == [0, 1, 2]

    # 4. DRAFT_EXTEND_V2 with boundary-KV front tokens
    draft_extend_batch = SimpleNamespace(
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        extend_num_tokens=6,
        extend_seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        extend_prefix_lens=None,
        seq_lens=torch.tensor([5, 8], dtype=torch.int32),
        spec_info=SimpleNamespace(
            __class__=EagleDraftExtendInput,
            num_front_tokens=1,
        ),
        mamba_track_mask=None,
        mamba_track_indices=None,
        mamba_track_seqlens=None,
    )
    # Use a real EagleDraftExtendInput-like isinstance check via monkeypatch
    draft_extend_batch.spec_info = EagleDraftExtendInput.__new__(EagleDraftExtendInput)
    draft_extend_batch.spec_info.num_front_tokens = 1
    backend.init_forward_metadata(draft_extend_batch)
    md = backend.conv_state_metadata(0, draft_extend_batch)
    assert md.has_initial_state.tolist() == [False, False]
    assert md.track_conv_indices is None

    # 5. TARGET_VERIFY + speculative conv state rollback/scatter
    verify_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=torch.tensor([8, 11], dtype=torch.int32),
        spec_info=SimpleNamespace(draft_token_num=4),
        mamba_track_mask=None,
        mamba_track_indices=torch.tensor([4, 5], dtype=torch.int64),
        mamba_track_seqlens=None,
    )
    backend.init_forward_metadata(verify_batch)
    md = backend.conv_state_metadata(0, verify_batch)
    assert md.query_start_loc.tolist() == [0, 4, 8]
    assert md.has_initial_state.tolist() == [True, True]

    # Populate intermediate conv windows for stream 0 & 1 and verify rollback copies
    # the accepted step into both working slots and tracked checkpoint slots.
    params = pool.get_speculative_mamba2_params_all_layers()
    for s_idx, win in enumerate(params.intermediate_conv_window):
        win.zero_()
        for step in range(4):
            win[:, :2, step] = float((s_idx + 1) * 10 + step + 1)

    full_attn_mock = SimpleNamespace(
        token_to_kv_pool=backend.token_to_kv_pool,
        req_to_token_pool=backend.req_to_token_pool,
        runner_ids=None,
        needs_cpu_seq_lens=False,
        use_captured_forward_metadata_for_breakable_cuda_graph=False,
        init_forward_metadata=lambda fb: None,
        supports_draft_extend_metadata_staging=False,
        supports_ragged_verify_graph=False,
        supports_full_cuda_graph_chunked_prefix=False,
        draft_extend_metadata_captured_in_graph=lambda: False,
    )
    with patch(
        "sglang.srt.layers.attention.hybrid_linear_attn_backend.check_cuda_graph_backend",
        return_value=False,
    ):
        hybrid = InklingShortConvHybridAttnBackend(full_attn_mock, backend, [0, 1])
    last_correct_step_indices = torch.tensor([2, 0], dtype=torch.int64)
    mamba_steps_to_track = torch.tensor([1, -1], dtype=torch.int64)
    hybrid.update_mamba_state_after_mtp_verify(
        last_correct_step_indices=last_correct_step_indices,
        mamba_track_indices=verify_batch.mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
    )
    working_slots = backend.forward_state_slots(2, verify_batch.mamba_track_indices)
    w0, w1 = int(working_slots.working[0].item()), int(working_slots.working[1].item())
    t0 = int(working_slots.track[0].item())
    assert torch.allclose(params.conv[0][0, w0].float(), torch.full((3, 16), 13.0))
    assert torch.allclose(params.conv[0][0, w1].float(), torch.full((3, 16), 11.0))
    assert torch.allclose(params.conv[1][1, w0].float(), torch.full((3, 16), 23.0))
    assert torch.allclose(params.conv[0][0, t0].float(), torch.full((3, 16), 12.0))


# ---------------------------------------------------------------------------
# 3. vocab_gather on CPU
# ---------------------------------------------------------------------------


def test_vocab_gather_cpu():
    from sglang.srt.distributed.device_communicators.vocab_gather import (
        LocalVocabGather,
        NcclVocabGather,
        make_vocab_gather,
    )

    assert isinstance(make_vocab_gather(None, local_width=8), LocalVocabGather)
    g1 = SimpleNamespace(world_size=1)
    assert isinstance(make_vocab_gather(g1, local_width=8), LocalVocabGather)

    calls = []

    class _FakeGroup:
        world_size = 4

        def all_gather(self, x, dim=-1):
            calls.append((dim, tuple(x.shape)))
            return torch.cat([x] * self.world_size, dim=dim)

    g4 = _FakeGroup()
    with patch(
        "sglang.srt.distributed.device_communicators.vocab_gather._nvlink_lease",
        side_effect=AssertionError("_nvlink_lease must not be called on CPU"),
    ):
        vg = make_vocab_gather(
            g4, local_width=16, dtype=torch.bfloat16, prefer_nvlink=True
        )
    assert isinstance(vg, NcclVocabGather)

    x = torch.randn(3, 16, dtype=torch.bfloat16)
    out = vg(x)
    assert out.shape == (3, 64)
    stacked = vg.gather_stacked(x)
    assert stacked.shape == (12, 16)
    assert calls == [(-1, (3, 16)), (0, (3, 16))]


# ---------------------------------------------------------------------------
# 4. ops/speculative/inkling2_mtp.py CPU vs reference
# ---------------------------------------------------------------------------


def test_inkling2_mtp_ops_cpu_vs_reference():
    from sglang.kernels.ops.speculative.inkling2_mtp import (
        mix_rmsnorm,
        rnn_gates,
        rnn_inputs,
    )

    g = torch.Generator(device="cpu").manual_seed(42)
    for dtype in (torch.bfloat16, torch.float32):
        rows, taps, hidden, rank, vocab = 5, 4, 32, 16, 64
        # Non-contiguous targets slice
        targets_base = torch.randn(rows, taps + 1, hidden, generator=g, dtype=dtype)
        targets = targets_base[:, :taps, :]
        alpha = torch.softmax(torch.randn(taps, generator=g), dim=0).to(dtype)
        weight = torch.randn(hidden, generator=g, dtype=dtype)
        eps = 1e-6

        got = mix_rmsnorm(targets, alpha, weight, eps)
        mixed = (targets.float() * alpha.float().view(1, taps, 1)).sum(dim=1)
        inv_rms = torch.rsqrt(mixed.pow(2).mean(dim=-1, keepdim=True) + eps)
        ref = (mixed * inv_rms * weight.float()).to(dtype)
        assert got.dtype == dtype
        assert torch.allclose(got.float(), ref.float(), atol=1e-3, rtol=1e-3)

        # Empty rows
        empty_got = mix_rmsnorm(targets[:0], alpha, weight, eps)
        assert empty_got.shape == (0, hidden)

        # rnn_gates
        proj = torch.randn(rows, 3 * rank, generator=g, dtype=dtype)
        state = torch.randn(rows, rank, generator=g, dtype=dtype)
        new_state, latent = rnn_gates(proj, state)

        g_raw, c_raw, o_raw = proj.chunk(3, dim=-1)
        gate_ref = torch.sigmoid(g_raw.float()).to(torch.bfloat16).float()
        kept_ref = (gate_ref * state.float()).to(torch.bfloat16).float()
        blend_ref = (
            (
                (1.0 - gate_ref).to(torch.bfloat16).float()
                * torch.tanh(c_raw.float()).to(torch.bfloat16).float()
            )
            .to(torch.bfloat16)
            .float()
        )
        new_state_ref = (kept_ref + blend_ref).to(dtype)
        latent_ref = torch.tanh(o_raw.float()).to(dtype)
        assert torch.equal(new_state, new_state_ref)
        assert torch.equal(latent, latent_ref)

        es, el = rnn_gates(proj[:0], state[:0])
        assert es.shape == (0, rank) and el.shape == (0, rank)

        # rnn_inputs
        tokens = torch.randint(0, vocab, (rows,), generator=g, dtype=torch.int64)
        embedding = torch.randn(vocab, rank, generator=g, dtype=dtype)
        hidden_in = torch.randn(rows, hidden, generator=g, dtype=dtype)
        f_rnn = torch.randn(rows, hidden, generator=g, dtype=dtype)
        ri_got = rnn_inputs(state, tokens, embedding, hidden_in, f_rnn)
        ri_ref = torch.cat([state, embedding[tokens], hidden_in, f_rnn], dim=-1)
        assert torch.equal(ri_got, ri_ref)

        ri_empty = rnn_inputs(
            state[:0], tokens[:0], embedding, hidden_in[:0], f_rnn[:0]
        )
        assert ri_empty.shape == (0, 2 * rank + 2 * hidden)


# ---------------------------------------------------------------------------
# 5. dspark_components on CPU
# ---------------------------------------------------------------------------


def test_dspark_components_cpu():
    from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
        BlockwiseRejection,
        make_rejection_noise,
    )
    from sglang.kernels.ops.speculative.dspark.dspark_accept import (
        CapCorrectLen,
        FinalizeAcceptLens,
        SelectMixedAccept,
    )
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
    from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
    from sglang.srt.speculative.dspark_components.dspark_observability import (
        DecodeStepObservation,
        DsparkInfoDumper,
        InfoComponent,
        InfoSegment,
    )
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        accept_draft_tokens,
    )

    g = torch.Generator(device="cpu").manual_seed(7)
    bs, gamma, vocab = 3, 4, 32
    stride = gamma + 1

    # markov_sample (greedy + sampled, softcap + vocab_limit)
    base_logits = torch.randn(bs, vocab + 8, generator=g, dtype=torch.bfloat16)
    bias = torch.randn(bs, vocab, generator=g, dtype=torch.bfloat16)
    temps = torch.tensor([0.0, 0.8, 0.0], dtype=torch.float32)
    offset = torch.tensor([3], dtype=torch.int64)
    logits_out = torch.empty_like(base_logits)
    sampled_toks = MarkovSampleStep.execute(
        base_logits=base_logits,
        bias=bias,
        temperatures=temps,
        key=99,
        offset=offset,
        step=1,
        step_dtype=torch.bfloat16,
        vocab_start=4,
        logits_out=logits_out,
        softcap=30.0,
        vocab_limit=vocab + 2,
    )
    assert sampled_toks.shape == (bs,)
    assert ((sampled_toks >= 4) & (sampled_toks < 4 + vocab)).all()
    greedy_expected = logits_out.float()[:, :vocab].argmax(dim=-1) + 4
    assert torch.equal(sampled_toks[[0, 2]], greedy_expected[[0, 2]])

    # blockwise_rejection on CPU
    noise = make_rejection_noise(123, torch.device("cpu"))
    candidates = torch.randint(0, vocab, (bs, stride), generator=g, dtype=torch.int64)
    target_logits = torch.randn(bs * stride, vocab, generator=g, dtype=torch.float32)
    draft_logits = torch.randn(bs, gamma, vocab, generator=g, dtype=torch.float32)
    t_temps = torch.tensor([0.8, 1.0, 0.7], dtype=torch.float32)
    d_temps = torch.tensor([0.8, 1.0, 0.7], dtype=torch.float32)
    cutoff = torch.tensor([5, 3, 5], dtype=torch.int32)

    c_len, bonus, trim = BlockwiseRejection.execute(
        candidates=candidates,
        target_logits=target_logits,
        draft_logits=draft_logits,
        target_temperatures=t_temps,
        draft_temperatures=d_temps,
        noise=noise,
        cutoff_verify_lens=cutoff,
    )
    assert int(noise.counter.item()) == 1
    assert c_len.shape == (bs,) and bonus.shape == (bs,) and trim.shape == (bs,)
    assert (c_len >= 0).all() and (c_len <= cutoff - 1).all()
    assert ((bonus >= 0) & (bonus < vocab)).all()

    # verify accept_draft_tokens: greedy, sampling, mixed
    greedy_info = SimpleNamespace(
        temperatures=torch.zeros(bs, dtype=torch.float32),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )
    draft_input = DFlashDraftInputV2(
        topk_p=torch.empty((bs, 0)),
        topk_index=torch.empty((bs, 0), dtype=torch.int64),
        bonus_tokens=torch.zeros((bs,), dtype=torch.int64),
        new_seq_lens=torch.full((bs,), 10, dtype=torch.int64),
        hidden_states=torch.empty((bs, 0)),
    )
    draft_block_greedy = DraftBlockResult(
        draft_tokens=candidates[:, 1:],
        corrected_logits=None,
        greedy_mask=torch.ones(bs, dtype=torch.bool),
        temperatures=torch.zeros(bs, dtype=torch.float32),
        greedy_draft=True,
    )
    g_len, g_bonus, g_trim = accept_draft_tokens(
        candidates=candidates,
        target_logits=target_logits,
        draft_block=draft_block_greedy,
        sampling_info=greedy_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=stride,
    )
    assert g_len.shape == (bs,) and g_bonus.shape == (bs,) and g_trim.shape == (bs,)

    mixed_info = SimpleNamespace(
        temperatures=torch.tensor([0.0, 0.9, 0.7], dtype=torch.float32),
        is_all_greedy=False,
        is_any_greedy=True,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )
    draft_block_mixed = DraftBlockResult(
        draft_tokens=candidates[:, 1:],
        corrected_logits=draft_logits,
        greedy_mask=torch.tensor([True, False, False], dtype=torch.bool),
        temperatures=torch.tensor([0.0, 0.9, 0.7], dtype=torch.float32),
        greedy_draft=False,
    )
    m_len, m_bonus, m_trim = accept_draft_tokens(
        candidates=candidates,
        target_logits=target_logits,
        draft_block=draft_block_mixed,
        sampling_info=mixed_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=stride,
        rejection_noise=noise,
    )
    assert m_len[0].item() == g_len[0].item() and m_bonus[0].item() == g_bonus[0].item()

    # FinalizeAcceptLens & CapCorrectLen
    prefix_lens = torch.full((bs,), 4, dtype=torch.int32)
    fin = FinalizeAcceptLens.execute(
        correct_len=m_len,
        cap_trim_lens=m_trim,
        prefix_lens=prefix_lens,
    )
    assert torch.equal(fin.commit_lens, m_len + 1)
    assert torch.equal(fin.new_seq_lens, prefix_lens + m_len + 1)
    capped, c_trim = CapCorrectLen.execute(
        correct_len=torch.tensor([4, 3, 2], dtype=torch.int32),
        verify_lens=torch.tensor([3, 5, 2], dtype=torch.int32),
    )
    assert capped.tolist() == [2, 3, 1]
    assert c_trim.tolist() == [2, 0, 1]

    # SelectMixedAccept & CPU sampling fallback (rejection_noise=None)
    s_len, s_bonus, s_trim = accept_draft_tokens(
        candidates=candidates,
        target_logits=target_logits,
        draft_block=DraftBlockResult(
            draft_tokens=candidates[:, 1:],
            corrected_logits=draft_logits,
            greedy_mask=torch.zeros(bs, dtype=torch.bool),
            temperatures=torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32),
            greedy_draft=False,
        ),
        sampling_info=SimpleNamespace(
            temperatures=torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
        ),
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=stride,
        rejection_noise=None,
    )
    sel = SelectMixedAccept.execute(
        greedy_mask=torch.tensor([True, False, True]),
        greedy_len=g_len,
        greedy_bonus=g_bonus,
        greedy_trim=g_trim,
        sampling_len=s_len,
        sampling_bonus=s_bonus,
        sampling_trim=s_trim,
    )
    assert sel.correct_len[0] == g_len[0] and sel.correct_len[1] == s_len[1]

    # DsparkInfoDumper on CPU
    obs = DsparkInfoDumper(
        gamma=gamma,
        verify_num_draft_tokens=stride,
        attn_tp_rank=0,
        device="cpu",
        mode_value="compact",
        components=[
            InfoComponent.CORE,
            InfoComponent.STEP_GPU_TIME,
            InfoComponent.REQS,
        ],
    )
    obs.begin_step()
    with obs.segment(InfoSegment.DRAFT):
        pass
    obs.observe_decode_step(
        DecodeStepObservation(
            forward_ct=1,
            bs=bs,
            mode="compact",
            budget=bs * gamma,
            lag_steps=0,
            num_verify_tokens=bs * stride,
            verify_tokens_local=bs * stride,
            verify_tokens_dp_synced=bs * stride,
            verify_tokens_graph_key=bs * stride,
            predicted_step_ms=1.0,
            predicted_theta=0.5,
            verify_lens=None,
            confidence=None,
            rids=["r0", "r1", "r2"],
            req_pool_indices=torch.arange(bs, dtype=torch.int64),
            prefix_lens=torch.full((bs,), 4, dtype=torch.int64),
            draft_tokens=candidates[:, 1:],
            bonus_tokens=m_bonus,
            correct_len=m_len,
            cap_trim_lens=m_trim,
            commit_lens=m_len + 1,
        )
    )
    dumped = obs.dump()
    assert dumped is not None and len(dumped["records"]) == 1


def test_inkling2_mtp_head_and_worker_cpu():
    from sglang.srt.models.inkling2_mtp import Inkling2MTPHead
    from sglang.srt.runtime_context import override_platform
    from sglang.srt.speculative.inkling2_mtp_worker import Inkling2MTPWorker

    head = Inkling2MTPHead(
        vocab_size=32, markov_rank=16, hidden_size=24, rms_norm_eps=1e-6
    ).to(torch.bfloat16)
    lm_weight = torch.randn(32, 24, dtype=torch.bfloat16)
    head.attach_fold(lm_weight, multiplier=2.0)
    f_rnn = torch.randn(2, 24, dtype=torch.bfloat16)
    head._get_staged_f = lambda: f_rnn
    head._gather_logits = lambda x: x

    hidden = torch.randn(2, 24, dtype=torch.bfloat16)
    state = head.init_state(hidden)
    prev_tokens = torch.tensor([1, 5], dtype=torch.int64)
    bias, next_state = head.step_bias(prev_tokens, hidden_states=hidden, state=state)
    assert bias.shape == (2, 32)
    assert next_state[0].shape == (2, 16)

    # Inkling2MTPWorker.init_attention_backends allows intel_amx / torch_native on CPU
    # and rejects fa4 on CPU.
    worker = Inkling2MTPWorker.__new__(Inkling2MTPWorker)
    worker.draft_model_runner = SimpleNamespace(
        prefill_attention_backend_str="intel_amx",
        decode_attention_backend_str="intel_amx",
        token_to_kv_pool=None,
    )
    worker.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty((4, 8)))
    )
    staged_sizes = []
    worker.draft_model = SimpleNamespace(
        ensure_propose_staging=lambda sz: staged_sizes.append(sz)
    )
    with (
        override_platform(is_cpu=True, is_cuda=False),
        patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2.init_attention_backends",
            return_value=None,
        ),
    ):
        worker.init_attention_backends()
        assert staged_sizes == [4]

        worker.draft_model_runner.prefill_attention_backend_str = "fa4"
        with pytest.raises(ValueError, match="intel_amx"):
            worker.init_attention_backends()


def test_attn_res_and_message_controllers_contiguity():
    from sglang.kernels.ops.attention.inkling2_attn_res import attn_res
    from sglang.kernels.ops.elementwise.inkling2_loop import message_controllers

    for tokens in (1, 5):
        for n_q in (1, 3):
            blocks = [torch.randn(tokens, 64, dtype=torch.float32) for _ in range(4)]
            queries = torch.randn(n_q, 64, dtype=torch.float32)
            msgs = attn_res(blocks, None, queries, normalize_values=True)
            assert all(m.dtype == torch.float32 and m.is_contiguous() for m in msgs)
            dests = tuple(
                torch.randn(tokens, 64, dtype=torch.float32) for _ in range(n_q)
            )
            down = torch.randn(n_q, 16, 128, dtype=torch.bfloat16)
            gain = torch.randn(n_q, 64, 16, dtype=torch.bfloat16)
            forget = torch.randn(n_q, 64, 16, dtype=torch.bfloat16)
            out = message_controllers(dests, msgs, down, gain, forget)
            assert out.shape == (n_q, tokens, 64) and out.dtype == torch.float32


def test_amx_packed_lm_head_paths(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from sglang.srt.layers import logits_processor as logits_module
    from sglang.srt.layers.amx_utils import PackWeightMethod
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from sglang.srt.models import dspark as dspark_module
    from sglang.srt.models.dspark import project_through_lm_head
    from sglang.srt.models.inkling2 import Inkling2LogitsProcessor
    from sglang.srt.models.inkling2_mtp import Inkling2MTPHead
    from sglang.srt.utils import cpu_has_amx_support

    if not cpu_has_amx_support():
        pytest.skip("CPU AMX not supported")

    monkeypatch.setattr(
        logits_module,
        "get_parallel",
        lambda: SimpleNamespace(
            enable_dp_lm_head=False,
            enable_tp_lm_head_all_to_all=False,
            tp_size=1,
            attn_dp_size=1,
        ),
    )
    monkeypatch.setattr(
        logits_module,
        "get_exec",
        lambda: SimpleNamespace(
            features=SimpleNamespace(
                enable_fp32_lm_head=False,
                enable_mis=False,
                aux_hidden_state_capture=None,
                return_aux_hidden_state_layer_ids=None,
            ),
            deterministic=SimpleNamespace(rl_on_policy_target=None),
        ),
    )
    monkeypatch.setattr(
        dspark_module,
        "get_spec",
        lambda: SimpleNamespace(speculative_dspark_lm_head_dtype=None),
    )

    vocab_size, hidden_size, markov_rank = 64, 64, 16
    raw_weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
    lm_head = ParallelLMHead(
        vocab_size,
        hidden_size,
        params_dtype=torch.bfloat16,
        padding_size=16,
        enable_tp=False,
    )
    lm_head.weight.data.copy_(raw_weight)
    lm_head.quant_method = PackWeightMethod(weight_names=["weight"])
    lm_head.quant_method.process_weights_after_loading(lm_head)
    assert getattr(lm_head, "use_intel_amx_backend", False) is True

    hidden = torch.randn(3, hidden_size, dtype=torch.bfloat16)
    ref_logits = hidden.float() @ raw_weight.float().T

    # 1. Inkling2LogitsProcessor._compute_lm_head with use_fp32_lm_head=True
    cfg = SimpleNamespace(
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        final_logit_softcapping=30.0,
    )
    proc = Inkling2LogitsProcessor(cfg, output_dtype=torch.float32)
    out_buf = torch.empty((3, vocab_size), dtype=torch.float32)
    got_logits = proc._compute_lm_head(hidden, lm_head, out=out_buf)
    ref_capped = 30.0 * torch.tanh(ref_logits / 30.0)
    assert got_logits.dtype == torch.float32
    torch.testing.assert_close(got_logits, ref_capped, atol=0.15, rtol=2e-2)

    # 2. project_through_lm_head (dspark)
    got_dspark = project_through_lm_head(hidden, lm_head)
    torch.testing.assert_close(got_dspark.float(), ref_logits, atol=0.15, rtol=2e-2)

    # 3. Unpacked fold equivalence in attach_shared_modules / Inkling2MTPHead
    head_ref = Inkling2MTPHead(vocab_size, markov_rank, hidden_size, 1e-6)
    head_amx = Inkling2MTPHead(vocab_size, markov_rank, hidden_size, 1e-6)
    head_amx.load_state_dict(head_ref.state_dict())
    eye = torch.eye(hidden_size, dtype=lm_head.weight.dtype)
    unpacked_w = torch.ops.sgl_kernel.weight_packed_linear(
        eye, lm_head.weight, None, True
    ).T.contiguous()
    head_ref.attach_fold(raw_weight, None)
    head_amx.attach_fold(unpacked_w, None)
    torch.testing.assert_close(head_amx._fold_weight, head_ref._fold_weight)
    torch.testing.assert_close(head_amx._fold_bias, head_ref._fold_bias)
