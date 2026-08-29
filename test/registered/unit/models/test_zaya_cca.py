"""Numerical and state-cache correctness tests for the ZAYA1 CCA module.

Every path that consumes the per-request conv-state cache -- single-chunk
extend, chunked prefill, decode, a resumed prefix, a batched mix of the two,
and the TP=2 head split -- is pinned against a reference that processes the
whole sequence at once. That equivalence is what fixes the *kind* of value
parked at a chunk boundary (``conv[1]`` holds ``val_proj2 . hs``, not ``hs``),
which no shape or dtype check would catch.

CPU tests use a tiny configuration and a mock pool mirroring the
``HybridReqToTokenPool`` / ``MambaPool`` interface. Tests marked
``skipUnless(torch.cuda.is_available())`` cover the Triton kernels.
"""

import os
import unittest
import unittest.mock
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _ensure_dist_initialized() -> None:
    """Minimal single-rank gloo environment plus the SGLang model-parallel
    groups (TP=1, PP=1, EP=1). ``CCA.__init__`` reads the TP rank and world size
    to size its head-parallel projections, so both must exist first.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29632")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
        )

    if not model_parallel_is_initialized():
        # WARNING, on a machine with a GPU: ``backend="gloo"`` does not keep
        # this on the CPU. ``GroupCoordinator.device`` is fixed at import from
        # the platform, so this builds CUDA communicators (RCCL,
        # custom-allreduce IPC) over a gloo process group, which aborts the
        # process on gfx950 -- asynchronously, so it lands anywhere. Only call
        # it from tests that genuinely need model-parallel state.
        #
        # kwargs, because ``ensure_model_parallel_initialized`` forwards a
        # positional ``backend`` into the ``attention_data_parallel_size`` slot
        # and then explodes on ``int // str``.
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


# ---------------------------------------------------------------------------
# Mock centralized pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MockLayerCache:
    conv: List[torch.Tensor]
    temporal: torch.Tensor


class _MockReqToTokenPool:
    """Minimal stand-in for ``HybridReqToTokenPool``: ``mamba2_layer_cache`` and
    ``get_mamba_indices``, the two methods CCA calls.

    ``tp_size`` controls the per-rank ``in_out_ch`` of ``conv[0]``. ``conv[1]``
    (the ``val_proj2`` lag) is replicated and takes its width from
    ``ZayaConfig.cca_v2_state_dim``, exactly as ``mamba2_cache_params`` sizes it.
    """

    def __init__(self, pool_size: int, cca_config, tp_size: int = 1):
        in_out_ch_full = (
            cca_config.num_attention_heads + cca_config.num_key_value_heads
        ) * cca_config.head_dim
        assert in_out_ch_full % tp_size == 0
        in_out_ch_per_rank = in_out_ch_full // tp_size
        total_padding = (cca_config.cca_time0 - 1) + (cca_config.cca_time1 - 1)
        num_layers = len(cca_config.linear_layer_ids)

        self.conv_state = torch.zeros(
            num_layers, pool_size + 1, in_out_ch_per_rank, total_padding
        )
        self.prev_hs_state = torch.zeros(
            num_layers, pool_size + 1, cca_config.cca_v2_state_dim, 1
        )
        self.temporal = torch.zeros(num_layers, pool_size + 1, 1, 1, 0)
        self._layer_map = {lid: i for i, lid in enumerate(cca_config.linear_layer_ids)}
        self._identity_map = torch.arange(pool_size + 1, dtype=torch.int32)

    def mamba2_layer_cache(self, layer_id: int):
        idx = self._layer_map[layer_id]
        return _MockLayerCache(
            conv=[self.conv_state[idx], self.prev_hs_state[idx]],
            temporal=self.temporal[idx],
        )

    def get_mamba_indices(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        return req_pool_indices.to(torch.int32)


class _MockShortConvBackend:
    """Stand-in for ``ShortConvHybridAttnBackend`` in the CPU unit tests.

    Exposes ``conv_state_metadata`` over a ``_MockReqToTokenPool``, mirroring
    ``ShortConvAttnBackend``: the req -> slot mapping (and, for extend, its host
    ``.tolist()`` mirror) is resolved once per step and shared across conv
    layers, while decode stays entirely on-device.
    """

    def __init__(self, pool: "_MockReqToTokenPool"):
        self.req_to_token_pool = pool
        self.token_to_kv_pool = None
        # Per-forward-step memoization keyed on the ForwardBatch identity,
        # mirroring ShortConvAttnBackend.init_forward_metadata.
        self._step_indices = {}  # id(forward_batch) -> device index tensor
        self._step_slot_ids = {}  # id(forward_batch) -> host list (extend only)
        self.extend_track_calls = []
        self.decode_track_calls = []

    def _resolve_indices(self, forward_batch):
        key = id(forward_batch)
        indices = self._step_indices.get(key)
        if indices is None:
            indices = self.req_to_token_pool.get_mamba_indices(
                forward_batch.req_pool_indices
            ).to(torch.long)
            self._step_indices[key] = indices
        return indices

    def _resolve_slot_ids(self, forward_batch, indices):
        key = id(forward_batch)
        slot_ids = self._step_slot_ids.get(key)
        if slot_ids is None:
            slot_ids = indices.tolist()
            self._step_slot_ids[key] = slot_ids
        return slot_ids

    def conv_state_metadata(self, layer_id, forward_batch):
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvMetadata,
        )

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        indices = self._resolve_indices(forward_batch)  # already int64
        if forward_batch.forward_mode.is_decode_or_idle():
            return ShortConvMetadata(layer_cache=layer_cache, cache_indices=indices)

        slot_ids = self._resolve_slot_ids(forward_batch, indices)
        has_prefix = [int(p) > 0 for p in forward_batch.extend_prefix_lens_cpu]
        return ShortConvMetadata(
            layer_cache=layer_cache,
            cache_indices=indices,
            slot_ids_cpu=slot_ids,
            has_prefix_cpu=has_prefix,
        )

    # The radix mamba-cache track hooks. Inert here (these tests run the
    # no_buffer equivalent), but CCA calls them unconditionally, so the mock
    # must carry them and record that it was reached.
    def track_conv_states_extend(self, conv_states, conv_inputs):
        self.extend_track_calls.append((tuple(conv_states), tuple(conv_inputs)))

    def track_conv_states_decode(self, forward_batch):
        self.decode_track_calls.append(forward_batch)


@contextmanager
def _mock_pool_context(pool: _MockReqToTokenPool):
    """Install a mock ``ForwardContext`` whose ``attn_backend`` exposes both
    ``req_to_token_pool`` and ``conv_state_metadata`` over ``pool``."""
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        set_forward_context,
    )

    backend = _MockShortConvBackend(pool)
    ctx = ForwardContext(attn_backend=backend)
    prev = set_forward_context(ctx)
    try:
        yield backend
    finally:
        set_forward_context(prev)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_forward_batch(
    *,
    is_decode: bool,
    extend_seq_lens_cpu,
    extend_prefix_lens_cpu,
    req_pool_indices,
    input_ids: torch.Tensor,
):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    mode = ForwardMode.DECODE if is_decode else ForwardMode.EXTEND

    forward_batch = SimpleNamespace()
    forward_batch.forward_mode = mode
    forward_batch.input_ids = input_ids
    forward_batch.req_pool_indices = torch.as_tensor(
        req_pool_indices, dtype=torch.int32
    )
    forward_batch.extend_seq_lens_cpu = list(extend_seq_lens_cpu)
    forward_batch.extend_prefix_lens_cpu = list(extend_prefix_lens_cpu)
    return forward_batch


def _make_tiny_config(num_hidden_layers: int = 2):
    from sglang.srt.configs.zaya import ZayaConfig

    return ZayaConfig(
        hidden_size=16,
        ffn_hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_experts=2,
        num_attention_heads=4,
        num_query_groups=2,
        num_key_value_heads=2,
        head_dim=8,
        cca_time0=2,
        cca_time1=2,
        max_position_embeddings=64,
        moe_router_topk=1,
        zaya_mlp_expansion=8,
        attention_bias=False,
    )


def _make_tiny_cca(
    seed: int = 0,
    tp_rank: Optional[int] = None,
    tp_size: Optional[int] = None,
    layer_id: int = 0,
    config=None,
):
    from sglang.srt.models.zaya import CCA

    if config is None:
        config = _make_tiny_config()
    torch.manual_seed(seed)
    cca = CCA(
        config=config,
        cca_num_k_heads=config.num_query_groups,
        cca_num_q_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        cca_time0=config.cca_time0,
        cca_time1=config.cca_time1,
        layer_id=layer_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    cca.eval()

    with torch.no_grad():
        for p in cca.parameters():
            p.data.normal_(mean=0.0, std=0.05)
        cca.temp.data.zero_()

    return cca, config


class TestZayaCCA(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def test_single_chunk_matches_reference(self):
        """A single-chunk extend with empty prefix matches the no-state path."""
        cca, config = _make_tiny_cca(seed=1)
        cca_ref, _ = _make_tiny_cca(seed=1)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        S = 5
        hs = torch.randn(S, cca.hidden_size, dtype=torch.float32) * 0.1

        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        fb = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[0],
            input_ids=torch.arange(S, dtype=torch.int64),
        )
        with _mock_pool_context(pool):
            q, k, v = cca.forward(hs, fb)

        torch.testing.assert_close(q, q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k, k_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v, v_ref, atol=1e-5, rtol=1e-5)

    def test_chunked_prefill_across_request_boundary(self):
        """A resumed chunk must read the PROJECTED boundary value from its slot.

        Two requests in one extend batch -- one resuming a cached prefix, one
        fresh -- exercise the mixed path where the lag row preceding the first
        token comes from the pool for one and from zero for the other. A boundary
        that parks the raw hidden state gives the same shape, the same dtype, no
        error, and wrong ``v`` on every resumed token.
        """
        cca, config = _make_tiny_cca(seed=31)
        cca_ref, _ = _make_tiny_cca(seed=31)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        A0, A1, S_b = 3, 4, 5
        S_a = A0 + A1
        torch.manual_seed(311)
        hs_a = torch.randn(S_a, cca.hidden_size, dtype=torch.float32) * 0.1
        hs_b = torch.randn(S_b, cca.hidden_size, dtype=torch.float32) * 0.1

        qa_ref, ka_ref, va_ref = cca_ref._forward_no_state(hs_a)
        qb_ref, kb_ref, vb_ref = cca_ref._forward_no_state(hs_b)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        # Hold both batches alive: the mock memoizes its per-step slot mirror on
        # id(forward_batch), and a GC'd-then-reused address would false-collide.
        fb0 = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[A0],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[1],
            input_ids=torch.arange(A0, dtype=torch.int64),
        )
        fb1 = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[A1, S_b],
            extend_prefix_lens_cpu=[A0, 0],
            req_pool_indices=[1, 4],
            input_ids=torch.arange(A1 + S_b, dtype=torch.int64),
        )
        with _mock_pool_context(pool):
            # Chunk 0 of request 0 only.
            cca.forward(hs_a[:A0], fb0)
            # Chunk 1 of request 0 (resumes) batched with a fresh request 1.
            q, k, v = cca.forward(torch.cat([hs_a[A0:], hs_b], dim=0), fb1)

        torch.testing.assert_close(q[:A1], qa_ref[A0:], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k[:A1], ka_ref[A0:], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v[:A1], va_ref[A0:], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(q[A1:], qb_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k[A1:], kb_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v[A1:], vb_ref, atol=1e-4, rtol=1e-4)

    def test_chunked_prefill_then_decode_matches_full_sequence(self):
        """Two prefill chunks then decode steps, all against one reference.

        Extends the boundary case above past the prefill: after a resumed chunk
        the request keeps decoding, so a boundary row that is wrong in *kind*
        (raw instead of projected) would also poison the first decode step.
        """
        cca, config = _make_tiny_cca(seed=32)
        cca_ref, _ = _make_tiny_cca(seed=32)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        S0, S1, S2 = 3, 3, 2
        S_total = S0 + S1 + S2
        torch.manual_seed(321)
        hs = torch.randn(S_total, cca.hidden_size, dtype=torch.float32) * 0.1
        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        # ``batches`` keeps every ForwardBatch alive: the mock memoizes its
        # per-step slot mirror on id(forward_batch), so a recycled address would
        # hand a later step the wrong request list.
        batches = []
        for chunk, prefix in ((slice(0, S0), 0), (slice(S0, S0 + S1), S0)):
            n = chunk.stop - chunk.start
            batches.append(
                (
                    hs[chunk],
                    _make_forward_batch(
                        is_decode=False,
                        extend_seq_lens_cpu=[n],
                        extend_prefix_lens_cpu=[prefix],
                        req_pool_indices=[3],
                        input_ids=torch.arange(n, dtype=torch.int64),
                    ),
                )
            )
        for t in range(S2):
            idx = S0 + S1 + t
            batches.append(
                (
                    hs[idx : idx + 1],
                    _make_forward_batch(
                        is_decode=True,
                        extend_seq_lens_cpu=[],
                        extend_prefix_lens_cpu=[],
                        req_pool_indices=[3],
                        input_ids=torch.tensor([0], dtype=torch.int64),
                    ),
                )
            )

        outs = []
        with _mock_pool_context(pool):
            for chunk_hs, fb in batches:
                outs.append(cca.forward(chunk_hs, fb))

        for got, ref in zip(
            (torch.cat([o[i] for o in outs], dim=0) for i in range(3)),
            (q_ref, k_ref, v_ref),
        ):
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    def test_folded_prefill_then_decode_matches_the_two_stage_reference(self):
        """End-to-end with the folded decode conv, including the bias column.

        The other equivalence tests leave ``_decode_conv_folded`` False. This one
        folds the module under test and compares it against an UNFOLDED
        reference, covering the fold, the bias riding in a trailing weight column
        and the state plumbing in one assertion. On CPU the window comes from the
        unfused gather/concat, so this is the ``F.pad`` arm.
        """
        cca, config = _make_tiny_cca(seed=35)
        cca_ref, _ = _make_tiny_cca(seed=35)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())
        cca.fold_decode_conv()
        self.assertTrue(cca._decode_conv_folded)
        self.assertFalse(cca_ref._decode_conv_folded)

        S0, S1 = 4, 3
        torch.manual_seed(351)
        hs = torch.randn(S0 + S1, cca.hidden_size, dtype=torch.float32) * 0.1
        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        batches = [
            (
                hs[:S0],
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S0, dtype=torch.int64),
                ),
            )
        ] + [
            (
                hs[S0 + t : S0 + t + 1],
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                ),
            )
            for t in range(S1)
        ]
        outs = []
        with _mock_pool_context(pool):
            for chunk_hs, fb in batches:
                outs.append(cca.forward(chunk_hs, fb))

        for i, ref in enumerate((q_ref, k_ref, v_ref)):
            got = torch.cat([o[i] for o in outs], dim=0)
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    def test_projected_lag_is_narrower_than_the_hidden_state(self):
        """The whole point: conv[1] shrinks to the val_proj2 output width.

        Guards the arithmetic the pool sizing depends on -- that the cached
        quantity really is ``latent_k_dim / 2`` wide and that the model and the
        config agree on it. If these drift apart the pool entry and the value
        written into it disagree in width.
        """
        cca, config = _make_tiny_cca(seed=33)
        self.assertTrue(cca.cache_projected_v2)
        expected = (config.num_query_groups * config.head_dim) // 2
        self.assertEqual(cca.v2_lag_dim, expected)
        self.assertEqual(config.cca_v2_state_dim, expected)
        self.assertLess(cca.v2_lag_dim, cca.hidden_size)

    def test_biased_projection_falls_back_to_the_raw_hidden_state_lag(self):
        """With a bias, ``W . 0 != 0``, so the projected cache is refused.

        A fresh slot is zero and the first ``val_proj2`` input is defined to be
        zero, which caching the projection reproduces only without a bias term.
        The gate must fall back to the raw hidden state and still be numerically
        right.
        """
        from sglang.srt.configs.zaya import ZayaConfig

        cfg = _make_tiny_config()
        biased = ZayaConfig(
            hidden_size=cfg.hidden_size,
            ffn_hidden_size=cfg.ffn_hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_experts=cfg.num_experts,
            num_attention_heads=cfg.num_attention_heads,
            num_query_groups=cfg.num_query_groups,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            cca_time0=cfg.cca_time0,
            cca_time1=cfg.cca_time1,
            max_position_embeddings=cfg.max_position_embeddings,
            moe_router_topk=cfg.moe_router_topk,
            zaya_mlp_expansion=cfg.zaya_mlp_expansion,
            attention_bias=True,
        )
        self.assertFalse(biased.cca_cache_projected_v2)
        self.assertEqual(biased.cca_v2_state_dim, biased.hidden_size)

        cca, _ = _make_tiny_cca(seed=34, config=biased)
        cca_ref, _ = _make_tiny_cca(seed=34, config=biased)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())
        self.assertFalse(cca.cache_projected_v2)
        self.assertEqual(cca.v2_lag_dim, biased.hidden_size)
        self.assertIsNotNone(cca.val_proj2.bias)

        S0, S1 = 4, 2
        torch.manual_seed(341)
        hs = torch.randn(S0 + S1, cca.hidden_size, dtype=torch.float32) * 0.1
        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=biased)
        self.assertEqual(pool.prev_hs_state.shape[-2], biased.hidden_size)
        # Held for the id(forward_batch) memo (see the chunked-prefill tests).
        batches = [
            (
                hs[:S0],
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S0, dtype=torch.int64),
                ),
            )
        ] + [
            (
                hs[S0 + t : S0 + t + 1],
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                ),
            )
            for t in range(S1)
        ]
        outs = []
        with _mock_pool_context(pool):
            for chunk_hs, fb in batches:
                outs.append(cca.forward(chunk_hs, fb))

        for i, ref in enumerate((q_ref, k_ref, v_ref)):
            got = torch.cat([o[i] for o in outs], dim=0)
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    def test_prefill_then_decode_matches_full_sequence(self):
        """Prefill(S0) followed by ``S1`` single-token decode steps matches a
        one-shot reference over ``S0 + S1`` tokens."""
        cca, config = _make_tiny_cca(seed=2)
        cca_ref, _ = _make_tiny_cca(seed=2)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        S0, S1 = 4, 2
        S_total = S0 + S1
        torch.manual_seed(77)
        hs = torch.randn(S_total, cca.hidden_size, dtype=torch.float32) * 0.1

        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool):
            fb_prefill = _make_forward_batch(
                is_decode=False,
                extend_seq_lens_cpu=[S0],
                extend_prefix_lens_cpu=[0],
                req_pool_indices=[0],
                input_ids=torch.arange(S0, dtype=torch.int64),
            )
            q0, k0, v0 = cca.forward(hs[:S0], fb_prefill)

            q_decodes = [q0]
            k_decodes = [k0]
            v_decodes = [v0]
            for t in range(S1):
                fb_decode = _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                )
                qd, kd, vd = cca.forward(hs[S0 + t : S0 + t + 1], fb_decode)
                q_decodes.append(qd)
                k_decodes.append(kd)
                v_decodes.append(vd)

        q_cat = torch.cat(q_decodes, dim=0)
        k_cat = torch.cat(k_decodes, dim=0)
        v_cat = torch.cat(v_decodes, dim=0)

        torch.testing.assert_close(q_cat, q_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_cat, k_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v_cat, v_ref, atol=1e-4, rtol=1e-4)

    def test_batched_decode_matches_single_decode(self):
        """A two-request batched decode of request 0 must produce the same
        q / k / v tensors as a single-request decode of request 0."""
        cca_single, config = _make_tiny_cca(seed=11)
        cca_batched, _ = _make_tiny_cca(seed=11)
        with torch.no_grad():
            cca_batched.load_state_dict(cca_single.state_dict())

        S0 = 4
        torch.manual_seed(202)
        hs0 = torch.randn(S0, cca_single.hidden_size, dtype=torch.float32) * 0.1
        hs1 = torch.randn(S0, cca_single.hidden_size, dtype=torch.float32) * 0.1
        decode0 = torch.randn(cca_single.hidden_size, dtype=torch.float32) * 0.1
        decode1 = torch.randn(cca_single.hidden_size, dtype=torch.float32) * 0.1

        pool_single = _MockReqToTokenPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool_single):
            cca_single.forward(
                hs0,
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S0, dtype=torch.int64),
                ),
            )
            q_solo, k_solo, v_solo = cca_single.forward(
                decode0.unsqueeze(0),
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                ),
            )

        pool_batched = _MockReqToTokenPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool_batched):
            cca_batched.forward(
                torch.cat([hs0, hs1], dim=0),
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0, S0],
                    extend_prefix_lens_cpu=[0, 0],
                    req_pool_indices=[0, 1],
                    input_ids=torch.arange(2 * S0, dtype=torch.int64),
                ),
            )
            q_batch, k_batch, v_batch = cca_batched.forward(
                torch.stack([decode0, decode1], dim=0),
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0, 1],
                    input_ids=torch.tensor([0, 1], dtype=torch.int64),
                ),
            )

        torch.testing.assert_close(q_batch[0:1], q_solo, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_batch[0:1], k_solo, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_batch[0:1], v_solo, atol=1e-5, rtol=1e-5)

    def test_two_requests_state_isolation(self):
        """A batched prefill of two requests must update only the requests'
        own slots in the centralized pool."""
        cca, config = _make_tiny_cca(seed=4)

        S0, S1 = 3, 2
        hs0 = torch.randn(S0, cca.hidden_size, dtype=torch.float32) * 0.1
        hs1 = torch.randn(S1, cca.hidden_size, dtype=torch.float32) * 0.1
        hs = torch.cat([hs0, hs1], dim=0)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        fb = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S0, S1],
            extend_prefix_lens_cpu=[0, 0],
            req_pool_indices=[2, 5],
            input_ids=torch.arange(S0 + S1, dtype=torch.int64),
        )
        with _mock_pool_context(pool):
            cca.forward(hs, fb)

        layer_cache = pool.mamba2_layer_cache(0)
        conv_state = layer_cache.conv[0]
        lag_state = layer_cache.conv[1]

        self.assertTrue(torch.any(conv_state[2] != 0))
        self.assertTrue(torch.any(conv_state[5] != 0))

        # conv[1] holds the PROJECTED boundary value, not the raw hidden state.
        # The widths agree with the raw hidden state only by accident of a tiny
        # config, so pin the quantity: a mismatch degrades resumed prefixes
        # silently rather than raising.
        self.assertTrue(cca.cache_projected_v2)
        self.assertEqual(lag_state.shape[-2], config.cca_v2_state_dim)
        with torch.no_grad():
            expected0 = cca.val_proj2(hs0[-1:])[0]
            expected1 = cca.val_proj2(hs1[-1:])[0]
        torch.testing.assert_close(
            lag_state[2].squeeze(-1).to(torch.float32),
            expected0.squeeze(0).to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            lag_state[5].squeeze(-1).to(torch.float32),
            expected1.squeeze(0).to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )

        for idx in (0, 1, 3, 4):
            self.assertTrue(torch.all(conv_state[idx] == 0))
            self.assertTrue(torch.all(lag_state[idx] == 0))

    def test_mamba_indices_resolved_once_per_forward_step(self):
        """Two CCA layers driven by one ForwardBatch must trigger exactly one
        ``get_mamba_indices`` lookup and one host ``.tolist()`` sync. The mapping
        is identical for every layer, so recomputing it per layer costs a
        device->host sync per layer.
        """

        class _CountingPool(_MockReqToTokenPool):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.get_mamba_indices_calls = 0

            def get_mamba_indices(self, req_pool_indices):
                self.get_mamba_indices_calls += 1
                return super().get_mamba_indices(req_pool_indices)

        # num_hidden_layers=4 -> CCA (even) layers live at ids 0 and 2.
        config = _make_tiny_config(num_hidden_layers=4)
        self.assertEqual(config.linear_layer_ids, [0, 2])
        cca0, _ = _make_tiny_cca(seed=5, layer_id=0, config=config)
        cca2, _ = _make_tiny_cca(seed=6, layer_id=2, config=config)

        S = 4
        hs = torch.randn(S, config.hidden_size, dtype=torch.float32) * 0.1

        def _fresh_fb():
            return _make_forward_batch(
                is_decode=False,
                extend_seq_lens_cpu=[S],
                extend_prefix_lens_cpu=[0],
                req_pool_indices=[0],
                input_ids=torch.arange(S, dtype=torch.int64),
            )

        pool = _CountingPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool) as backend:
            fb = _fresh_fb()
            cca0.forward(hs, fb)
            cca2.forward(hs, fb)

            # Two CCA layers, one forward step -> one shared lookup, both the
            # device tensor and its host mirror memoized once per step on the
            # backend (ShortConvAttnBackend does this in init_forward_metadata).
            self.assertEqual(pool.get_mamba_indices_calls, 1)
            self.assertIn(id(fb), backend._step_indices)
            self.assertEqual(backend._step_slot_ids[id(fb)], [0])

            # A new forward step (fresh ForwardBatch) resolves the mapping again.
            cca0.forward(hs, _fresh_fb())
            self.assertEqual(pool.get_mamba_indices_calls, 2)

    def test_decode_path_does_not_sync_indices_to_host(self):
        """The decode path indexes the pool entirely on-device, so it must not
        populate the host-side index cache (keeping it CUDA-graph friendly)."""
        cca, config = _make_tiny_cca(seed=7)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool) as backend:
            # Keep a reference to the extend batch so its id() cannot be recycled
            # by the later decode batch (the mock keys its per-step memo on
            # id(forward_batch); a GC'd-then-reused address would false-collide).
            fb_extend = _make_forward_batch(
                is_decode=False,
                extend_seq_lens_cpu=[3],
                extend_prefix_lens_cpu=[0],
                req_pool_indices=[0],
                input_ids=torch.arange(3, dtype=torch.int64),
            )
            cca.forward(
                torch.randn(3, config.hidden_size, dtype=torch.float32) * 0.1,
                fb_extend,
            )
            fb_decode = _make_forward_batch(
                is_decode=True,
                extend_seq_lens_cpu=[],
                extend_prefix_lens_cpu=[],
                req_pool_indices=[0],
                input_ids=torch.tensor([0], dtype=torch.int64),
            )
            cca.forward(
                torch.randn(1, config.hidden_size, dtype=torch.float32) * 0.1,
                fb_decode,
            )

            # Decode resolves device indices, but the host ``.tolist()`` mirror
            # is only built by the extend path -- so the decode step stays
            # entirely on-device (CUDA-graph friendly).
            self.assertIn(id(fb_decode), backend._step_indices)
            self.assertNotIn(id(fb_decode), backend._step_slot_ids)


class TestCCAStateStepKernel(CustomTestCase):
    """The fused decode state step must be bit-identical to the torch chain.

    ``cca_state_step`` re-derives the conv history shift (``new[w] = old[w+1]``,
    last tap = this token) and the read-before-overwrite ordering of the lag
    gather/scatter while mutating the pools in place. An off-by-one in the shift,
    or writing the slot before reading it, corrupts only the *next* step for that
    request, so pin exactness on both the returned tensors and the pools.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_bit_identical_to_torch_chain(self):
        from sglang.kernels.ops.attention import cca_state_step as kernel

        dev = "cuda"
        torch.manual_seed(5)
        # total_padding 3 is a shift longer than ZAYA1's 2, where an off-by-one
        # in the history roll would otherwise be invisible. (600, 1100) spans
        # several channel AND hidden tiles (BLOCK_C=256, BLOCK_H=512), so a wrong
        # tile offset shows up; the small shapes are one tile per axis.
        shapes = ((64, 32, 2), (48, 16, 3), (600, 1100, 2))
        for num_channels, hidden_size, total_padding in shapes:
            for num_tokens in (1, 6, 40):
                with self.subTest(
                    c=num_channels, h=hidden_size, p=total_padding, t=num_tokens
                ):
                    slots = (
                        torch.randperm(63, device=dev)[:num_tokens].to(torch.long) + 1
                    )
                    qk = torch.randn(
                        num_tokens, num_channels, device=dev, dtype=torch.bfloat16
                    )
                    hs = torch.randn(
                        num_tokens, hidden_size, device=dev, dtype=torch.bfloat16
                    )
                    conv0 = torch.randn(
                        64,
                        num_channels,
                        total_padding,
                        device=dev,
                        dtype=torch.bfloat16,
                    )
                    prev0 = torch.randn(
                        64, hidden_size, 1, device=dev, dtype=torch.bfloat16
                    )

                    conv_ref, prev_ref = conv0.clone(), prev0.clone()
                    conv_got, prev_got = conv0.clone(), prev0.clone()

                    with torch.no_grad():
                        left = conv_ref.index_select(0, slots).to(hs.dtype)
                        window_ref = torch.cat([left, qk.unsqueeze(-1)], dim=-1)
                        conv_ref.index_copy_(
                            0,
                            slots,
                            window_ref[..., -total_padding:].to(conv_ref.dtype),
                        )
                        prevhs_ref = (
                            prev_ref.index_select(0, slots).squeeze(-1).to(hs.dtype)
                        )
                        prev_ref.index_copy_(
                            0, slots, hs.unsqueeze(-1).to(prev_ref.dtype)
                        )

                        self.assertTrue(
                            kernel.covered(
                                qk, hs, conv_got, prev_got, slots, total_padding
                            )
                        )
                        window_got, prevhs_got = kernel.cca_state_step(
                            qk, hs, conv_got, prev_got, slots, total_padding
                        )

                    # Bit-exact: the kernel only moves data, no arithmetic.
                    self.assertTrue(torch.equal(window_got, window_ref))
                    self.assertTrue(torch.equal(prevhs_got, prevhs_ref))
                    self.assertTrue(torch.equal(conv_got, conv_ref))
                    self.assertTrue(torch.equal(prev_got, prev_ref))

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_lag_free_variant_matches_the_conv_only_chain(self):
        """``HAS_LAG=False`` must leave the conv half byte-identical.

        A rank whose V heads all come from ``val_proj1`` passes no lag tensors, so
        the lag block compiles out. The conv window and the conv-history shift
        must be unchanged by that -- a constexpr that accidentally gated the conv
        loop too would return plausible-looking zeros.
        """
        from sglang.kernels.ops.attention import cca_state_step as kernel

        dev = "cuda"
        torch.manual_seed(7)
        num_channels, total_padding, num_tokens = 48, 2, 5
        slots = torch.randperm(32, device=dev)[:num_tokens].to(torch.long) + 1
        qk = torch.randn(num_tokens, num_channels, device=dev, dtype=torch.bfloat16)
        conv0 = torch.randn(
            64, num_channels, total_padding, device=dev, dtype=torch.bfloat16
        )
        prev0 = torch.randn(64, 16, 1, device=dev, dtype=torch.bfloat16)
        hs = torch.randn(num_tokens, 16, device=dev, dtype=torch.bfloat16)

        conv_ref, prev_ref = conv0.clone(), prev0.clone()
        conv_got, prev_got = conv0.clone(), prev0.clone()
        with torch.no_grad():
            # Reference: run WITH the lag, then discard the lag results.
            self.assertTrue(
                kernel.covered(qk, hs, conv_ref, prev_ref, slots, total_padding)
            )
            window_ref, _ = kernel.cca_state_step(
                qk, hs, conv_ref, prev_ref, slots, total_padding
            )
            self.assertTrue(
                kernel.covered(qk, None, conv_got, None, slots, total_padding)
            )
            window_got, lag_got = kernel.cca_state_step(
                qk, None, conv_got, None, slots, total_padding
            )

        self.assertIsNone(lag_got)
        self.assertTrue(torch.equal(window_got, window_ref))
        self.assertTrue(torch.equal(conv_got, conv_ref))
        # The lag pool the lag-free call was never handed stays untouched.
        self.assertTrue(torch.equal(prev_got, prev0))

    def test_uncovered_inputs_fall_back(self):
        # covered() gates an in-place pool mutation, so its negative branches
        # matter more than usual: a mismatched dtype would have the kernel write
        # raw bits into the pool.
        from sglang.kernels.ops.attention import cca_state_step as kernel

        qk = torch.randn(4, 8)
        hs = torch.randn(4, 6)
        conv = torch.randn(16, 8, 2)
        prev = torch.randn(16, 6, 1)
        slots = torch.arange(4)
        # CPU tensors are not served.
        self.assertFalse(kernel.covered(qk, hs, conv, prev, slots, 2))
        # Missing slot indices (before the backend resolves them).
        self.assertFalse(kernel.covered(qk, hs, conv, prev, None, 2))
        # A single tap leaves no history to shift.
        self.assertFalse(kernel.covered(qk, hs, conv, prev, slots, 0))
        # Pool dtype must match the value written into it.
        self.assertFalse(
            kernel.covered(qk.to(torch.bfloat16), hs, conv, prev, slots, 2)
        )
        # Same for the lag pool: it is written from lag_now with no conversion.
        self.assertFalse(
            kernel.covered(qk, hs, conv, prev.to(torch.bfloat16), slots, 2)
        )
        # The lag stream is either fully specified or fully absent. A
        # half-specified pair must be refused rather than guessed at: guessing
        # "no lag" skips the pool write and the next step reads a stale value,
        # guessing "lag" writes the wrong width into the pool. Neither raises.
        self.assertFalse(kernel.covered(qk, hs, conv, None, slots, 2))
        self.assertFalse(kernel.covered(qk, None, conv, prev, slots, 2))
        # Widths must agree between the value and the pool entry it lands in.
        self.assertFalse(kernel.covered(qk, torch.randn(4, 5), conv, prev, slots, 2))


class TestCCAQKMixKernel(CustomTestCase):
    """The fused q/k head-mix kernel must match the torch chain it replaces.

    ``cca_qk_mix`` re-derives the GQA group indexing (q head ==
    g * gqa_groups + j), the 0.5 blend weights, the per-k-head temperature and
    the two RMS normalizations from scratch. A transposed group index or a
    temperature applied to q instead of k still produces plausible tensors, so
    pin the equivalence.
    """

    def _run(
        self,
        num_q_heads: int,
        num_k_heads: int,
        num_tokens: int,
        head_dim: int = 32,
    ):
        import torch as _torch

        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        dev = "cuda"
        _torch.manual_seed(11)
        cca = _make_tiny_cca(seed=2)[0]
        # Drive the reference through the real module helpers so the test tracks
        # them rather than a re-derivation of the same formula.
        cca.num_q_heads = num_q_heads
        cca.num_k_heads = num_k_heads
        cca.gqa_groups = num_q_heads // num_k_heads
        cca.head_dim = head_dim
        cca.sqrt_head_dim = head_dim**0.5
        cca.clamp_temp = False
        cca.temp = torch.nn.Parameter(_torch.rand(num_k_heads) + 0.5)

        conv_qk = (
            _torch.randn(
                num_tokens,
                (num_q_heads + num_k_heads) * head_dim,
                dtype=_torch.bfloat16,
            )
            * 0.3
        )
        pre_q = (
            _torch.randn(num_tokens, num_q_heads * head_dim, dtype=_torch.bfloat16)
            * 0.3
        )
        base_k = (
            _torch.randn(num_tokens, num_k_heads * head_dim, dtype=_torch.bfloat16)
            * 0.3
        )

        with _torch.no_grad():
            q_ref, k_ref = cca._add_grouped_qk_means(
                conv_qk[:, : num_q_heads * head_dim].view(
                    num_tokens, num_q_heads, head_dim
                ),
                conv_qk[:, num_q_heads * head_dim :].view(
                    num_tokens, num_k_heads, head_dim
                ),
                pre_q.view(num_tokens, num_q_heads, head_dim),
                base_k.view(num_tokens, num_k_heads, head_dim),
            )
            q_ref, k_ref = cca._normalize_qk(q_ref, k_ref)

            k_scale = (cca.temp.detach().float() * cca.sqrt_head_dim).to(dev)
            args = (conv_qk.to(dev), pre_q.to(dev), base_k.to(dev), k_scale)
            self.assertTrue(
                kernel.covered(*args, num_q_heads, num_k_heads, head_dim),
                "kernel should cover these shapes",
            )
            q_got, k_got = kernel.cca_qk_mix(
                *args,
                num_q_heads=num_q_heads,
                num_k_heads=num_k_heads,
                head_dim=head_dim,
                q_scale=cca.sqrt_head_dim,
            )

        torch.testing.assert_close(
            q_got.cpu(), q_ref, rtol=2e-3, atol=2e-3, check_dtype=False
        )
        torch.testing.assert_close(
            k_got.cpu(), k_ref, rtol=2e-3, atol=2e-3, check_dtype=False
        )

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_matches_torch_chain_across_gqa_shapes(self):
        _ensure_dist_initialized()
        # 8:1 is ZAYA1-74B at attn_tp=2; 4:1 is 8B at tp=1; 1:1 exercises the
        # degenerate group where the k blend reduces to a single q head. 3:1 is
        # the only one whose group is not a power of two, so it is the case where
        # the [G, HD] tile is padded and the masked rows must contribute nothing
        # to either reduction.
        for num_q_heads, num_k_heads in ((8, 1), (8, 2), (2, 2), (6, 2)):
            for num_tokens in (1, 5):
                with self.subTest(q=num_q_heads, k=num_k_heads, t=num_tokens):
                    self._run(num_q_heads, num_k_heads, num_tokens)

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_matches_torch_chain_at_the_serving_shape(self):
        """head_dim 128 and a decode-sized batch, the launch config in prod.

        The other case pins the algebra at head_dim 32, one ROCm wavefront twice
        over; 128 is what the block size and warp count are chosen for, so a
        reduction that only happens to be right at 32 lanes shows up here.
        """
        _ensure_dist_initialized()
        for num_tokens in (1, 40):
            with self.subTest(t=num_tokens):
                self._run(8, 1, num_tokens, head_dim=128)

    def test_uncovered_inputs_fall_back(self):
        # covered() is the only thing standing between an unsupported shape and a
        # wrong-answer kernel launch, so its negative branches must hold. Most
        # importantly a missing scale vector (weights not folded yet) must report
        # False so the torch path runs.
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        conv_qk = torch.randn(4, 9 * 32)
        pre_q = torch.randn(4, 8 * 32)
        base_k = torch.randn(4, 1 * 32)
        scale = torch.ones(1)
        # No folded scales yet.
        self.assertFalse(kernel.covered(conv_qk, pre_q, base_k, None, 8, 1, 32))
        # CPU tensors are not served by the Triton path.
        self.assertFalse(kernel.covered(conv_qk, pre_q, base_k, scale, 8, 1, 32))
        # Head dim beyond one block.
        self.assertFalse(kernel.covered(conv_qk, pre_q, base_k, scale, 8, 1, 4096))
        # q heads not divisible by k heads.
        self.assertFalse(kernel.covered(conv_qk, pre_q, base_k, scale, 8, 3, 32))
        # A group too wide to hold as one [G, HD] register tile.
        wide_q = torch.randn(4, 64 * 32)
        wide_conv = torch.randn(4, 65 * 32)
        self.assertFalse(kernel.covered(wide_conv, wide_q, base_k, scale, 64, 1, 32))


class TestCCAQKMixRope(CustomTestCase):
    """The fused partial-rotary RoPE inside ``cca_qk_mix`` must compute the same
    function as the ``_normalize_qk`` -> ``rotary_emb`` chain it replaces.

    Not bit-identical, deliberately: the chain rounds q/k to bf16 before rotating
    and (on ROCm) reads a bf16 cos/sin cache, while the fused kernel keeps the
    normalized head in fp32 across the rotation and rounds once at the store. So
    the bf16 comparison is ``allclose`` at bf16 tolerance, and a separate fp64
    reference pins which side the difference falls on: the fused result must be
    at least as close to exact as the chain.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        from sglang.srt.runtime_context import get_context

        cls._server_args_override = get_context().override_server_args(
            model_path="dummy"
        )
        cls._server_args_override.install()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server_args_override.restore()

    # -- reference chains ---------------------------------------------------

    @staticmethod
    def _reference_chain(cca, conv_qk, pre_q, base_k, rotary_emb, positions, dtype):
        """``_add_grouped_qk_means`` -> ``_normalize_qk`` -> cast -> ``rotary_emb``.

        Exactly what ``ZayaAttention.forward`` did before the fusion, driven
        through the real module helpers so the test tracks them.
        """
        T = conv_qk.shape[0]
        nq, nk, hd = cca.num_q_heads, cca.num_k_heads, cca.head_dim
        with torch.no_grad():
            q, k = cca._add_grouped_qk_means(
                conv_qk[:, : nq * hd].view(T, nq, hd),
                conv_qk[:, nq * hd :].view(T, nk, hd),
                pre_q.view(T, nq, hd),
                base_k.view(T, nk, hd),
            )
            q, k = cca._normalize_qk(q, k)
            q = q.to(dtype).flatten(1).contiguous()
            k = k.to(dtype).flatten(1).contiguous()
            q, k = rotary_emb(positions, q, k)
        return q.view(T, nq, hd), k.view(T, nk, hd)

    @staticmethod
    def _reference_fp64(cca, conv_qk, pre_q, base_k, cos_sin_cache, positions):
        """The same function evaluated in fp64 from the stored cos/sin cache.

        The cache is used as stored (bf16 on ROCm) because both the fused kernel
        and the chain read that same table -- its quantization is common-mode and
        is not what this reference measures.
        """
        T = conv_qk.shape[0]
        nq, nk, hd = cca.num_q_heads, cca.num_k_heads, cca.head_dim
        g = cca.gqa_groups
        eps = 1e-12
        cq = conv_qk[:, : nq * hd].reshape(T, nk, g, hd).double()
        ck = conv_qk[:, nq * hd :].reshape(T, nk, hd).double()
        pq = pre_q.reshape(T, nk, g, hd).double()
        bk = base_k.reshape(T, nk, hd).double()

        q = cq + 0.5 * pq + 0.5 * bk.unsqueeze(-2)
        k = ck + 0.5 * pq.mean(dim=-2) + 0.5 * bk
        sqrt_hd = float(cca.sqrt_head_dim)
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + eps) * sqrt_hd
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + eps) * sqrt_hd
        temp = cca.temp.detach().double().view(1, nk, 1)
        k = k * temp
        q = q.reshape(T, nq, hd)

        rot = cos_sin_cache.shape[-1]
        half = rot // 2
        cs = cos_sin_cache.double()[positions.long()]
        cos, sin = cs[:, :half], cs[:, half:]

        def _rot(x):
            lo = x[..., :half]
            hi = x[..., half:rot]
            c = cos.unsqueeze(1)
            s = sin.unsqueeze(1)
            return torch.cat([lo * c - hi * s, hi * c + lo * s, x[..., rot:]], dim=-1)

        return _rot(q), _rot(k)

    # -- the fused run ------------------------------------------------------

    def _run(self, *, rope_base: int, num_tokens: int, dtype=torch.bfloat16):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel
        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.models.zaya import CCARope

        dev = "cuda"
        num_q_heads, num_k_heads, head_dim = 8, 1, 128
        torch.manual_seed(7)

        cca = _make_tiny_cca(seed=3)[0]
        cca.num_q_heads = num_q_heads
        cca.num_k_heads = num_k_heads
        cca.gqa_groups = num_q_heads // num_k_heads
        cca.head_dim = head_dim
        cca.sqrt_head_dim = head_dim**0.5
        cca.clamp_temp = False
        # ON THE DEVICE, not on the host: ``_normalize_qk`` and the fp64
        # reference multiply ``self.temp`` into CUDA activations. The tensors go
        # to CUDA, never the other way round -- only the two pure torch
        # expression trees are borrowed off this CPU-built module, which is never
        # itself called, so no fused op resolves a device kernel over host
        # pointers.
        cca.temp = torch.nn.Parameter((torch.rand(num_k_heads) + 0.5).to(dev))

        rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=256,
            base=rope_base,
            is_neox_style=True,
            partial_rotary_factor=0.5,
        )
        # ZAYA1's split: the cache is half the head dim wide, which is exactly
        # what the shared fused-rope kernel cannot express.
        self.assertEqual(rotary_emb.rotary_dim, head_dim // 2)
        self.assertEqual(rotary_emb.cos_sin_cache.shape[-1], head_dim // 2)

        # Built on the device up front rather than moved at each use: one
        # forgotten ``.to(dev)`` is a device mismatch, and there is no reason for
        # a host copy to exist at all.
        shape = (num_tokens, (num_q_heads + num_k_heads) * head_dim)
        conv_qk = torch.randn(*shape, dtype=dtype, device=dev) * 0.3
        pre_q = (
            torch.randn(num_tokens, num_q_heads * head_dim, dtype=dtype, device=dev)
            * 0.3
        )
        base_k = (
            torch.randn(num_tokens, num_k_heads * head_dim, dtype=dtype, device=dev)
            * 0.3
        )
        positions = torch.arange(num_tokens, dtype=torch.int64, device=dev)

        # ``get_rope`` caches process-wide and builds the cache on whatever
        # device was current, so pull the shared module onto the activations'
        # device before anything reads its buffer.
        rotary_dev = rotary_emb.to(dev)
        cache = rotary_dev.cos_sin_cache
        self.assertEqual(cache.device.type, conv_qk.device.type)
        q_ref, k_ref = self._reference_chain(
            cca, conv_qk, pre_q, base_k, rotary_dev, positions, dtype
        )
        q_f64, k_f64 = self._reference_fp64(
            cca, conv_qk.float(), pre_q.float(), base_k.float(), cache, positions
        )

        rope = CCARope.of(rotary_dev, positions)
        self.assertIsNotNone(rope, "the plain RotaryEmbedding must be fusable")
        # ``conv_qk.device`` (cuda:0), NOT ``torch.device("cuda")``: the index
        # is part of a device's identity. Assert on the reason, since a bare
        # False would mean the test silently never took the fused path.
        reason = kernel.rope_decline_reason(
            rope.positions,
            rope.cos_sin_cache,
            rope.rotary_dim,
            head_dim=head_dim,
            num_tokens=num_tokens,
            is_neox_style=rope.is_neox_style,
            device=conv_qk.device,
        )
        self.assertIsNone(reason, f"rope fusion declined: {reason}")

        k_scale = cca.temp.detach().float() * cca.sqrt_head_dim
        q_got, k_got = kernel.cca_qk_mix(
            conv_qk,
            pre_q,
            base_k,
            k_scale,
            num_q_heads=num_q_heads,
            num_k_heads=num_k_heads,
            head_dim=head_dim,
            q_scale=cca.sqrt_head_dim,
            out_dtype=dtype,
            positions=rope.positions,
            cos_sin_cache=rope.cos_sin_cache,
            rotary_dim=rope.rotary_dim,
        )

        # Elements are O(1) (RMS-normalized, then scaled by sqrt(head_dim)) and
        # the chain rounds twice more than the fused path, so bf16 is a couple of
        # ulps: 2 * 2^-8 ~= 8e-3 absolute. The fp32 budget is derived, not
        # guessed: a three-way split RMS reduction (~7e-7 relative), an
        # approximate ``tl.rsqrt`` (~2.4e-7) and three more fp32 rotation ops
        # come to ~2e-6, so 1e-4 leaves two decades of headroom.
        tol = 8e-3 if dtype is torch.bfloat16 else 1e-4
        torch.testing.assert_close(q_got, q_ref, rtol=tol, atol=tol)
        torch.testing.assert_close(k_got, k_ref, rtol=tol, atol=tol)

        # The fused path must be at least as close to the fp64 evaluation as
        # the chain, since it carries the rotation in fp32 and rounds once.
        # bf16 only: at fp32 both paths carry full precision and the residual
        # difference is rounding order, so "which is closer" would be a flake.
        if dtype is not torch.bfloat16:
            return
        for got, ref, exact, name in (
            (q_got, q_ref, q_f64, "q"),
            (k_got, k_ref, k_f64, "k"),
        ):
            err_fused = (got.double() - exact).abs().max().item()
            err_chain = (ref.double() - exact).abs().max().item()
            self.assertLessEqual(
                err_fused,
                err_chain * 1.05 + 1e-6,
                f"fused {name} drifted further from fp64 than the chain "
                f"({err_fused:.3e} vs {err_chain:.3e})",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_matches_the_unfused_chain_on_both_rope_bases(self):
        # ZAYA1-74B builds two caches and picks per layer: rope_theta 1e7 for the
        # full-attention layers, swa_rotary_base 10000 for the sliding ones. The
        # fused path must read the layer's OWN cache, and a base mix-up is
        # invisible at position 0 -- so exercise both, over several positions.
        for rope_base in (10_000, 10_000_000):
            for num_tokens in (1, 33):
                with self.subTest(base=rope_base, t=num_tokens):
                    self._run(rope_base=rope_base, num_tokens=num_tokens)

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_fp32_output_matches_the_chain_tightly(self):
        # With an fp32 store the only remaining difference from the chain is the
        # intermediate rounding it does and this path does not, so the gap
        # collapses by ~80x (8e-3 -> 1e-4; see the tolerance derivation in _run).
        self._run(rope_base=10_000_000, num_tokens=17, dtype=torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_the_two_layer_caches_give_distinct_results(self):
        # Guard against the fused path latching a model-wide cache pointer: with
        # different bases the same positions must produce different rotations.
        from sglang.kernels.ops.attention import cca_qk_mix as kernel
        from sglang.srt.layers.rotary_embedding import get_rope

        head_dim, nq, nk, T = 128, 8, 1, 8
        torch.manual_seed(5)
        conv_qk = torch.randn(T, (nq + nk) * head_dim, dtype=torch.bfloat16).cuda()
        pre_q = torch.randn(T, nq * head_dim, dtype=torch.bfloat16).cuda()
        base_k = torch.randn(T, nk * head_dim, dtype=torch.bfloat16).cuda()
        positions = torch.arange(1, T + 1, dtype=torch.int64).cuda()
        scale = torch.ones(nk, device="cuda")

        outs = []
        for base in (10_000, 10_000_000):
            rope = get_rope(
                head_size=head_dim,
                rotary_dim=head_dim,
                max_position=256,
                base=base,
                is_neox_style=True,
                partial_rotary_factor=0.5,
            ).cuda()
            outs.append(
                kernel.cca_qk_mix(
                    conv_qk,
                    pre_q,
                    base_k,
                    scale,
                    num_q_heads=nq,
                    num_k_heads=nk,
                    head_dim=head_dim,
                    q_scale=head_dim**0.5,
                    out_dtype=torch.bfloat16,
                    positions=positions,
                    cos_sin_cache=rope.cos_sin_cache,
                    rotary_dim=rope.rotary_dim,
                )
            )
        self.assertFalse(torch.equal(outs[0][0], outs[1][0]))
        self.assertFalse(torch.equal(outs[0][1], outs[1][1]))


class TestCCAQKMixRopeCoverage(CustomTestCase):
    """``rope_covered`` is the only thing between an unsupported head shape and a
    silently wrong rotation, so pin its negative branches.

    These run without a GPU: the head-geometry half is split out as
    ``rope_geometry_covered`` precisely so it can be checked on CPU, and the
    device half is checked by handing ``rope_covered`` CPU tensors.
    """

    def test_geometry_rejects_a_non_half_rotary_dim(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        # ZAYA1's shape: partial_rotary_factor 0.5 over head_dim 128.
        self.assertTrue(kernel.rope_geometry_covered(64, 128))
        # Full rotary (the common case for other models) is NOT covered: the
        # pass-through tail would be empty and the tile widths no longer match.
        self.assertFalse(kernel.rope_geometry_covered(128, 128))
        # A quarter-rotary split leaves a 3/4-width tail.
        self.assertFalse(kernel.rope_geometry_covered(32, 128))
        # And a rotary wider than the head is nonsense.
        self.assertFalse(kernel.rope_geometry_covered(192, 128))
        self.assertFalse(kernel.rope_geometry_covered(0, 128))

    def test_geometry_rejects_an_odd_half(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        # head_dim 12 -> rotary_dim 6 -> half 3, an odd per-lane half.
        self.assertFalse(kernel.rope_geometry_covered(6, 12))
        # head_dim 20 -> rotary_dim 10 -> half 5.
        self.assertFalse(kernel.rope_geometry_covered(10, 20))
        # The neighbouring even halves are fine.
        self.assertTrue(kernel.rope_geometry_covered(8, 16))
        self.assertTrue(kernel.rope_geometry_covered(4, 8))

    def test_geometry_rejects_gptj_layout(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        # GPT-J interleaves the rotated pair inside a lane -- the cross-lane case
        # the three-tile split exists to avoid.
        self.assertFalse(kernel.rope_geometry_covered(64, 128, is_neox_style=False))

    def test_rope_covered_rejects_cpu_and_malformed_inputs(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        hd, rot, T = 128, 64, 4
        pos = torch.arange(T, dtype=torch.int64)
        cache = torch.randn(256, rot)
        common = dict(head_dim=hd, num_tokens=T)

        # Missing either argument (no rope offered at all).
        self.assertFalse(kernel.rope_covered(None, cache, rot, **common))
        self.assertFalse(kernel.rope_covered(pos, None, rot, **common))
        # CPU tensors are not served by the Triton path.
        self.assertFalse(kernel.rope_covered(pos, cache, rot, **common))
        # A cache whose last dim is not rotary_dim (e.g. a full-rotary cache
        # handed in with a partial rotary_dim) must be refused, not sliced.
        self.assertFalse(kernel.rope_covered(pos, torch.randn(256, 128), rot, **common))
        # Float positions, a token-count mismatch and a 2-D (mrope) position
        # layout are all refused.
        self.assertFalse(kernel.rope_covered(pos.float(), cache, rot, **common))
        self.assertFalse(
            kernel.rope_covered(pos[:2], cache, rot, head_dim=hd, num_tokens=T)
        )
        self.assertFalse(kernel.rope_covered(pos.view(2, 2), cache, rot, **common))

    def test_cca_rope_snapshot_declines_non_plain_rotaries(self):
        _ensure_dist_initialized()
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
        from sglang.srt.models.zaya import CCARope

        pos = torch.arange(4, dtype=torch.int64)

        # A stand-in for a scaled / mrope subclass: not the exact base class, so
        # the snapshot declines rather than reading a cache it does not model.
        class _Subclass(RotaryEmbedding):
            pass

        sub = _Subclass(128, 64, 64, 10000, True, torch.float32)
        self.assertIsNone(CCARope.of(sub, pos))

        base = RotaryEmbedding(128, 64, 64, 10000, True, torch.float32)
        snap = CCARope.of(base, pos)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rotary_dim, 64)
        self.assertTrue(snap.is_neox_style)
        # The buffer is handed over by reference, not copied: the kernel must
        # read whatever the module currently holds.
        self.assertIs(snap.cos_sin_cache, base.cos_sin_cache)
        # A 2-D (multimodal) position layout declines too.
        self.assertIsNone(CCARope.of(base, pos.view(2, 2)))

    def test_cpu_fallback_leaves_rope_to_the_caller(self):
        # On the torch fallback (CPU, or before fold_qk_scales) nothing is
        # rotated inside CCA, so ``rope_fused`` must stay False or
        # ``ZayaAttention`` would skip the rotary entirely.
        _ensure_dist_initialized()
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
        from sglang.srt.models.zaya import CCARope

        cca = _make_tiny_cca(seed=1)[0]
        T = 3
        hd, nq, nk = cca.head_dim, cca.num_q_heads, cca.num_k_heads
        qk_out = torch.randn(T, (nq + nk) * hd)
        q_raw = torch.randn(T, nq * hd)
        k_raw = torch.randn(T, nk * hd)
        rope = CCARope.of(
            RotaryEmbedding(hd, hd // 2, 64, 10000, True, torch.float32),
            torch.arange(T, dtype=torch.int64),
        )
        query, key = cca._mix_and_normalize_qk(
            qk_out,
            q_raw,
            k_raw,
            qk_out[:, : nq * hd].view(T, nq, hd),
            qk_out[:, nq * hd :].view(T, nk, hd),
            q_raw.view(T, nq, hd),
            k_raw.view(T, nk, hd),
            out_dtype=torch.float32,
            rope=rope,
        )
        self.assertFalse(cca.rope_fused)
        # And the fallback still matches the two torch helpers exactly.
        ref_q, ref_k = cca._add_grouped_qk_means(
            qk_out[:, : nq * hd].view(T, nq, hd),
            qk_out[:, nq * hd :].view(T, nk, hd),
            q_raw.view(T, nq, hd),
            k_raw.view(T, nk, hd),
        )
        ref_q, ref_k = cca._normalize_qk(ref_q, ref_k)
        torch.testing.assert_close(query, ref_q)
        torch.testing.assert_close(key, ref_k)


class TestCCAQKMixKVStore(CustomTestCase):
    """The fused KV scatter must write exactly what ``set_kv_buffer`` would.

    A wrong slot, a wrong head stride or a missed ``full_to_swa`` indirection
    does not crash, it corrupts KV. So compare the pool contents against a
    second, identical pool driven through the real ``set_kv_buffer``, on both a
    sliding-window layer (the write goes through ``full_to_swa``) and a
    full-attention one.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        from sglang.srt.runtime_context import get_context

        cls._server_args_override = get_context().override_server_args(
            model_path="dummy"
        )
        cls._server_args_override.install()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server_args_override.restore()

    @staticmethod
    def _make_pool(*, size, size_swa, head_num, head_dim, swa_ids, full_ids, device):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        return SWAKVPool(
            size=size,
            size_swa=size_swa,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_ids,
            full_attention_layer_ids=full_ids,
            device=device,
            enable_memory_saver=False,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_fused_store_matches_set_kv_buffer_on_both_layer_kinds(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        dev = "cuda"
        # 8:1 GQA at head_dim 128 -- ZAYA1-74B at attn_tp=2.
        nq, nk, hd, T = 8, 1, 128, 6
        # Both sub-pools the same size so a *missed* full->SWA indirection would
        # write an in-range but wrong row (caught by the comparison) rather than
        # running off the end of a smaller SWA pool.
        size, size_swa = 64, 64
        # Layer 0 sliding, layer 2 full: the hybrid_layer_pattern _make_swa_config
        # produces for [4096, 0, 0, 0].
        swa_ids, full_ids = [0], [2]

        torch.manual_seed(13)
        rope = get_rope(
            head_size=hd,
            rotary_dim=hd,
            max_position=256,
            base=10_000,
            is_neox_style=True,
            partial_rotary_factor=0.5,
        ).to(dev)

        for layer_id in (0, 2):
            with self.subTest(layer=layer_id, sliding=layer_id in swa_ids):
                fused_pool = self._make_pool(
                    size=size,
                    size_swa=size_swa,
                    head_num=nk,
                    head_dim=hd,
                    swa_ids=swa_ids,
                    full_ids=full_ids,
                    device=dev,
                )
                ref_pool = self._make_pool(
                    size=size,
                    size_swa=size_swa,
                    head_num=nk,
                    head_dim=hd,
                    swa_ids=swa_ids,
                    full_ids=full_ids,
                    device=dev,
                )
                # Mirror SWATokenToKVPoolAllocator's mapping: one entry per full
                # slot plus the trailing -1 sentinel. A random PERMUTATION, so it
                # is injective (no two tokens race for one row) and non-identity
                # (an unapplied indirection cannot pass by luck).
                n_full = size + 1
                mapping = torch.empty(n_full + 1, dtype=torch.int64, device=dev)
                mapping[:n_full] = torch.randperm(n_full, device=dev)
                mapping[-1] = -1
                fused_pool.register_mapping(mapping)
                ref_pool.register_mapping(mapping)

                conv_qk = (
                    torch.randn(T, (nq + nk) * hd, dtype=torch.bfloat16, device=dev)
                    * 0.3
                )
                pre_q = torch.randn(T, nq * hd, dtype=torch.bfloat16, device=dev) * 0.3
                base_k = torch.randn(T, nk * hd, dtype=torch.bfloat16, device=dev) * 0.3
                value = torch.randn(T, nk, hd, dtype=torch.bfloat16, device=dev)
                k_scale = torch.rand(nk, device=dev) + 0.5
                positions = torch.arange(3, 3 + T, dtype=torch.int64, device=dev)
                # Spread the write locations so a "slot == token index" bug fails.
                out_cache_loc = torch.tensor(
                    [5, 1, 40, 17, 2, 63], dtype=torch.int64, device=dev
                )

                is_sliding = layer_id in swa_ids
                full_to_swa = mapping if is_sliding else None
                k_cache = fused_pool.get_key_buffer(layer_id)
                v_cache = fused_pool.get_value_buffer(layer_id)
                # ``conv_qk.device`` (cuda:0), NOT ``torch.device("cuda")``;
                # see the same note in TestCCAQKMixRope. Assert on the reason:
                # a silent decline means the pool comparison below never saw
                # the fused write at all.
                reason = kernel.store_decline_reason(
                    value,
                    k_cache,
                    v_cache,
                    out_cache_loc,
                    full_to_swa,
                    num_k_heads=nk,
                    head_dim=hd,
                    num_tokens=T,
                    out_dtype=torch.bfloat16,
                    device=conv_qk.device,
                )
                self.assertIsNone(reason, f"kv store fusion declined: {reason}")

                q_got, k_got = kernel.cca_qk_mix(
                    conv_qk,
                    pre_q,
                    base_k,
                    k_scale,
                    num_q_heads=nq,
                    num_k_heads=nk,
                    head_dim=hd,
                    q_scale=hd**0.5,
                    out_dtype=torch.bfloat16,
                    positions=positions,
                    cos_sin_cache=rope.cos_sin_cache,
                    rotary_dim=rope.rotary_dim,
                    value=value,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    out_cache_loc=out_cache_loc,
                    full_to_swa=full_to_swa,
                )

                # The reference: hand the SAME k/v to the real set_kv_buffer, via
                # the same KVWriteLoc bundle the attention backend builds.
                layer = RadixAttention(
                    num_heads=nq,
                    head_dim=hd,
                    scaling=hd**-0.5,
                    num_kv_heads=nk,
                    layer_id=layer_id,
                    sliding_window_size=4095 if is_sliding else -1,
                )
                swa_loc = (
                    ref_pool.translate_loc_from_full_to_swa(out_cache_loc)
                    if is_sliding
                    else None
                )
                ref_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(out_cache_loc, swa_loc),
                    k_got.contiguous(),
                    value.contiguous(),
                )

                torch.testing.assert_close(
                    fused_pool.get_key_buffer(layer_id),
                    ref_pool.get_key_buffer(layer_id),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    fused_pool.get_value_buffer(layer_id),
                    ref_pool.get_value_buffer(layer_id),
                    rtol=0,
                    atol=0,
                )
                # And the pool actually received something -- an all-zero buffer
                # would trivially match an all-zero reference.
                self.assertGreater(
                    fused_pool.get_key_buffer(layer_id).abs().sum().item(), 0.0
                )
                self.assertEqual(q_got.shape, (T, nq, hd))

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_negative_slots_are_skipped(self):
        # The full->SWA mapping's trailing -1 is how a padding row says "write
        # nothing". Writing it as slot -1 (or as slot 0) would corrupt a live
        # entry, so pin the skip.
        from sglang.kernels.ops.attention import cca_qk_mix as kernel
        from sglang.srt.layers.rotary_embedding import get_rope

        dev = "cuda"
        nq, nk, hd, T, rows = 8, 1, 128, 3, 8
        torch.manual_seed(4)
        rope = get_rope(
            head_size=hd,
            rotary_dim=hd,
            max_position=64,
            base=10_000,
            is_neox_style=True,
            partial_rotary_factor=0.5,
        ).to(dev)
        k_cache = torch.zeros(rows, nk, hd, dtype=torch.bfloat16, device=dev)
        v_cache = torch.zeros(rows, nk, hd, dtype=torch.bfloat16, device=dev)
        # Only token 1 has a live slot; tokens 0 and 2 map to the sentinel.
        mapping = torch.full((rows + 1,), -1, dtype=torch.int64, device=dev)
        mapping[1] = 4
        out_cache_loc = torch.tensor([0, 1, 2], dtype=torch.int64, device=dev)

        kernel.cca_qk_mix(
            torch.randn(T, (nq + nk) * hd, dtype=torch.bfloat16, device=dev),
            torch.randn(T, nq * hd, dtype=torch.bfloat16, device=dev),
            torch.randn(T, nk * hd, dtype=torch.bfloat16, device=dev),
            torch.ones(nk, device=dev),
            num_q_heads=nq,
            num_k_heads=nk,
            head_dim=hd,
            q_scale=hd**0.5,
            out_dtype=torch.bfloat16,
            positions=torch.arange(T, dtype=torch.int64, device=dev),
            cos_sin_cache=rope.cos_sin_cache,
            rotary_dim=rope.rotary_dim,
            value=torch.randn(T, nk, hd, dtype=torch.bfloat16, device=dev),
            k_cache=k_cache,
            v_cache=v_cache,
            out_cache_loc=out_cache_loc,
            full_to_swa=mapping,
        )
        written = (k_cache.abs().sum(dim=(1, 2)) > 0).nonzero().flatten().tolist()
        self.assertEqual(written, [4])
        written_v = (v_cache.abs().sum(dim=(1, 2)) > 0).nonzero().flatten().tolist()
        self.assertEqual(written_v, [4])


class TestCCAQKMixStoreCoverage(CustomTestCase):
    """``store_covered`` negatives.

    A rejected input costs one extra ``set_kv_buffer`` launch; an input that
    should have been rejected corrupts KV silently. The gate compares against an
    explicit caller device rather than ``is_cuda``, so every branch runs on CPU.
    """

    @staticmethod
    def _ok_args(nk=2, hd=8, T=3, rows=17, dtype=torch.bfloat16):
        return dict(
            value=torch.zeros(T, nk, hd, dtype=dtype),
            k_cache=torch.zeros(rows, nk, hd, dtype=dtype),
            v_cache=torch.zeros(rows, nk, hd, dtype=dtype),
            out_cache_loc=torch.zeros(T, dtype=torch.int64),
            full_to_swa=None,
        )

    def _covered(self, **over):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        args = self._ok_args()
        args.update(over)
        return kernel.store_covered(
            args["value"],
            args["k_cache"],
            args["v_cache"],
            args["out_cache_loc"],
            args["full_to_swa"],
            num_k_heads=2,
            head_dim=8,
            num_tokens=3,
            out_dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

    def test_the_baseline_is_covered(self):
        self.assertTrue(self._covered())

    def test_rejects_non_flash_kv_layouts(self):
        # The 5-D SHUFFLE (vectorized_5d) layout: (blocks, H, D//x, page, x).
        self.assertFalse(self._covered(k_cache=torch.zeros(4, 2, 4, 1, 2)))
        self.assertFalse(self._covered(v_cache=torch.zeros(4, 2, 1, 8, 2)))
        # The 4-D HND layout: (pages, H, page_size, D). Its per-slot row is not a
        # contiguous [H, D] block, so a flat slot index would land elsewhere.
        self.assertFalse(self._covered(k_cache=torch.zeros(17, 2, 1, 8)))
        # And the paged 4-D flash view a caller might reshape into.
        self.assertFalse(self._covered(k_cache=torch.zeros(17, 1, 2, 8)))

    def test_rejects_a_non_bf16_pool(self):
        # An fp8 pool stores under a different store_dtype and needs per-tensor
        # scales the kernel does not apply.
        fp8 = torch.zeros(17, 2, 8, dtype=torch.float8_e4m3fn)
        self.assertFalse(self._covered(k_cache=fp8))
        self.assertFalse(self._covered(v_cache=torch.zeros(17, 2, 8).float()))
        self.assertFalse(self._covered(value=torch.zeros(3, 2, 8).float()))

    def test_rejects_shape_and_stride_mismatches(self):
        # Wrong head count / head dim on either buffer.
        self.assertFalse(
            self._covered(k_cache=torch.zeros(17, 4, 8, dtype=torch.bfloat16))
        )
        self.assertFalse(
            self._covered(v_cache=torch.zeros(17, 2, 16, dtype=torch.bfloat16))
        )
        # A non-unit innermost stride (a transposed or strided view).
        strided = torch.zeros(17, 8, 2, dtype=torch.bfloat16).transpose(1, 2)
        self.assertFalse(self._covered(k_cache=strided))
        # V with the wrong token count.
        self.assertFalse(
            self._covered(value=torch.zeros(2, 2, 8, dtype=torch.bfloat16))
        )
        # A 2-D V (not yet reshaped into heads).
        self.assertFalse(self._covered(value=torch.zeros(3, 16, dtype=torch.bfloat16)))

    def test_rejects_bad_locs_and_mappings(self):
        # Float locs, the wrong length, a 2-D loc.
        self.assertFalse(self._covered(out_cache_loc=torch.zeros(3)))
        self.assertFalse(self._covered(out_cache_loc=torch.zeros(4, dtype=torch.int64)))
        self.assertFalse(
            self._covered(out_cache_loc=torch.zeros(3, 1, dtype=torch.int64))
        )
        # A non-int64 or empty full->SWA mapping.
        self.assertFalse(self._covered(full_to_swa=torch.zeros(18, dtype=torch.int32)))
        self.assertFalse(self._covered(full_to_swa=torch.zeros(0, dtype=torch.int64)))
        # A well-formed one is fine.
        self.assertTrue(self._covered(full_to_swa=torch.zeros(18, dtype=torch.int64)))

    def test_rejects_missing_arguments(self):
        for missing in ("value", "k_cache", "v_cache", "out_cache_loc"):
            with self.subTest(missing=missing):
                self.assertFalse(self._covered(**{missing: None}))

    def test_rejects_a_cross_device_buffer(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        args = self._ok_args()
        # A "meta" tensor stands in for a buffer that belongs to another device;
        # the check is device equality, so it fires without needing two real ones.
        self.assertFalse(
            kernel.store_covered(
                args["value"],
                args["k_cache"].to("meta"),
                args["v_cache"],
                args["out_cache_loc"],
                None,
                num_k_heads=2,
                head_dim=8,
                num_tokens=3,
                out_dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
        )

    def test_an_unindexed_reference_device_still_matches(self):
        # Regression. ``torch.device("cuda") != torch.device("cuda:0")``, so a
        # caller writing the reference device by hand used to fail every check --
        # and since this gate declines by FALLING BACK, that read as "the fusion
        # is silently off", not as an error. An un-indexed reference means "any
        # device of this type" and must match.
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        t = torch.zeros(2)
        self.assertTrue(kernel._same_device(t, torch.device("cpu")))
        self.assertFalse(kernel._same_device(t, torch.device("cuda")))
        self.assertFalse(kernel._same_device(t, torch.device("cuda:0")))
        self.assertFalse(kernel._same_device(t, torch.device("meta")))
        # An indexed reference still pins the index.
        meta = t.to("meta")
        self.assertTrue(kernel._same_device(meta, torch.device("meta")))

    def test_the_decline_reason_names_the_gate(self):
        # The reason strings are what ``_note_fusion`` logs and what the GPU
        # tests assert on, so a bare "False" can never again hide which gate
        # said no. Each case must name its own gate, not a neighbour's.
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        args = self._ok_args()

        def _reason(**over):
            a = dict(args)
            a.update(over)
            return kernel.store_decline_reason(
                a["value"],
                a["k_cache"],
                a["v_cache"],
                a["out_cache_loc"],
                a["full_to_swa"],
                num_k_heads=2,
                head_dim=8,
                num_tokens=3,
                out_dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )

        self.assertIsNone(_reason())
        self.assertIn("not supplied", _reason(value=None))
        self.assertIn("3-D NHD", _reason(k_cache=torch.zeros(4, 2, 4, 1, 2)))
        self.assertIn("dtype", _reason(v_cache=torch.zeros(17, 2, 8).float()))
        self.assertIn(
            "out_cache_loc", _reason(out_cache_loc=torch.zeros(4, dtype=torch.int64))
        )
        self.assertIn(
            "full_to_swa", _reason(full_to_swa=torch.zeros(18, dtype=torch.int32))
        )
        self.assertIn("k_cache on meta", _reason(k_cache=args["k_cache"].to("meta")))

    def test_the_rope_decline_reason_names_the_gate(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        hd, rot, T = 128, 64, 4
        pos = torch.arange(T, dtype=torch.int64)
        cache = torch.randn(256, rot)
        common = dict(head_dim=hd, num_tokens=T)

        self.assertIn(
            "no rotary offered", kernel.rope_decline_reason(None, cache, rot, **common)
        )
        self.assertIn(
            "rotary_dim", kernel.rope_decline_reason(pos, cache, 128, **common)
        )
        self.assertIn(
            "gptj",
            kernel.rope_decline_reason(pos, cache, rot, is_neox_style=False, **common),
        )
        self.assertIn(
            "positions", kernel.rope_decline_reason(pos[:2], cache, rot, **common)
        )
        # On CPU the accelerator check is the last one standing, and it must say
        # so rather than blaming a shape.
        self.assertIn(
            "accelerator", kernel.rope_decline_reason(pos, cache, rot, **common)
        )

    def test_store_is_only_offered_when_the_rope_fused(self):
        # Storing an un-rotated k would be silent KV corruption, and the rotation
        # happens inside the same kernel -- so the store must never be enabled
        # while the rope fell back. Pinned on the CPU fallback, where neither
        # fuses.
        _ensure_dist_initialized()
        cca = _make_tiny_cca(seed=6)[0]
        T = 3
        hd, nq, nk = cca.head_dim, cca.num_q_heads, cca.num_k_heads
        qk_out = torch.randn(T, (nq + nk) * hd)
        q_raw = torch.randn(T, nq * hd)
        k_raw = torch.randn(T, nk * hd)
        from sglang.srt.models.zaya import CCAKVStore

        store = CCAKVStore(
            k_cache=torch.zeros(8, nk, hd),
            v_cache=torch.zeros(8, nk, hd),
            out_cache_loc=torch.arange(T, dtype=torch.int64),
            full_to_swa=None,
        )
        cca._mix_and_normalize_qk(
            qk_out,
            q_raw,
            k_raw,
            qk_out[:, : nq * hd].view(T, nq, hd),
            qk_out[:, nq * hd :].view(T, nk, hd),
            q_raw.view(T, nq, hd),
            k_raw.view(T, nk, hd),
            out_dtype=torch.float32,
            rope=None,
            value=torch.randn(T, nk, hd),
            kv_store=store,
        )
        self.assertFalse(cca.rope_fused)
        self.assertFalse(cca.kv_store_fused)
        # Nothing was written into the stand-in pool.
        self.assertEqual(store.k_cache.abs().sum().item(), 0.0)


class TestZayaAttentionKVStoreGate(CustomTestCase):
    """``ZayaAttention._kv_store`` must decline every pool it cannot address.

    The resolver is pure Python over the live pool objects, so its rejects are
    checkable on CPU with stand-ins. An unrecognized pool type is refused before
    any tensor is touched.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        from sglang.srt.runtime_context import get_context

        cls._server_args_override = get_context().override_server_args(
            model_path="dummy"
        )
        cls._server_args_override.install()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server_args_override.restore()

    def _attention(self, layer_id=0):
        from sglang.srt.models.zaya import ZayaAttention

        config = _make_swa_config(num_hidden_layers=4, swa_layers=[4096, 0, 0, 0])
        return ZayaAttention(config=config, layer_id=layer_id)

    def test_unknown_pool_types_decline(self):
        import sglang.srt.models.zaya as zaya_mod

        attn = self._attention()
        fb = SimpleNamespace(out_cache_loc=torch.zeros(2, dtype=torch.int64))

        class _SomeOtherPool:
            pass

        with unittest.mock.patch.object(
            zaya_mod, "get_token_to_kv_pool", lambda: _SomeOtherPool()
        ):
            self.assertIsNone(attn._kv_store(fb))

    def test_missing_out_cache_loc_declines(self):
        attn = self._attention()
        self.assertIsNone(attn._kv_store(SimpleNamespace(out_cache_loc=None)))

    def test_a_scaled_attention_layer_declines(self):
        import sglang.srt.models.zaya as zaya_mod

        attn = self._attention()
        attn.attn.k_scale = torch.tensor(1.0)
        fb = SimpleNamespace(out_cache_loc=torch.zeros(2, dtype=torch.int64))

        called = []

        def _boom():
            called.append(1)
            raise AssertionError("must not reach the pool")

        with unittest.mock.patch.object(zaya_mod, "get_token_to_kv_pool", _boom):
            self.assertIsNone(attn._kv_store(fb))
        self.assertEqual(called, [])

    def test_a_dcp_masked_batch_declines(self):
        import sglang.srt.models.zaya as zaya_mod

        attn = self._attention()
        fb = SimpleNamespace(
            out_cache_loc=torch.zeros(2, dtype=torch.int64),
            dcp_kv_mask=torch.ones(2, dtype=torch.bool),
        )
        with unittest.mock.patch.object(zaya_mod, "get_token_to_kv_pool", lambda: None):
            self.assertIsNone(attn._kv_store(fb))


class TestCCADecodeConvFold(CustomTestCase):
    """``CCA.fold_decode_conv`` must reproduce the two-stage conv exactly.

    At decode only one output timestep is needed, so conv_qk[0] composed with
    conv_qk[1] collapses to a single grouped matmul whose weight is
    precomputable. The tap-index bookkeeping
    (``A[..., j+k] += w1[..., j] * w0[..., k]``) and the depthwise bias
    pass-through are easy to get subtly wrong -- an off-by-one in the tap offset
    still produces plausible output -- so pin the identity directly.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def _check(self, cca, T: int):
        from sglang.srt.models.zaya import _cca_decode_conv

        taps = cca.total_padding + 1
        torch.manual_seed(7)
        window = torch.randn(T, cca.in_out_ch, taps, dtype=torch.float32) * 0.3
        with torch.no_grad():
            # Reference: the real two-stage conv, which yields exactly one step.
            ref = cca.conv_qk(window)
            self.assertEqual(ref.shape, (T, cca.in_out_ch, 1))
            ref = ref.squeeze(-1)

            cca.fold_decode_conv()
            # Drive the production consumer rather than re-deriving the einsum:
            # the folded weight now carries the conv bias in a trailing column
            # activated by a constant-1.0 tap, and re-deriving it here would let
            # the two layouts drift apart unnoticed.
            got = _cca_decode_conv(
                window,
                cca.conv_qk,
                cca.decode_conv_weight,
                cca.decode_conv_bias,
                cca.decode_conv_groups,
            )
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

        # The window that ``cca_state_step`` emits already carries the ones
        # column; the one built by the unfused fallback does not and gets it
        # appended. Both must give the same answer, or the fused and fallback
        # decode paths disagree only on GPU, where the fused one runs.
        with torch.no_grad():
            widened = torch.cat([window, torch.ones(T, cca.in_out_ch, 1)], dim=-1)
            got_widened = _cca_decode_conv(
                widened,
                cca.conv_qk,
                cca.decode_conv_weight,
                cca.decode_conv_bias,
                cca.decode_conv_groups,
            )
        torch.testing.assert_close(got_widened, got, rtol=0, atol=0)

    def test_folded_weight_carries_the_bias_column(self):
        """The bias rides in a trailing weight column, activated by a
        constant-1.0 tap, so it lands in the matmul's fp32 accumulator instead of
        a separate add. Pin both halves: the column holds the bias on the last
        input channel and zero on every other, so each output picks it up exactly
        once.
        """
        cca, _ = _make_tiny_cca(seed=9)
        cca.fold_decode_conv()
        groups = cca.decode_conv_groups
        cg = cca.in_out_ch // groups
        taps = cca.total_padding + 1

        self.assertEqual(cca.decode_conv_taps_ext, taps + 1)
        self.assertEqual(
            tuple(cca.decode_conv_weight.shape), (groups, cg, cg * (taps + 1))
        )
        w = cca.decode_conv_weight.reshape(groups, cg, cg, taps + 1)
        torch.testing.assert_close(
            w[:, :, cg - 1, taps], cca.decode_conv_bias, rtol=0, atol=0
        )
        self.assertTrue(torch.all(w[:, :, : cg - 1, taps] == 0))

    def test_fold_matches_two_stage_conv(self):
        cca, _ = _make_tiny_cca(seed=3)
        for T in (1, 4):
            self._check(cca, T)

    def test_fold_matches_under_tensor_parallel_slicing(self):
        # conv_qk is TP-sliced per rank, so the fold must be built from the live
        # per-rank weights -- a rank that folded the full-width weight would
        # silently mix in another rank's heads.
        for rank in (0, 1):
            cca, _ = _make_tiny_cca(seed=4, tp_rank=rank, tp_size=2)
            self._check(cca, T=2)

    def test_unfolded_cca_does_not_use_the_zero_buffers(self):
        # The folded buffers are zero-initialized and only valid after
        # fold_decode_conv() has run against loaded weights, so the flag that
        # gates them must start False. NOTE: this asserts the flag only -- it
        # runs no forward, so it does not cover the bias-only output a forward
        # that consumed the zero buffers would produce.
        cca, _ = _make_tiny_cca(seed=6)
        self.assertFalse(cca._decode_conv_folded)
        cca.fold_decode_conv()
        self.assertTrue(cca._decode_conv_folded)

    def test_grouped_matmul_is_bitwise_the_einsum_it_replaces(self):
        """The folded conv runs as an explicit ``bmm`` into a strided ``out=``
        view instead of ``einsum("tgk,gok->tgo")``.

        The motive is purely the output layout -- einsum batches over ``g``, so
        its gemm writes ``[G, T, Cg]`` and the permute back to token-major costs
        a second launch. Pin the two bit-identical: anything short of that means
        the rewrite changed the arithmetic, not just where the result lands.
        """
        from sglang.srt.models.zaya import _cca_decode_conv

        cca, _ = _make_tiny_cca(seed=11)
        cca.fold_decode_conv()
        T = 3
        torch.manual_seed(11)
        # Already ``taps_ext`` wide, i.e. the ones column ``cca_state_step``
        # writes is present, so no padding step stands between the two forms.
        window = (
            torch.randn(T, cca.in_out_ch, cca.decode_conv_taps_ext, dtype=torch.float32)
            * 0.3
        )
        with torch.no_grad():
            grouped = window.reshape(T, cca.decode_conv_groups, -1)
            ref = torch.einsum("tgk,gok->tgo", grouped, cca.decode_conv_weight).reshape(
                T, -1
            )
            got = _cca_decode_conv(
                window,
                cca.conv_qk,
                cca.decode_conv_weight,
                cca.decode_conv_bias,
                cca.decode_conv_groups,
            )
        self.assertEqual(got.shape, ref.shape)
        self.assertTrue(got.is_contiguous())
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_fold_is_refreshed_not_stale(self):
        # The buffers are non-persistent and derived, so a weight change must be
        # picked up by re-folding; a cached-once implementation would go stale on
        # weight reload (the RL / update_weights path).
        cca, _ = _make_tiny_cca(seed=5)
        cca.fold_decode_conv()
        before = cca.decode_conv_weight.clone()
        with torch.no_grad():
            cca.conv_qk[0].weight.mul_(2.0)
        cca.fold_decode_conv()
        self.assertFalse(torch.allclose(before, cca.decode_conv_weight))
        self._check(cca, T=1)


class TestShortConvPaddingSlotClamp(CustomTestCase):
    """``ShortConvAttnBackend`` must clamp the -1 batch-padding sentinel.

    Batch padding poisons unused rows' mamba slot ids to -1, and CCA feeds the
    shared index view straight into ``index_select`` / ``index_copy_``. A
    negative index there is an out-of-bounds device gather, which on ROCm aborts
    the queue with HSA_STATUS_ERROR_EXCEPTION 0x1016 instead of raising. Clamping
    to 0 is safe because ``MambaSlotAllocator`` reserves slot 0.
    """

    @staticmethod
    def _bare_backend(buf: Optional[torch.Tensor], idx: Optional[torch.Tensor]):
        # _refresh_cache_indices touches only these three attributes, so a bare
        # instance exercises the clamping contract without a live ModelRunner.
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend._cache_indices_buf = buf
        backend._cache_indices = None
        backend.forward_metadata = (
            None if idx is None else SimpleNamespace(mamba_cache_indices=idx)
        )
        return backend

    def test_graph_buffer_path_clamps_padding(self):
        # Buffered path (cuda/cpu graph): the persistent buffer is refilled in
        # place, so the clamp must land in the buffer the captured graph reads.
        buf = torch.empty(8, dtype=torch.int64)
        idx = torch.tensor([3, 5, -1, -1], dtype=torch.int64)
        backend = self._bare_backend(buf, idx)
        backend._refresh_cache_indices()

        out = backend._cache_indices
        self.assertEqual(out.tolist(), [3, 5, 0, 0])
        self.assertGreaterEqual(int(out.min()), 0)
        # Must be a view of the persistent buffer, not a fresh allocation.
        self.assertIs(out.untyped_storage(), buf.untyped_storage())

    def test_fallback_path_clamps_without_mutating_source(self):
        # No buffer (eager, or bs beyond the buffer): the clamp must be
        # out-of-place. ``to(torch.long)`` aliases an already-int64 tensor, so an
        # in-place clamp would corrupt the backend's own mamba_cache_indices --
        # destroying the -1 sentinel other consumers rely on to skip padded rows.
        idx = torch.tensor([2, -1], dtype=torch.int64)
        backend = self._bare_backend(None, idx)
        backend._refresh_cache_indices()

        self.assertEqual(backend._cache_indices.tolist(), [2, 0])
        self.assertEqual(idx.tolist(), [2, -1])

    def test_no_metadata_yields_no_indices(self):
        # Before the first step forward_metadata is None; the view must stay None
        # rather than clamping a nonexistent tensor.
        backend = self._bare_backend(torch.empty(4, dtype=torch.int64), None)
        backend._refresh_cache_indices()
        self.assertIsNone(backend._cache_indices)


# ---------------------------------------------------------------------------
# Mamba radix cache -- extra_buffer track snapshot
# ---------------------------------------------------------------------------


def _torch_track_reference(
    state_a,
    state_b,
    src_rows,
    mask_rows,
    dst_rows,
    total_rows,
    check_freed_slots=False,
):
    """CPU stand-in for ``track_mamba_states_if_needed`` (Triton, CUDA only).

    Same contract: for every row ``i < total_rows`` whose mask is set, copy
    ``state[src_rows[i]] -> state[dst_rows[i]]`` in BOTH state tensors, and
    skip rows whose src/dst is negative when ``check_freed_slots``.
    """
    for i in range(total_rows):
        if not bool(mask_rows[i]):
            continue
        src = int(src_rows[i])
        dst = int(dst_rows[i])
        if check_freed_slots and (src < 0 or dst < 0):
            continue
        for state in (state_a, state_b):
            if state[0].numel() == 0:
                continue
            state[dst] = state[src].clone()


class _TrackHarness:
    """Bare ``ShortConvAttnBackend`` wired for the track paths, CPU only.

    ``object.__new__`` keeps this free of a live ModelRunner: the track code
    reads only the attributes set here plus ``mamba_cache_chunk_size`` off the
    server args, which the tests patch on the module.
    """

    def __init__(
        self,
        *,
        num_layers=2,
        num_slots=6,
        num_channels=3,
        hidden_size=4,
        windows=(2, 1),
        extra_buffer=True,
        chunk_size=8,
        track_interval=16,
    ):
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        conv = [
            torch.zeros(num_layers, num_slots, ch, w)
            for ch, w in zip((num_channels, hidden_size), windows)
        ]
        temporal = torch.zeros(num_layers, num_slots, 1, 1, 0)
        self.mamba_cache = SimpleNamespace(conv=conv, temporal=temporal)
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.server_args = SimpleNamespace(
            mamba_cache_chunk_size=chunk_size,
            mamba_track_interval=track_interval,
            speculative_algorithm=None,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend.device = torch.device("cpu")
        backend.enable_unified_memory = False
        backend.conv_states_shape = conv[0].shape
        backend.conv_window_lens = [int(c.shape[-1]) for c in conv]
        backend._cache_indices = None
        backend._cache_indices_buf = None
        backend.forward_metadata = None
        backend._track_conv_indices = None
        backend._track_dst = None
        backend._track_layer_row_base = None
        backend._track_pairs = None
        backend.enable_mamba_extra_buffer = extra_buffer
        if extra_buffer:
            backend._init_track_state(self.server_args, self.mamba_cache)
        self.backend = backend

    def layer_cache(self, layer_idx):
        return _MockLayerCache(
            conv=[c[layer_idx] for c in self.mamba_cache.conv],
            temporal=self.mamba_cache.temporal[layer_idx],
        )


def _track_forward_batch(*, extend_seq_lens, prefix_lens, track_mask, track_indices):
    """Extend ForwardBatch stub carrying exactly the track inputs.

    ``mamba_track_seqlens`` mirrors what the scheduler builds in
    ``_mamba_radix_cache_v2_req_prepare_for_extend``: prefix + extend length
    for tracked rows, -1 for untracked ones.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    seqlens = [
        (p + e) if m else -1
        for p, e, m in zip(prefix_lens, extend_seq_lens, track_mask)
    ]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        mamba_track_seqlens=torch.tensor(seqlens, dtype=torch.int64),
        mamba_track_mask=torch.tensor(track_mask, dtype=torch.bool),
        mamba_track_indices=torch.tensor(track_indices, dtype=torch.int64),
    )


def _query_start_loc(extend_seq_lens):
    out = [0]
    for s in extend_seq_lens:
        out.append(out[-1] + s)
    return torch.tensor(out, dtype=torch.int64)


# TestShortConvTrackIndices / TrackExtendSnapshot / TrackDecode live in
# test/registered/unit/layers/attention/test_short_conv_track.py: they pin the
# backend's index arithmetic and need no model. The harness above stays here
# because the ZAYA1-specific cases below drive it against a real CCA.


class TestShortConvTrackWidthContract(CustomTestCase):
    """The snapshot must cache exactly what the conv state holds.

    ``conv[1]`` holds the PROJECTED ``val_proj2`` value, not the raw hidden
    state, so an extend snapshot handing ``hidden_states`` is both the wrong
    width and the wrong quantity -- and had the widths matched, a prefix restore
    would reload a value the decode path never writes. What pins it is that the
    model hands the backend the state VIEW it wrote through, and the backend
    checks that view against the input it gathers from.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def _cca_and_pool(self, tp_size: int, tp_rank: int):
        config = _make_tiny_config()
        cca, _ = _make_tiny_cca(
            seed=0,
            tp_rank=tp_rank if tp_size > 1 else None,
            tp_size=tp_size if tp_size > 1 else None,
            config=config,
        )
        pool = _MockReqToTokenPool(pool_size=4, cca_config=config, tp_size=tp_size)
        return cca, pool, config

    def _snapshot_pairs(self, cca, pool, num_tokens=6):
        """Exactly what CCA.forward hands track_conv_states_extend."""
        layer_cache = pool.mamba2_layer_cache(cca.layer_id)
        hidden_states = torch.randn(num_tokens, cca.hidden_size)
        qk, _ = cca.linear_qk(hidden_states)
        conv_state = layer_cache.conv[0]
        lag_state = cca._lag_state(layer_cache.conv[1])
        lag_now = cca._lag_now(hidden_states)
        return layer_cache, (conv_state, lag_state), (qk, lag_now), hidden_states

    def test_state_and_input_widths_agree_at_every_tp(self):
        # The assertion that would have caught the break at merge: for BOTH
        # conv entries, the channel axis of the state the conv wrote equals the
        # feature width of the stream the snapshot gathers from.
        for tp_size, tp_rank in ((1, 0), (2, 0), (2, 1)):
            with self.subTest(tp_size=tp_size, tp_rank=tp_rank):
                cca, pool, config = self._cca_and_pool(tp_size, tp_rank)
                _, states, inputs, _ = self._snapshot_pairs(cca, pool)
                for entry, (state, x) in enumerate(zip(states, inputs)):
                    if state is None or x is None:
                        # Both sides must vanish together, or the pair is a
                        # half-tracked state.
                        self.assertIsNone(state, f"entry {entry}")
                        self.assertIsNone(x, f"entry {entry}")
                        continue
                    self.assertEqual(
                        state.shape[-2],
                        x.shape[-1],
                        f"tp{tp_size} rank{tp_rank} conv[{entry}] width",
                    )

    def test_pool_entry_widths_match_the_config(self):
        # The pool side of the same contract: conv[0] shards with the rank,
        # conv[1] is rank-uniform at cca_v2_state_dim.
        for tp_size in (1, 2):
            with self.subTest(tp_size=tp_size):
                cca, pool, config = self._cca_and_pool(tp_size, 0)
                layer_cache = pool.mamba2_layer_cache(cca.layer_id)
                in_out_ch_full = (
                    config.num_attention_heads + config.num_key_value_heads
                ) * config.head_dim
                self.assertEqual(
                    layer_cache.conv[0].shape[-2], in_out_ch_full // tp_size
                )
                self.assertEqual(layer_cache.conv[1].shape[-2], config.cca_v2_state_dim)
                self.assertLess(
                    config.cca_v2_state_dim,
                    config.hidden_size,
                    "the projected lag must be narrower than the raw hidden "
                    "state, or the narrowing did not take effect",
                )

    def test_rank_lag_view_is_a_leading_subslice_of_the_pool_entry(self):
        # A rank may own only part of val_proj2's output. The snapshot has to
        # land in the same sub-slice the conv wrote, which is why the model
        # passes the view rather than the pool entry.
        cca, pool, _ = self._cca_and_pool(tp_size=2, tp_rank=1)
        entry = pool.mamba2_layer_cache(cca.layer_id).conv[1]
        lag_state = cca._lag_state(entry)
        if lag_state is None:
            self.skipTest("this rank owns no lag stream")
        self.assertLessEqual(lag_state.shape[-2], entry.shape[-2])
        self.assertIs(lag_state.untyped_storage(), entry.untyped_storage())

    def test_snapshot_writes_the_projected_lag_not_the_hidden_state(self):
        # End to end through the backend: the tracked lag row must equal the
        # projected value the state itself carries, not the raw hidden state.
        cca, pool, config = self._cca_and_pool(tp_size=1, tp_rank=0)
        layer_cache, states, inputs, hidden_states = self._snapshot_pairs(
            cca, pool, num_tokens=20
        )
        conv_state, lag_state = states
        qk, lag_now = inputs
        if lag_state is None:
            self.skipTest("this rank owns no lag stream")

        harness = _TrackHarness(
            windows=(conv_state.shape[-1], lag_state.shape[-1]),
            num_channels=conv_state.shape[-2],
            hidden_size=lag_state.shape[-2],
            chunk_size=8,
            track_interval=256,
        )
        fb = _track_forward_batch(
            extend_seq_lens=[20],
            prefix_lens=[0],
            track_mask=[True],
            track_indices=[4],
        )
        from sglang.srt.layers.attention.linear import short_conv_backend

        with unittest.mock.patch.object(
            short_conv_backend,
            "mamba_cache_chunk_size",
            lambda: harness.server_args.mamba_cache_chunk_size,
        ):
            harness.backend._track_conv_indices = (
                harness.backend._init_track_conv_indices(_query_start_loc([20]), fb)
            )
        harness.backend._track_dst = fb.mamba_track_indices[fb.mamba_track_mask]
        harness.backend.track_conv_states_extend((conv_state, lag_state), (qk, lag_now))

        aligned = 16
        self.assertTrue(
            torch.allclose(lag_state[4], lag_now[aligned - 1].unsqueeze(-1))
        )
        self.assertFalse(
            torch.allclose(
                lag_state[4],
                hidden_states[aligned - 1, : lag_state.shape[-2]].unsqueeze(-1),
            ),
            "the snapshot cached the raw hidden state, not the projection",
        )

    def test_mismatched_width_is_rejected_loudly(self):
        # Handing the old quantity (hidden_states) must fail at the assert, not
        # silently scatter or broadcast.
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=[20],
            prefix_lens=[0],
            track_mask=[True],
            track_indices=[4],
        )
        from sglang.srt.layers.attention.linear import short_conv_backend

        with unittest.mock.patch.object(
            short_conv_backend,
            "mamba_cache_chunk_size",
            lambda: harness.server_args.mamba_cache_chunk_size,
        ):
            harness.backend._track_conv_indices = (
                harness.backend._init_track_conv_indices(_query_start_loc([20]), fb)
            )
        harness.backend._track_dst = fb.mamba_track_indices[fb.mamba_track_mask]
        layer_cache = harness.layer_cache(0)
        too_wide = torch.randn(20, layer_cache.conv[1].shape[-2] * 4)
        with self.assertRaises(AssertionError) as caught:
            harness.backend.track_conv_states_extend(
                tuple(layer_cache.conv), (torch.randn(20, 3), too_wide)
            )
        self.assertIn("channels", str(caught.exception))

    def test_absent_lag_entry_is_skipped(self):
        # attn_tp=2 rank 0 reads only val_proj1: no lag stream, no pool write,
        # nothing to snapshot. The hook must skip the entry, not crash and not
        # write a zero-width row.
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=[20],
            prefix_lens=[0],
            track_mask=[True],
            track_indices=[4],
        )
        from sglang.srt.layers.attention.linear import short_conv_backend

        with unittest.mock.patch.object(
            short_conv_backend,
            "mamba_cache_chunk_size",
            lambda: harness.server_args.mamba_cache_chunk_size,
        ):
            harness.backend._track_conv_indices = (
                harness.backend._init_track_conv_indices(_query_start_loc([20]), fb)
            )
        harness.backend._track_dst = fb.mamba_track_indices[fb.mamba_track_mask]
        layer_cache = harness.layer_cache(0)
        qk = torch.randn(20, 3)
        harness.backend.track_conv_states_extend(
            (layer_cache.conv[0], None), (qk, None)
        )
        self.assertTrue(
            torch.allclose(layer_cache.conv[0][4], qk[14:16].transpose(0, 1))
        )
        self.assertEqual(float(harness.mamba_cache.conv[1].abs().sum()), 0.0)

    def test_decode_row_copy_is_width_agnostic(self):
        # The decode side never sees a model tensor: it copies whole pool rows,
        # so its widths come from the pool and the conv[1] narrowing could not
        # desync it. Pin that by giving the two entries different widths and
        # checking both are copied whole.
        harness = _TrackHarness(
            num_layers=2, num_slots=6, num_channels=7, hidden_size=3, windows=(2, 1)
        )
        conv0, conv1 = harness.mamba_cache.conv
        self.assertNotEqual(conv0.shape[-2], conv1.shape[-2])
        torch.manual_seed(9)
        conv0.normal_()
        conv1.normal_()
        before0, before1 = conv0.clone(), conv1.clone()

        from sglang.srt.layers.attention.linear import short_conv_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = harness.backend
        backend._cache_indices = torch.tensor([1], dtype=torch.int64)
        backend.forward_metadata = SimpleNamespace(
            mamba_track_indices=torch.tensor([4], dtype=torch.int64),
            mamba_cache_indices=torch.tensor([1], dtype=torch.int64),
        )
        with unittest.mock.patch.object(
            short_conv_backend,
            "track_mamba_states_if_needed",
            _torch_track_reference,
        ):
            backend.track_conv_states_decode(
                SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    mamba_track_mask=torch.tensor([True], dtype=torch.bool),
                )
            )
        for layer in range(2):
            self.assertTrue(torch.equal(conv0[layer, 4], before0[layer, 1]))
            self.assertTrue(torch.equal(conv1[layer, 4], before1[layer, 1]))


class TestShortConvNoBufferUnchanged(CustomTestCase):
    """``no_buffer`` must be exactly what it was before extra_buffer existed."""

    def test_no_track_state_is_built(self):
        harness = _TrackHarness(extra_buffer=False)
        self.assertIsNone(harness.backend._track_pairs)
        self.assertIsNone(harness.backend._track_layer_row_base)

    def test_both_track_entrypoints_are_inert(self):
        from sglang.srt.layers.attention.linear import short_conv_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        harness = _TrackHarness(extra_buffer=False)
        record = []
        harness.backend._cache_indices = torch.tensor([1], dtype=torch.int64)
        harness.backend.forward_metadata = SimpleNamespace(
            mamba_track_indices=torch.tensor([4], dtype=torch.int64)
        )
        with unittest.mock.patch.object(
            short_conv_backend,
            "track_mamba_states_if_needed",
            lambda *a, **k: record.append(a),
        ):
            harness.backend.track_conv_states_decode(
                SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    mamba_track_mask=torch.tensor([True], dtype=torch.bool),
                )
            )
        harness.backend.track_conv_states_extend(
            tuple(harness.layer_cache(0).conv),
            (torch.randn(4, 3), torch.randn(4, 4)),
        )
        self.assertEqual(record, [])
        self.assertEqual(float(harness.mamba_cache.conv[0].abs().sum()), 0.0)
        self.assertEqual(float(harness.mamba_cache.conv[1].abs().sum()), 0.0)

    def test_zaya_resolves_to_extra_buffer_only_under_auto(self):
        # The arch is now on the allowlist, but an explicit strategy still
        # wins and `auto` without overlap/paging still lands on no_buffer.
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _mamba_radix_cache_resolution,
        )

        def _view(**kw):
            defaults = dict(
                disable_radix_cache=False,
                mamba_radix_cache_strategy="auto",
                disable_overlap_schedule=False,
                page_size=None,
                linear_attn_backend="triton",
            )
            defaults.update(kw)
            hf = SimpleNamespace(architectures=["ZayaForCausalLM"])
            return ResolvedView(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(hf_config=hf),
                    # model_config_of hands a fixture-supplied config back
                    # as-is (no _model_config_built_from key), so resolution
                    # never builds a real ModelConfig off this fake record.
                    _model_config=SimpleNamespace(hf_config=hf, is_hybrid_swa=False),
                    **defaults,
                )
            )

        self.assertEqual(
            _mamba_radix_cache_resolution(_view()),
            {
                "uses_mamba_radix_cache": True,
                "mamba_radix_cache_strategy": "extra_buffer",
            },
        )
        declared = _mamba_radix_cache_resolution(
            _view(disable_overlap_schedule=True, page_size=1)
        )
        self.assertEqual(declared["mamba_radix_cache_strategy"], "no_buffer")
        self.assertIs(declared["disable_overlap_schedule"], True)
        pinned = _mamba_radix_cache_resolution(
            _view(mamba_radix_cache_strategy="no_buffer")
        )
        self.assertEqual(pinned, {"uses_mamba_radix_cache": True})

    def test_lfm2_is_deliberately_left_on_no_buffer(self):
        from sglang.srt.arg_groups.overrides import _MAMBA_EXTRA_BUFFER_ARCHS

        self.assertIn("ZayaForCausalLM", _MAMBA_EXTRA_BUFFER_ARCHS)
        self.assertNotIn("Lfm2ForCausalLM", _MAMBA_EXTRA_BUFFER_ARCHS)

    def test_ssm_backends_keep_their_temporal_track(self):
        # has_temporal_state is the only base-class behaviour change; every
        # SSM backend must still build its SSM track indices.
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            Mamba2AttnBackend,
            MambaAttnBackendBase,
        )
        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        self.assertTrue(MambaAttnBackendBase.has_temporal_state)
        self.assertTrue(Mamba2AttnBackend.has_temporal_state)
        self.assertTrue(GDNAttnBackend.has_temporal_state)
        self.assertFalse(ShortConvAttnBackend.has_temporal_state)


class TestShortConvTrackStateGuards(CustomTestCase):
    """``_init_track_state`` refuses configurations it cannot snapshot."""

    def test_window_must_fit_inside_a_chunk(self):
        # The gather reaches `window` rows back from the chunk-aligned point;
        # a window >= chunk_size would have to read the cached prefix, which
        # is not in the current extend's token buffer at all.
        with self.assertRaises(AssertionError):
            _TrackHarness(chunk_size=2, windows=(2, 1))

    def test_track_interval_must_cover_a_chunk(self):
        with self.assertRaises(AssertionError):
            _TrackHarness(chunk_size=8, track_interval=4, windows=(2, 1))

    def test_single_conv_entry_pairs_with_the_empty_temporal(self):
        # LFM2 shape: one conv entry. The shared scatter takes two state
        # tensors per launch, so the zero-element temporal rides along.
        harness = _TrackHarness(windows=(3,))
        self.assertEqual(len(harness.backend._track_pairs), 1)
        state_a, state_b = harness.backend._track_pairs[0]
        self.assertEqual(state_a.shape[0], harness.num_layers * harness.num_slots)
        self.assertEqual(state_b[0].numel(), 0)

    def test_two_conv_entries_ride_one_launch(self):
        harness = _TrackHarness(windows=(2, 1))
        self.assertEqual(len(harness.backend._track_pairs), 1)
        state_a, state_b = harness.backend._track_pairs[0]
        self.assertEqual(state_a.shape[-1], 2)
        self.assertEqual(state_b.shape[-1], 1)

    def test_flattened_views_alias_the_pool(self):
        # The snapshot must land in the live pool, not a copy of it.
        harness = _TrackHarness(windows=(2, 1))
        state_a, state_b = harness.backend._track_pairs[0]
        self.assertIs(
            state_a.untyped_storage(),
            harness.mamba_cache.conv[0].untyped_storage(),
        )
        self.assertIs(
            state_b.untyped_storage(),
            harness.mamba_cache.conv[1].untyped_storage(),
        )

    def _extra_buffer_view(self, **over):
        view = SimpleNamespace(
            linear_attn_backend="triton",
            mamba_radix_cache_strategy="extra_buffer",
            speculative_num_draft_tokens=None,
            page_size=None,
            cuda_graph_backend_prefill=None,
            speculative_algorithm=None,
        )
        for k, v in over.items():
            setattr(view, k, v)
        return view

    def _validate(self, view, arch="ZayaForCausalLM"):
        from sglang.srt.arg_groups.mamba_hook import validate_mamba_extra_buffer

        validate_mamba_extra_buffer(view, arch, mamba_cache_chunk_size_of=lambda: 64)

    def test_prefill_cuda_graph_is_refused(self):
        # The extend gather's row count is mamba_track_mask.sum(), so a
        # captured prefill graph would freeze it at whatever capture saw
        # (zero) and never snapshot again. Refused at resolution, not at
        # backend construction, so the operator hears about it before the
        # weights load.
        with self.assertRaises(AssertionError):
            self._validate(self._extra_buffer_view(cuda_graph_backend_prefill="full"))
        # Both spellings of "no prefill graph" are the supported configuration.
        self._validate(self._extra_buffer_view(cuda_graph_backend_prefill=None))
        self._validate(self._extra_buffer_view(cuda_graph_backend_prefill="disabled"))

    def test_speculative_decoding_is_refused(self):
        # The decode graph runner drops its mamba-track buffers when a spec
        # algorithm is set, so the snapshot would silently never fire.
        with self.assertRaises(AssertionError):
            self._validate(self._extra_buffer_view(speculative_algorithm="EAGLE"))

    def test_limits_apply_only_to_the_short_conv_track(self):
        # An SSM arch verifies through prepare_mamba_track_for_verify and has a
        # shape-stable extend gather, so neither refusal is its business.
        from sglang.srt.arg_groups.overrides import requires_short_conv_track_limits

        self.assertTrue(requires_short_conv_track_limits("ZayaForCausalLM"))
        self.assertFalse(requires_short_conv_track_limits("Qwen3NextForCausalLM"))

    def test_strided_conv_layout_is_rejected(self):
        # Page-major / envelope conv views are strided, so `flatten(0, 1)` row
        # addressing is invalid. Fail loudly rather than snapshot wrong bytes.
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend.device = torch.device("cpu")
        strided = torch.zeros(2, 6, 3, 4).transpose(2, 3)
        backend.conv_window_lens = [int(strided.shape[-1])]
        mamba_cache = SimpleNamespace(
            conv=[strided], temporal=torch.zeros(2, 6, 1, 1, 0)
        )
        with self.assertRaises(NotImplementedError):
            backend._init_track_state(
                SimpleNamespace(
                    mamba_cache_chunk_size=8,
                    mamba_track_interval=16,
                    speculative_algorithm=None,
                ),
                mamba_cache,
            )


class TestShortConvCacheChunkSize(CustomTestCase):
    """``mamba_cache_chunk_size`` must not be the model's scan length.

    ``hf_config.mamba_chunk_size`` is the SSM chunk-scan length, reported as
    ``1`` by models with no chunked recurrence; ``ServerArgs.
    mamba_cache_chunk_size`` is the radix caching-point granularity. Taking the
    first as the second gives a granularity of 1, below ZAYA1's conv window,
    which makes ``ShortConvAttnBackend`` refuse to start.
    """

    @staticmethod
    def _lfm2_config():
        from sglang.srt.configs.lfm2 import Lfm2Config

        return Lfm2Config(
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            conv_L_cache=3,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
        )

    def test_zaya_window_is_the_two_conv_time_constants(self):
        # The window is (cca_time0 - 1) + (cca_time1 - 1) for conv[0] and 1 for
        # conv[1] -- a pure function of the conv time constants, with no TP term
        # (only the channel axis shards). That is what lets the memoized
        # property read it from either side of distributed init.
        from sglang.srt.arg_groups.overrides import _max_conv_state_window

        config = _make_tiny_config()
        expected = (config.cca_time0 - 1) + (config.cca_time1 - 1)
        self.assertEqual(_max_conv_state_window(config), expected)
        self.assertEqual(expected, 2)

    def test_conv_window_axis_is_tp_invariant(self):
        # Direct proof of the property the memoization relies on: sharding
        # moves the channel axis and leaves the window axis alone.
        from sglang.srt.configs.mamba_utils import Mamba2StateShape

        shapes = [
            Mamba2StateShape.create(
                tp_world_size=tp,
                intermediate_size=64,
                n_groups=1,
                num_heads=tp,
                head_dim=64,
                state_size=0,
                conv_kernel=4,
            )
            for tp in (1, 2, 4)
        ]
        windows = {s.conv[0][-1] for s in shapes}
        channels = {s.conv[0][0] for s in shapes}
        self.assertEqual(windows, {3})
        self.assertEqual(len(channels), 3)

    def test_zaya_gets_the_generic_chunk_not_its_scan_length(self):
        from sglang.srt.arg_groups.overrides import (
            FLA_CHUNK_SIZE,
            _short_conv_cache_chunk_size,
        )

        config = _make_tiny_config()
        self.assertEqual(config.mamba_chunk_size, 1)
        self.assertEqual(_short_conv_cache_chunk_size(config), FLA_CHUNK_SIZE)

    def test_lfm2_hits_the_same_wall_and_the_same_fix(self):
        # LFM2 is on no_buffer today, so this never fires -- but its config
        # reports mamba_chunk_size == 1 with a conv window of conv_L_cache - 1,
        # exactly ZAYA1's shape, so promoting it would have hit the identical
        # assertion. The derivation cures both.
        from sglang.srt.arg_groups.overrides import (
            FLA_CHUNK_SIZE,
            _max_conv_state_window,
            _short_conv_cache_chunk_size,
        )

        config = self._lfm2_config()
        self.assertEqual(config.mamba_chunk_size, 1)
        self.assertEqual(_max_conv_state_window(config), config.conv_L_cache - 1)
        self.assertEqual(_short_conv_cache_chunk_size(config), FLA_CHUNK_SIZE)

    def test_wide_window_rounds_up_and_stays_page_divisible(self):
        # A window at or above the generic chunk raises the granularity, but to
        # a whole number of FLA_CHUNK_SIZE so the caller's
        # max(chunk, page) % min(chunk, page) assert still holds for every page
        # size in use.
        from sglang.srt.arg_groups.overrides import (
            FLA_CHUNK_SIZE,
            _short_conv_cache_chunk_size,
        )

        wide = SimpleNamespace(
            mamba_chunk_size=1,
            mamba2_cache_params=SimpleNamespace(
                shape=SimpleNamespace(conv=[(8, FLA_CHUNK_SIZE + 36), (4, 1)])
            ),
        )
        chunk = _short_conv_cache_chunk_size(wide)
        self.assertEqual(chunk, 2 * FLA_CHUNK_SIZE)
        self.assertGreater(chunk, FLA_CHUNK_SIZE + 36)
        for page_size in (1, 16, 32, 64, 128):
            self.assertEqual(max(chunk, page_size) % min(chunk, page_size), 0)

    def test_unknown_conv_shape_falls_back_up_never_down(self):
        # An unreadable shape must not silently produce a granularity below the
        # generic chunk; the backend guard is then the thing that fails loudly.
        from sglang.srt.arg_groups.overrides import (
            FLA_CHUNK_SIZE,
            _max_conv_state_window,
            _short_conv_cache_chunk_size,
        )

        bare = SimpleNamespace(mamba_chunk_size=1)
        self.assertEqual(_max_conv_state_window(bare), 0)
        self.assertEqual(_short_conv_cache_chunk_size(bare), FLA_CHUNK_SIZE)

        class _Exploding:
            mamba_chunk_size = 1

            @property
            def mamba2_cache_params(self):
                raise AssertionError("tp size not divisible")

        self.assertEqual(_max_conv_state_window(_Exploding()), 0)
        self.assertEqual(_short_conv_cache_chunk_size(_Exploding()), FLA_CHUNK_SIZE)

    def test_multimodal_wrapper_window_is_found_via_text_config(self):
        from sglang.srt.arg_groups.overrides import _max_conv_state_window

        inner = self._lfm2_config()
        wrapper = SimpleNamespace(mamba_chunk_size=1, text_config=inner)
        self.assertEqual(_max_conv_state_window(wrapper), inner.conv_L_cache - 1)

    def test_ssm_models_keep_the_historical_derivation(self):
        # Only the mamba_chunk_size <= 1 branch changed. A real scan length, and
        # the FLA_CHUNK_SIZE default for a config that declares none, must both
        # go through max(scan_chunk, page_size) untouched -- so the short-conv
        # helper is never consulted for them.
        from sglang.srt.arg_groups.overrides import FLA_CHUNK_SIZE

        def resolve(hf_config, page_size):
            scan = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)
            self.assertGreater(scan, 1)  # takes the untouched branch
            return max(scan, page_size)

        self.assertEqual(resolve(SimpleNamespace(mamba_chunk_size=256), 1), 256)
        self.assertEqual(resolve(SimpleNamespace(mamba_chunk_size=256), 128), 256)
        self.assertEqual(resolve(SimpleNamespace(), 1), FLA_CHUNK_SIZE)
        self.assertEqual(resolve(SimpleNamespace(), 128), 128)
        # Inkling already worked around this by pinning 64 in its config; the
        # new branch must leave that answer alone.
        self.assertEqual(resolve(SimpleNamespace(mamba_chunk_size=64), 1), 64)

    def test_derived_chunk_satisfies_the_backend_guard(self):
        # The regression this fixes end to end: with the old chunk of 1 the
        # short-conv backend refused to build its track state; with the derived
        # chunk it builds.
        from sglang.srt.arg_groups.overrides import _short_conv_cache_chunk_size

        config = _make_tiny_config()
        derived = _short_conv_cache_chunk_size(config)

        with self.assertRaises(AssertionError):
            _TrackHarness(chunk_size=1, windows=(2, 1))
        # mamba_track_interval defaults to 256, comfortably above the derived
        # 64 -- the backend's other precondition.
        harness = _TrackHarness(chunk_size=derived, track_interval=256, windows=(2, 1))
        self.assertEqual(len(harness.backend._track_pairs), 1)
        self.assertGreaterEqual(256, derived)

    def test_guard_names_the_minimum_viable_chunk(self):
        # The failure an operator sees has to say what value would work.
        with self.assertRaises(AssertionError) as caught:
            _TrackHarness(chunk_size=1, windows=(2, 1))
        message = str(caught.exception)
        self.assertIn("minimum viable chunk here is 3", message)
        self.assertIn("mamba_cache_chunk_size", message)


def _make_swa_config(
    *,
    num_hidden_layers: int,
    swa_layers,
    swa_rotary_base=10000,
    rope_theta: float = 10_000_000.0,
):
    """ZAYA1-74B-style config: a per-layer ``swa_layers`` window list (aligned
    with the global layer index, 0 == full attention) plus a dedicated RoPE
    base for the sliding-window layers."""
    from sglang.srt.configs.zaya import ZayaConfig

    return ZayaConfig(
        hidden_size=16,
        ffn_hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_experts=2,
        num_attention_heads=4,
        num_query_groups=2,
        num_key_value_heads=2,
        head_dim=8,
        cca_time0=2,
        cca_time1=2,
        max_position_embeddings=64,
        moe_router_topk=1,
        zaya_mlp_expansion=8,
        attention_bias=False,
        rope_theta=rope_theta,
        swa_layers=swa_layers,
        swa_rotary_base=swa_rotary_base,
    )


class TestZayaSlidingWindowAttention(CustomTestCase):
    """ZAYA1-74B interleaves sliding-window attention with full attention and
    gives the sliding layers their own RoPE base (``swa_rotary_base``). These
    tests verify that ``ZayaConfig`` resolves the per-layer window from
    ``swa_layers`` and that ``ZayaAttention`` wires the matching
    ``RadixAttention.sliding_window_size`` and rotary base into each layer.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        # Building a RoPE cache reads a published config leaf, so this class
        # needs a published context; ``override_server_args`` is the sanctioned
        # way for a test to get one (it publishes and projects the bags).
        from sglang.srt.runtime_context import get_context

        cls._server_args_override = get_context().override_server_args(
            model_path="dummy"
        )
        cls._server_args_override.install()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server_args_override.restore()

    def _build_attention(self, config, layer_id: int):
        from sglang.srt.models.zaya import ZayaAttention

        return ZayaAttention(config=config, layer_id=layer_id)

    def test_config_reports_per_layer_window(self):
        # 8 layers: attention layers live at the even ids 0/2/4/6 (zaya_layers
        # is None), and ``swa_layers`` marks 0 and 4 as sliding (window 4096).
        config = _make_swa_config(
            num_hidden_layers=8,
            swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0],
        )
        self.assertEqual(config.sliding_window_for_layer(0), 4096)
        self.assertEqual(config.sliding_window_for_layer(4), 4096)
        self.assertEqual(config.sliding_window_for_layer(2), 0)
        self.assertEqual(config.sliding_window_for_layer(6), 0)
        self.assertEqual(config.swa_window_size, 4096)
        # window - 1: the exclusive convention shared with the attention
        # backends (matches the Gemma reference models).
        self.assertEqual(config.get_attention_sliding_window_size(), 4095)

    def test_non_uniform_window_is_rejected(self):
        # The runtime tracks a single global window, so mixed window sizes
        # across SWA layers are unsupported and must fail loudly -- and must do
        # so while *constructing* the config (the hybrid-SWA opt-in resolves the
        # window in __init__), not lazily on first attribute read, so an
        # unsupported checkpoint is rejected at load rather than mid-serving.
        with self.assertRaises(AssertionError):
            _make_swa_config(
                num_hidden_layers=8,
                swa_layers=[4096, 0, 0, 0, 2048, 0, 0, 0],
            )

    def test_attention_selects_window_and_rope_base_per_layer(self):
        config = _make_swa_config(
            num_hidden_layers=8,
            swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0],
            swa_rotary_base=10000,
            rope_theta=10_000_000.0,
        )
        sliding = self._build_attention(config, layer_id=0)
        full = self._build_attention(config, layer_id=2)

        # Sliding layer: window-1 handed to RadixAttention, SWA rope base.
        self.assertTrue(sliding.is_sliding)
        self.assertEqual(sliding.attn.sliding_window_size, 4095)
        self.assertEqual(sliding.rotary_emb.base, 10000)

        # Full layer: no window (-1) and the global rope base.
        self.assertFalse(full.is_sliding)
        self.assertEqual(full.attn.sliding_window_size, -1)
        self.assertEqual(full.rotary_emb.base, 10_000_000)

        # Distinct rope bases must not collapse onto a shared rotary cache entry.
        self.assertIsNot(sliding.rotary_emb, full.rotary_emb)

    def test_base_checkpoint_has_no_sliding_window(self):
        # No ``swa_layers`` -> every attention layer is full attention and the
        # model reports no global window, preserving base-model behavior.
        config = _make_swa_config(num_hidden_layers=4, swa_layers=None)
        self.assertEqual(config.sliding_window_for_layer(0), 0)
        self.assertIsNone(config.swa_window_size)
        self.assertIsNone(config.get_attention_sliding_window_size())

        attn = self._build_attention(config, layer_id=0)
        self.assertFalse(attn.is_sliding)
        self.assertEqual(attn.attn.sliding_window_size, -1)
        self.assertEqual(attn.rotary_emb.base, int(config.rope_theta))

    def test_hybrid_layer_pattern_is_indexed_by_global_layer_id(self):
        # The pattern is indexed by *global* layer id and covers every layer so
        # get_hybrid_layer_ids can index it directly. Note the two same-named
        # properties mean different things: ZayaConfig.full_attention_layer_ids is
        # the mamba interface's "every attention layer" (each carries a CCA conv
        # state), whereas ModelConfig.full_attention_layer_ids after the SWA split
        # means "non-sliding attention layers" only.
        config = _make_swa_config(
            num_hidden_layers=8,
            swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0],
        )
        self.assertEqual(config.hybrid_layer_pattern, [1, -1, 0, -1, 1, -1, 0, -1])
        self.assertEqual(config.swa_attention_layer_ids, [0, 4])
        self.assertEqual(config.full_attention_layer_ids, [0, 2, 4, 6])

    def test_base_checkpoint_reports_no_hybrid_pattern(self):
        # Without swa_layers the model must not opt in to the hybrid-SWA KV
        # pool: a non-None pattern here would make get_hybrid_layer_ids split a
        # uniformly full-attention model into empty/complete halves.
        config = _make_swa_config(num_hidden_layers=4, swa_layers=None)
        self.assertIsNone(config.hybrid_layer_pattern)
        self.assertEqual(config.swa_attention_layer_ids, [])

    def test_window_size_conventions(self):
        # Two different conventions coexist and must not be conflated:
        # ``sliding_window_size`` is the inclusive window that ModelConfig reads
        # from the HF config, while the attention backends take the exclusive
        # ``window - 1`` via the model's get_attention_sliding_window_size.
        swa = _make_swa_config(
            num_hidden_layers=8, swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0]
        )
        self.assertEqual(swa.sliding_window_size, 4096)
        self.assertEqual(swa.get_attention_sliding_window_size(), 4095)

        base = _make_swa_config(num_hidden_layers=4, swa_layers=None)
        self.assertIsNone(base.sliding_window_size)

    def test_hybrid_swa_optin_is_per_checkpoint(self):
        # is_hybrid_swa is the generic escape from ModelConfig's arch allowlist, so
        # it must key off the checkpoint (swa_layers) rather than the architecture:
        # base ZAYA1 checkpoints have no sliding-window layers and must stay on the
        # single-pool path.
        from sglang.srt.configs.model_config import is_hybrid_swa_model

        swa = _make_swa_config(
            num_hidden_layers=8, swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0]
        )
        self.assertTrue(swa.is_hybrid_swa)
        self.assertTrue(is_hybrid_swa_model(["ZayaForCausalLM"], swa))

        base = _make_swa_config(num_hidden_layers=4, swa_layers=None)
        self.assertFalse(base.is_hybrid_swa)
        self.assertFalse(is_hybrid_swa_model(["ZayaForCausalLM"], base))

    def test_moe_layers_are_excluded_from_both_kv_layer_lists(self):
        # ZAYA1's odd layers are MoE and hold no KV. get_hybrid_layer_ids
        # derives swa_attention_layer_ids from `pattern == 1` and
        # full_attention_layer_ids from `pattern == 0`, and those lists SIZE the
        # sub-pools rather than merely indexing them: reporting MoE layers as 0
        # made the full sub-pool 90 layers wide instead of 30 on the 74B. They
        # must be in NEITHER list.
        from sglang.srt.configs.model_config import get_hybrid_layer_ids

        config = _make_swa_config(
            num_hidden_layers=8, swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0]
        )
        self.assertEqual(config.hybrid_layer_pattern, [1, -1, 0, -1, 1, -1, 0, -1])

        swa_ids, full_ids = get_hybrid_layer_ids(["ZayaForCausalLM"], config)
        self.assertEqual(swa_ids, [0, 4])
        self.assertEqual(full_ids, [2, 6])
        self.assertEqual(set(swa_ids) & set(full_ids), set())
        # Every KV-bearing layer is an attention layer, and no MoE layer appears.
        moe_layers = {1, 3, 5, 7}
        self.assertEqual(set(swa_ids + full_ids) & moe_layers, set())
        self.assertEqual(
            sorted(swa_ids + full_ids), sorted(config.full_attention_layer_ids)
        )


class TestZayaCCATensorParallel(CustomTestCase):
    """Head-parallel TP equivalence: each rank's q / k / v must equal the head
    slice of the TP=1 reference for that rank's heads, so the grouped-mean step
    and ``conv_qk.1`` partition across heads with no cross-rank leakage.
    """

    TP_SIZE = 2

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def _slice_full_state_dict_into_rank(self, ref_cca, tp_cca, tp_rank: int):
        """Copy the reference's full weights into the per-rank CCA through the
        per-parameter ``weight_loader`` the CCA installs in ``__init__``,
        mirroring what ``ZayaForCausalLM.load_weights`` does at serving time.
        """
        ref_state = dict(ref_cca.state_dict())
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        with torch.no_grad():
            for name, param in tp_cca.named_parameters():
                full_weight = ref_state[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, full_weight)

    def _check_per_rank_outputs(
        self,
        full_q: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        rank_q: torch.Tensor,
        rank_k: torch.Tensor,
        rank_v: torch.Tensor,
        tp_rank: int,
        cfg,
    ):
        """Compare a TP=2 rank's output against the corresponding head slice
        of the TP=1 reference output. Q heads and K heads are partitioned
        contiguously across ranks: rank ``r`` owns
        ``[r*Q_per_rank, (r+1)*Q_per_rank)`` for Q and similarly for K.
        """
        q_heads_per_rank = cfg.num_attention_heads // self.TP_SIZE
        k_heads_per_rank = cfg.num_query_groups // self.TP_SIZE
        q_lo, q_hi = tp_rank * q_heads_per_rank, (tp_rank + 1) * q_heads_per_rank
        k_lo, k_hi = tp_rank * k_heads_per_rank, (tp_rank + 1) * k_heads_per_rank
        torch.testing.assert_close(
            rank_q, full_q[:, q_lo:q_hi, :], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            rank_k, full_k[:, k_lo:k_hi, :], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            rank_v, full_v[:, k_lo:k_hi, :], atol=1e-5, rtol=1e-5
        )

    def test_tp2_extend_matches_full(self):
        """Single-chunk extend with TP=2 produces the same q / k / v slices
        as a TP=1 reference, verified rank-by-rank.
        """
        ref_cca, cfg = _make_tiny_cca(seed=21, tp_rank=0, tp_size=1)
        S = 6
        torch.manual_seed(901)
        hs = torch.randn(S, ref_cca.hidden_size, dtype=torch.float32) * 0.1

        ref_pool = _MockReqToTokenPool(pool_size=8, cca_config=cfg, tp_size=1)
        ref_fb = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[0],
            input_ids=torch.arange(S, dtype=torch.int64),
        )
        with _mock_pool_context(ref_pool):
            full_q, full_k, full_v = ref_cca.forward(hs, ref_fb)

        for tp_rank in range(self.TP_SIZE):
            rank_cca, _ = _make_tiny_cca(
                seed=21 + tp_rank, tp_rank=tp_rank, tp_size=self.TP_SIZE
            )
            self._slice_full_state_dict_into_rank(ref_cca, rank_cca, tp_rank)
            rank_pool = _MockReqToTokenPool(
                pool_size=8, cca_config=cfg, tp_size=self.TP_SIZE
            )
            rank_fb = _make_forward_batch(
                is_decode=False,
                extend_seq_lens_cpu=[S],
                extend_prefix_lens_cpu=[0],
                req_pool_indices=[0],
                input_ids=torch.arange(S, dtype=torch.int64),
            )
            with _mock_pool_context(rank_pool):
                rank_q, rank_k, rank_v = rank_cca.forward(hs, rank_fb)
            self._check_per_rank_outputs(
                full_q, full_k, full_v, rank_q, rank_k, rank_v, tp_rank, cfg
            )

    def test_tp2_decode_matches_full(self):
        """Prefill(S0) + decode(1 token) with TP=2 produces the same q / k / v
        slices as a TP=1 reference, verifying that the per-rank conv state
        and prev_hs cache (which is replicated on every rank) agree.
        """
        ref_cca, cfg = _make_tiny_cca(seed=22, tp_rank=0, tp_size=1)
        S0 = 5
        torch.manual_seed(902)
        hs_prefill = torch.randn(S0, ref_cca.hidden_size, dtype=torch.float32) * 0.1
        hs_decode = torch.randn(1, ref_cca.hidden_size, dtype=torch.float32) * 0.1

        ref_pool = _MockReqToTokenPool(pool_size=8, cca_config=cfg, tp_size=1)
        with _mock_pool_context(ref_pool):
            ref_cca.forward(
                hs_prefill,
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S0, dtype=torch.int64),
                ),
            )
            full_q, full_k, full_v = ref_cca.forward(
                hs_decode,
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                ),
            )

        for tp_rank in range(self.TP_SIZE):
            rank_cca, _ = _make_tiny_cca(
                seed=22 + tp_rank, tp_rank=tp_rank, tp_size=self.TP_SIZE
            )
            self._slice_full_state_dict_into_rank(ref_cca, rank_cca, tp_rank)
            rank_pool = _MockReqToTokenPool(
                pool_size=8, cca_config=cfg, tp_size=self.TP_SIZE
            )
            with _mock_pool_context(rank_pool):
                rank_cca.forward(
                    hs_prefill,
                    _make_forward_batch(
                        is_decode=False,
                        extend_seq_lens_cpu=[S0],
                        extend_prefix_lens_cpu=[0],
                        req_pool_indices=[0],
                        input_ids=torch.arange(S0, dtype=torch.int64),
                    ),
                )
                rank_q, rank_k, rank_v = rank_cca.forward(
                    hs_decode,
                    _make_forward_batch(
                        is_decode=True,
                        extend_seq_lens_cpu=[],
                        extend_prefix_lens_cpu=[],
                        req_pool_indices=[0],
                        input_ids=torch.tensor([0], dtype=torch.int64),
                    ),
                )
            self._check_per_rank_outputs(
                full_q, full_k, full_v, rank_q, rank_k, rank_v, tp_rank, cfg
            )

    def test_tp2_conv_state_is_per_rank_sliced(self):
        """After a TP=2 prefill, each rank's conv state must equal the head
        slice of the TP=1 conv state corresponding to that rank's heads.
        """
        ref_cca, cfg = _make_tiny_cca(seed=23, tp_rank=0, tp_size=1)
        S = 4
        torch.manual_seed(903)
        hs = torch.randn(S, ref_cca.hidden_size, dtype=torch.float32) * 0.1

        ref_pool = _MockReqToTokenPool(pool_size=4, cca_config=cfg, tp_size=1)
        with _mock_pool_context(ref_pool):
            ref_cca.forward(
                hs,
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S, dtype=torch.int64),
                ),
            )
        full_state = ref_pool.mamba2_layer_cache(0).conv[0][0]  # [in_out_ch_full, pad]

        head_dim = cfg.head_dim
        num_q_heads_full = cfg.num_attention_heads
        num_k_heads_full = cfg.num_query_groups
        latent_q_full = num_q_heads_full * head_dim
        q_per_rank = num_q_heads_full // self.TP_SIZE
        k_per_rank = num_k_heads_full // self.TP_SIZE

        for tp_rank in range(self.TP_SIZE):
            rank_cca, _ = _make_tiny_cca(
                seed=23 + tp_rank, tp_rank=tp_rank, tp_size=self.TP_SIZE
            )
            self._slice_full_state_dict_into_rank(ref_cca, rank_cca, tp_rank)
            rank_pool = _MockReqToTokenPool(
                pool_size=4, cca_config=cfg, tp_size=self.TP_SIZE
            )
            with _mock_pool_context(rank_pool):
                rank_cca.forward(
                    hs,
                    _make_forward_batch(
                        is_decode=False,
                        extend_seq_lens_cpu=[S],
                        extend_prefix_lens_cpu=[0],
                        req_pool_indices=[0],
                        input_ids=torch.arange(S, dtype=torch.int64),
                    ),
                )
            rank_state = rank_pool.mamba2_layer_cache(0).conv[0][0]

            q_lo = tp_rank * q_per_rank * head_dim
            q_hi = q_lo + q_per_rank * head_dim
            k_lo = latent_q_full + tp_rank * k_per_rank * head_dim
            k_hi = k_lo + k_per_rank * head_dim
            expected = torch.cat([full_state[q_lo:q_hi], full_state[k_lo:k_hi]], dim=0)

            torch.testing.assert_close(rank_state, expected, atol=1e-5, rtol=1e-5)

    def test_tp2_lag_stream_only_exists_on_the_val_proj2_rank(self):
        """At attn_tp == 2 with two K heads, only rank 1 reads ``val_proj2``.

        Rank 0 takes its V heads entirely from ``val_proj1`` and needs no lag at
        all. Pin both halves: the write side and the read side derive the range
        separately, and if they disagree rank 1 reads a slot nothing filled
        (silently zero) or rank 0 pays a GEMM it never uses.
        """
        ref_cca, cfg = _make_tiny_cca(seed=24, tp_rank=0, tp_size=1)
        S = 4
        torch.manual_seed(904)
        hs = torch.randn(S, ref_cca.hidden_size, dtype=torch.float32) * 0.1

        # tp=1 straddles the boundary and uses all of val_proj2's output.
        self.assertTrue(ref_cca.v_uses_val1)
        self.assertTrue(ref_cca.v_uses_val2)
        self.assertEqual(ref_cca.v2_lag_dim, cfg.cca_v2_state_dim)

        expected = {0: (True, False, 0), 1: (False, True, cfg.cca_v2_state_dim)}
        for tp_rank in range(self.TP_SIZE):
            rank_cca, _ = _make_tiny_cca(
                seed=24 + tp_rank, tp_rank=tp_rank, tp_size=self.TP_SIZE
            )
            self._slice_full_state_dict_into_rank(ref_cca, rank_cca, tp_rank)
            uses1, uses2, lag_dim = expected[tp_rank]
            self.assertEqual(rank_cca.v_uses_val1, uses1)
            self.assertEqual(rank_cca.v_uses_val2, uses2)
            self.assertEqual(rank_cca.v2_lag_dim, lag_dim)

            rank_pool = _MockReqToTokenPool(
                pool_size=4, cca_config=cfg, tp_size=self.TP_SIZE
            )
            with _mock_pool_context(rank_pool):
                rank_cca.forward(
                    hs,
                    _make_forward_batch(
                        is_decode=False,
                        extend_seq_lens_cpu=[S],
                        extend_prefix_lens_cpu=[0],
                        req_pool_indices=[0],
                        input_ids=torch.arange(S, dtype=torch.int64),
                    ),
                )
            lag_slot = rank_pool.mamba2_layer_cache(0).conv[1][0]
            # The pool entry stays rank-uniform (a per-rank size would desync
            # max_mamba_cache_size across the TP group); rank 0 just leaves it at
            # the zeros MambaPool handed out.
            self.assertEqual(lag_slot.shape[-2], cfg.cca_v2_state_dim)
            if uses2:
                self.assertTrue(torch.any(lag_slot != 0))
                with torch.no_grad():
                    expected_row = rank_cca.val_proj2(hs[-1:])[0].squeeze(0)
                torch.testing.assert_close(
                    lag_slot.squeeze(-1), expected_row, atol=1e-5, rtol=1e-5
                )
            else:
                self.assertTrue(torch.all(lag_slot == 0))

    def test_tp_assertions_reject_indivisible_head_counts(self):
        """The CCA constructor must reject TP sizes that don't evenly divide
        both num_q_heads and num_k_heads, since both grouped-mean and
        conv_qk.1 require each rank to hold whole K-head groups.
        """
        from sglang.srt.models.zaya import CCA

        cfg = _make_tiny_config()
        # tiny config has num_query_groups=2; TP=4 cannot divide it cleanly.
        with self.assertRaises(AssertionError):
            CCA(
                config=cfg,
                cca_num_k_heads=cfg.num_query_groups,
                cca_num_q_heads=cfg.num_attention_heads,
                hidden_size=cfg.hidden_size,
                head_dim=cfg.head_dim,
                cca_time0=cfg.cca_time0,
                cca_time1=cfg.cca_time1,
                layer_id=0,
                tp_rank=0,
                tp_size=4,
            )


@contextmanager
def _dp_layout(sizes: List[int], rank: int, is_max_len: bool):
    """Install ``sizes`` as the per-rank padded token counts, viewed from ``rank``.

    Mirrors what ``prepare_mlp_sync_batch`` publishes for a real forward: the DP
    rank globals plus the process-wide DP buffer metadata.
    """
    from sglang.srt.layers import dp_attention as dpa

    saved = (dpa._ATTN_DP_RANK, dpa._ATTN_DP_SIZE)
    saved_buffer = (
        dpa._DpGatheredBufferWrapper._global_dp_buffer_len,
        dpa._DpGatheredBufferWrapper._local_dp_buffer_len,
        dpa._DpGatheredBufferWrapper._dp_max_padding,
        dpa._DpGatheredBufferWrapper._global_num_tokens,
    )
    dpa._ATTN_DP_RANK, dpa._ATTN_DP_SIZE = rank, len(sizes)
    dpa.set_dp_buffer_len(sum(sizes), sizes[rank], is_max_len, list(sizes))
    try:
        yield
    finally:
        dpa._ATTN_DP_RANK, dpa._ATTN_DP_SIZE = saved
        dpa.set_dp_buffer_len(*saved_buffer)


class TestZayaGlobalResidualLayout(CustomTestCase):
    """Row arithmetic for the global-residual DP dataflow.

    The CPU offsets in ``GlobalResidualLayout`` and the device-side offsets
    ``dp_gather_partial`` writes at (``get_dp_local_info``) must describe the same
    rows. Nothing at runtime cross-checks them: if they drift, attention silently
    reads another replica's tokens.
    """

    def _layout(self, sizes: List[int], rank: int, is_max_len: bool):
        from unittest import mock

        from sglang.srt.environ import envs
        from sglang.srt.models import zaya

        # The parallel-layout precondition is orthogonal to the arithmetic under
        # test and needs a live runtime context, so stub it out.
        with _dp_layout(sizes, rank, is_max_len), mock.patch.object(
            zaya, "dp_gather_required", return_value=True
        ), envs.SGLANG_OPT_ZAYA_GLOBAL_RESIDUAL.override(True):
            return zaya.global_residual_layout()

    def _device_slice(self, sizes: List[int], rank: int):
        from sglang.srt.layers.dp_attention import get_dp_local_info

        forward_batch = SimpleNamespace(
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            global_num_tokens_gpu=torch.tensor(sizes, dtype=torch.int32),
        )
        with _dp_layout(sizes, rank, is_max_len=False):
            start, length = get_dp_local_info(forward_batch)
        return int(start), int(length)

    def test_offsets_match_the_gather_destination(self):
        """SUM_LEN: unequal per-rank counts, so a wrong cumsum shows up as a shift."""
        sizes = [3, 7, 1, 5]
        for rank in range(len(sizes)):
            layout = self._layout(sizes, rank, is_max_len=False)
            self.assertEqual(
                (layout.local_start, layout.local_len),
                self._device_slice(sizes, rank),
                f"CPU layout disagrees with get_dp_local_info at rank {rank}",
            )

    def test_offsets_match_under_max_len_padding(self):
        """MAX_LEN: every rank padded to the same width (the cuda-graph layout)."""
        sizes = [8, 8, 8, 8]
        for rank in range(len(sizes)):
            layout = self._layout(sizes, rank, is_max_len=True)
            self.assertEqual(
                (layout.local_start, layout.local_len),
                (rank * 8, 8),
            )

    def test_slices_tile_the_buffer_without_gaps_or_overlap(self):
        """Every row of the global buffer belongs to exactly one replica.

        The MoE layers run over the whole buffer, so a gap would feed the experts
        uninitialized rows and an overlap would double-count a token.
        """
        sizes = [3, 7, 1, 5]
        covered = torch.zeros(sum(sizes), dtype=torch.int32)
        for rank in range(len(sizes)):
            layout = self._layout(sizes, rank, is_max_len=False)
            covered[layout.local_start : layout.local_start + layout.local_len] += 1
        self.assertTrue(torch.equal(covered, torch.ones_like(covered)))

    def test_idle_replica_gets_an_empty_slice(self):
        """A replica with no requests must slice to zero rows, not to its neighbour's.

        Its attention still runs (and returns early on the empty batch) and it must
        still join the gather, so the layout has to survive a zero-width block.
        """
        sizes = [4, 0, 4]
        layout = self._layout(sizes, 1, is_max_len=False)
        self.assertEqual((layout.local_start, layout.local_len), (4, 0))
        hidden = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
        self.assertEqual(layout.local_view(hidden).shape, (0, 2))

    def test_local_view_is_a_contiguous_alias(self):
        """Attention and the gather both assert contiguity on what they are handed."""
        layout = self._layout([3, 7, 1, 5], 1, is_max_len=False)
        hidden = torch.randn(16, 4)
        view = layout.local_view(hidden)
        self.assertTrue(view.is_contiguous())
        self.assertEqual(view.data_ptr(), hidden[3].data_ptr())

    def test_disabled_by_default_without_touching_parallel_state(self):
        """No global layout when the expert reduce does not span DP replicas.

        NOTE: this does not exercise the env flag, which defaults to True --
        ``global_residual_layout`` returns None here because
        ``dp_gather_required()`` is false for this fixture. The claim that the
        env check short-circuits ahead of the parallel-layout probe is untested.
        """
        from sglang.srt.models import zaya

        with _dp_layout([3, 7, 1, 5], 0, is_max_len=False):
            self.assertIsNone(zaya.global_residual_layout())


class TestZayaPartialGatherFoldsTheAttnReduce(CustomTestCase):
    """The algebra that lets one collective replace two.

    The global-residual dataflow gathers the *unreduced* o_proj partials instead
    of running ``attn_tp_all_reduce``: every attention-TP rank memcpys its
    partial into the same slot, so the cross-replica all-reduce also sums within
    each replica. Pins that equivalence and the failure mode of using the
    replicate gather, which keeps rank 0's rows and discards the rest.
    """

    HIDDEN = 4

    def _gather(self, partials: List[List[torch.Tensor]], sizes: List[int], is_partial):
        """Replay ``_dp_gather_via_all_reduce`` over all ranks on host tensors.

        Each rank zero-fills the buffer and memcpys its own rows into its replica's
        slot -- only attention-TP rank 0 does so on the replicate gather -- and the
        all-reduce leaves the sum of those per-rank buffers behind.
        """
        from sglang.srt.layers.dp_attention import get_dp_local_info

        total = torch.zeros(sum(sizes), self.HIDDEN)
        for replica, rank_partials in enumerate(partials):
            forward_batch = SimpleNamespace(
                dp_local_start_pos=None,
                dp_local_num_tokens=None,
                global_num_tokens_gpu=torch.tensor(sizes, dtype=torch.int32),
            )
            with _dp_layout(sizes, replica, is_max_len=False):
                start, length = get_dp_local_info(forward_batch)
            for attn_tp_rank, partial in enumerate(rank_partials):
                if not is_partial and attn_tp_rank != 0:
                    continue
                contribution = torch.zeros(sum(sizes), self.HIDDEN)
                contribution[int(start) : int(start) + int(length)] = partial
                total = total + contribution
        return total

    def _partials(self, sizes: List[int], attn_tp_size: int):
        torch.manual_seed(0)
        return [
            [torch.randn(rows, self.HIDDEN) for _ in range(attn_tp_size)]
            for rows in sizes
        ]

    def test_partial_gather_equals_reduce_then_replicate_gather(self):
        sizes = [3, 7, 1, 5]
        partials = self._partials(sizes, attn_tp_size=2)
        # Baseline: reduce within the replica, then gather the reduced rows.
        reduced = [[sum(rank_partials)] for rank_partials in partials]
        baseline = self._gather(reduced, sizes, is_partial=False)
        folded = self._gather(partials, sizes, is_partial=True)
        torch.testing.assert_close(folded, baseline)

    def test_replicate_gather_of_partials_drops_a_rank(self):
        """Guards the swap this campaign has already made in the other direction.

        ``dp_gather_replicate`` is correct for already-reduced rows and wrong for
        partials: it keeps attention-TP rank 0's contribution and silently discards
        every other rank's, which at attn_tp=1 is invisible and at attn_tp=2 is
        garbage output.
        """
        sizes = [3, 7, 1, 5]
        partials = self._partials(sizes, attn_tp_size=2)
        folded = self._gather(partials, sizes, is_partial=True)
        wrong = self._gather(partials, sizes, is_partial=False)
        self.assertFalse(torch.allclose(folded, wrong))

    def test_single_attn_tp_rank_makes_the_two_gathers_agree(self):
        """At attn_tp=1 there is nothing to sum, so both gathers must coincide."""
        sizes = [3, 7, 1, 5]
        partials = self._partials(sizes, attn_tp_size=1)
        torch.testing.assert_close(
            self._gather(partials, sizes, is_partial=True),
            self._gather(partials, sizes, is_partial=False),
        )


class TestDpGatherStagingFusion(CustomTestCase):
    """The two launches per sum_len partial gather that the staging fusion drops.

    The ``fill_(0)`` folds into the memcpy, and the copy-back out of the
    out-of-place all-reduce is skipped by callers that take the returned tensor.
    Both are meant to be *exactly* the old behaviour, so pin the fused kernel
    against ``fill_(0) + memcpy`` and pin what the gather returns for an
    out-of-place and an in-place collective.
    """

    @staticmethod
    @contextmanager
    def _sum_len_all_reduce(all_reduce):
        """Drive the sum_len gather path on host tensors.

        Patches out everything that needs a live NCCL group: the WORLD-gather
        switch, the attention-TP rank, the world size, the collective itself and
        the memcpy dispatch (CPU tensors cannot go through Triton).
        """
        from unittest import mock

        from sglang.srt.layers import dp_attention as dpa

        with mock.patch.object(
            dpa, "memcpy_scatter_zero_rest_func", dpa.memcpy_scatter_zero_rest_cpu
        ), mock.patch.object(
            dpa, "world_dp_gather_enabled", lambda: False
        ), mock.patch.object(
            dpa, "get_attn_tensor_model_parallel_rank", lambda: 0
        ), mock.patch.object(
            dpa, "get_tensor_model_parallel_world_size", lambda: 1
        ), mock.patch.object(
            dpa, "tensor_model_parallel_all_reduce", all_reduce
        ):
            yield

    @staticmethod
    def _forward_batch(sizes: List[int]):
        from sglang.srt.layers.dp_attention import DpPaddingMode

        return SimpleNamespace(
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            global_num_tokens_gpu=torch.tensor(sizes, dtype=torch.int32),
            dp_padding_mode=DpPaddingMode.SUM_LEN,
        )

    def test_fused_scatter_zero_matches_fill_then_memcpy(self):
        """The fused staging write must equal ``fill_(0)`` then ``memcpy``."""
        from sglang.srt.layers.dp_attention import (
            memcpy_cpu,
            memcpy_scatter_zero_rest_cpu,
        )

        torch.manual_seed(3)
        rows, hidden = 16, 5
        # sz == 0 (idle replica) and sz < src rows (a src padded for cuda graph)
        # are the two cases where the fill is the only thing zeroing a row.
        for offset, sz, src_rows in ((0, 4, 4), (4, 3, 3), (12, 4, 4), (7, 0, 2)):
            with self.subTest(offset=offset, sz=sz):
                src = torch.randn(src_rows, hidden)
                # A dirty destination: the fill is what makes the untouched rows
                # zero, so starting from zeros would hide a missing zero-fill.
                dirty = torch.randn(rows, hidden)
                ref = dirty.clone()
                ref.fill_(0)
                memcpy_cpu(
                    ref,
                    src,
                    0,
                    torch.tensor(offset),
                    torch.tensor(sz),
                    False,
                )
                got = dirty.clone()
                memcpy_scatter_zero_rest_cpu(
                    got, src, 0, torch.tensor(offset), torch.tensor(sz)
                )
                self.assertTrue(torch.equal(got, ref))

    @unittest.skipUnless(torch.cuda.is_available(), "Triton memcpy needs a GPU")
    def test_fused_scatter_zero_matches_on_device(self):
        """Same equivalence for the Triton kernel the serving path uses."""
        from sglang.kernels.ops.memory.memcpy_triton import (
            memcpy_scatter_zero_rest_triton,
            memcpy_triton,
        )

        dev = "cuda"
        torch.manual_seed(4)
        cases = (
            # (dst shape, src shape, offset, sz)
            ((16, 5), (4, 5), 4, 4),
            ((16, 5), (8, 5), 0, 3),
            ((16, 5), (4, 5), 12, 4),
            ((16, 5), (4, 5), 7, 0),
            # 1-D int32 input ids: chunk_size collapses to 1.
            ((16,), (4,), 8, 4),
        )
        for dst_shape, src_shape, offset, sz in cases:
            for dtype in (torch.bfloat16, torch.int32):
                with self.subTest(dst=dst_shape, off=offset, sz=sz, dt=dtype):
                    if dtype.is_floating_point:
                        src = torch.randn(src_shape, device=dev).to(dtype)
                        dirty = torch.randn(dst_shape, device=dev).to(dtype)
                    else:
                        src = torch.randint(1, 99, src_shape, device=dev, dtype=dtype)
                        dirty = torch.randint(1, 99, dst_shape, device=dev, dtype=dtype)
                    off_t = torch.tensor(offset, device=dev, dtype=torch.int32)
                    sz_t = torch.tensor(sz, device=dev, dtype=torch.int32)

                    ref = dirty.clone()
                    ref.fill_(0)
                    memcpy_triton(ref, src, 0, off_t, sz_t, False)
                    got = dirty.clone()
                    memcpy_scatter_zero_rest_triton(got, src, 0, off_t, sz_t)
                    self.assertTrue(torch.equal(got, ref))

    def test_out_variant_returns_the_collective_output(self):
        """An out-of-place all-reduce: the result is the collective's buffer."""
        from sglang.srt.layers.dp_attention import dp_gather_partial_out

        sizes = [3, 7, 1, 5]
        hidden = 4
        marker = torch.full((sum(sizes), hidden), 7.0)

        def all_reduce(x):
            # Mimic the custom all-reduce: a fresh buffer, input untouched.
            return x + marker

        local = torch.randn(sizes[1], hidden)
        staging = torch.randn(sum(sizes), hidden)
        fb = self._forward_batch(sizes)
        with _dp_layout(sizes, 1, is_max_len=False):
            with self._sum_len_all_reduce(all_reduce):
                out = dp_gather_partial_out(staging, local, fb)

        expected = torch.zeros(sum(sizes), hidden)
        expected[3 : 3 + 7] = local
        self.assertIsNot(out, staging)
        torch.testing.assert_close(out, expected + marker)
        # The staging buffer is left holding the pre-reduce content, which is
        # what makes the copy-back removable: nothing reads it afterwards.
        torch.testing.assert_close(staging, expected)

    def test_out_variant_returns_the_buffer_for_an_in_place_collective(self):
        """An in-place all-reduce returns its input, so the buffer is the result."""
        from sglang.srt.layers.dp_attention import dp_gather_partial_out

        sizes = [2, 2]
        hidden = 3

        def all_reduce(x):
            x.add_(1.0)
            return x

        local = torch.randn(sizes[0], hidden)
        staging = torch.randn(sum(sizes), hidden)
        fb = self._forward_batch(sizes)
        with _dp_layout(sizes, 0, is_max_len=False):
            with self._sum_len_all_reduce(all_reduce):
                out = dp_gather_partial_out(staging, local, fb)

        expected = torch.zeros(sum(sizes), hidden)
        expected[:2] = local
        self.assertIs(out, staging)
        torch.testing.assert_close(out, expected + 1.0)

    def test_in_place_wrapper_still_fills_the_caller_buffer(self):
        """``dp_gather_partial`` keeps its in-place contract for every caller.

        This is what makes C1 a no-op for the models that did not opt into the
        returning variant: the copy back into ``global_tokens`` is still there
        when (and only when) the collective handed back a different tensor.
        """
        from sglang.srt.layers.dp_attention import dp_gather_partial

        sizes = [2, 2]
        hidden = 3

        def all_reduce(x):
            return x + 5.0

        local = torch.randn(sizes[1], hidden)
        buf = torch.randn(sum(sizes), hidden)
        fb = self._forward_batch(sizes)
        with _dp_layout(sizes, 1, is_max_len=False):
            with self._sum_len_all_reduce(all_reduce):
                dp_gather_partial(buf, local, fb)

        expected = torch.zeros(sum(sizes), hidden)
        expected[2:] = local
        torch.testing.assert_close(buf, expected + 5.0)


def _reference_extend_conv(
    qk: torch.Tensor,
    lag_now: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    lag_state: torch.Tensor,
    seq_lens: List[int],
    slot_ids: List[int],
    has_prefix: List[bool],
    total_padding: int,
    groups: int,
):
    """Per-request torch reference for the fused prefill conv.

    Deliberately the naive formulation -- one ``F.conv1d`` per request with an
    explicitly built left pad -- rather than a copy of ``cca_extend``, so the two
    agree only if the varlen indexing is right.
    """
    qk_out = torch.empty_like(qk)
    lag_out = torch.empty_like(lag_now)
    new_conv_state = conv_state.clone()
    new_lag_state = lag_state.clone()

    start = 0
    for i, seq_len in enumerate(seq_lens):
        end = start + seq_len
        slot = slot_ids[i]
        cur = qk[start:end].transpose(0, 1).unsqueeze(0)  # [1, C, S]
        if has_prefix[i]:
            left = conv_state[slot].unsqueeze(0).to(cur.dtype)
        else:
            left = cur.new_zeros((1, cur.shape[1], total_padding))
        padded = torch.cat([left, cur], dim=-1)
        out = torch.nn.functional.conv1d(padded, weight, bias, groups=groups)
        qk_out[start:end] = out.squeeze(0).transpose(0, 1)
        new_conv_state[slot] = (
            padded[..., -total_padding:].squeeze(0).to(conv_state.dtype)
        )

        lag_cur = lag_now[start:end]
        if has_prefix[i]:
            first = lag_state[slot].squeeze(-1).to(lag_cur.dtype).unsqueeze(0)
        else:
            first = lag_cur.new_zeros((1, lag_cur.shape[-1]))
        lag_out[start:end] = torch.cat([first, lag_cur[:-1]], dim=0)
        new_lag_state[slot] = lag_cur[-1].unsqueeze(-1).to(lag_state.dtype)
        start = end

    return qk_out, lag_out, new_conv_state, new_lag_state


class TestCCAFusedPrefillConv(CustomTestCase):
    """Fused varlen prefill conv vs the per-request torch reference.

    The fused kernel resolves each token's request, start offset, pool slot and
    prefix flag from device tensors instead of a host loop. A token attributed to
    the wrong request reads another request's conv history, and nothing
    downstream notices.
    """

    GROUPS = 3
    CG = 16  # tl.dot needs a tile at least 16 wide
    # The lag stream is the projected val_proj2 value, not the hidden state, so
    # it is narrower than the conv channel count -- keep the test shape that way
    # too rather than reusing a hidden_size-shaped stream.
    LAG_DIM = 8
    TOTAL_PADDING = 2

    def _inputs(self, seq_lens: List[int], has_prefix: List[bool], seed: int = 0):
        torch.manual_seed(seed)
        channels = self.GROUPS * self.CG
        taps = self.TOTAL_PADDING + 1
        total = sum(seq_lens)
        num_slots = len(seq_lens) + 2
        dev = "cuda"
        dt = torch.bfloat16
        return dict(
            qk=torch.randn(total, channels, device=dev, dtype=dt),
            lag_now=torch.randn(total, self.LAG_DIM, device=dev, dtype=dt),
            weight=torch.randn(channels, self.CG, taps, device=dev, dtype=dt) * 0.1,
            bias=torch.randn(channels, device=dev, dtype=dt) * 0.1,
            conv_state=torch.randn(
                num_slots, channels, self.TOTAL_PADDING, device=dev, dtype=dt
            ),
            lag_state=torch.randn(num_slots, self.LAG_DIM, 1, device=dev, dtype=dt),
            seq_lens=seq_lens,
            has_prefix=has_prefix,
        )

    def _run(self, seq_lens, has_prefix, slot_ids=None, seed=0):
        from sglang.kernels.ops.attention import cca_conv1d

        args = self._inputs(seq_lens, has_prefix, seed)
        slot_ids = slot_ids or list(range(len(seq_lens)))
        offsets = [0]
        for s_len in seq_lens:
            offsets.append(offsets[-1] + s_len)
        cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        slots = torch.tensor(slot_ids, dtype=torch.int64, device="cuda")
        prefix = torch.tensor(has_prefix, dtype=torch.bool, device="cuda")

        ref_qk, ref_lag, ref_cs, ref_ls = _reference_extend_conv(
            args["qk"],
            args["lag_now"],
            args["weight"],
            args["bias"],
            args["conv_state"],
            args["lag_state"],
            seq_lens,
            slot_ids,
            has_prefix,
            self.TOTAL_PADDING,
            self.GROUPS,
        )

        conv_state = args["conv_state"].clone()
        lag_state = args["lag_state"].clone()
        self.assertTrue(
            cca_conv1d.covered(
                args["qk"],
                args["lag_now"],
                args["weight"],
                args["bias"],
                conv_state,
                lag_state,
                cu,
                prefix,
                slots,
                self.TOTAL_PADDING,
                self.GROUPS,
            )
        )
        got_qk, got_lag = cca_conv1d.cca_conv1d_fn(
            args["qk"],
            args["lag_now"],
            args["weight"],
            args["bias"],
            conv_state,
            lag_state,
            cu,
            prefix,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )
        return (got_qk, got_lag, conv_state, lag_state), (
            ref_qk,
            ref_lag,
            ref_cs,
            ref_ls,
        )

    def _assert_matches(self, got, ref):
        names = ("qk_out", "lag_prev", "conv_state", "lag_state")
        for name, g, r in zip(names, got, ref):
            torch.testing.assert_close(
                g.float(), r.float(), rtol=2e-2, atol=2e-2, msg=f"{name} mismatch"
            )

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_fresh_prefill_multi_request(self):
        got, ref = self._run([5, 3, 8], [False, False, False])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_resumed_prefix_reads_carried_state(self):
        """Mixed prefix flags: the halo taps must come from each slot's history."""
        got, ref = self._run([6, 4, 7], [True, False, True])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_chunk_shorter_than_the_conv_window(self):
        """A 1-token chunk with a prefix: the outgoing window is mostly carried.

        This is the branch where the new conv_state is not fully determined by the
        current chunk, so the tail write has to shift the incoming history rather
        than just copy the last tokens.
        """
        got, ref = self._run([1, 2, 1], [True, True, True])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_non_identity_slot_mapping(self):
        """Slots are pool indices, not request indices, and need not be ordered."""
        got, ref = self._run([4, 4, 4], [True, True, True], slot_ids=[3, 0, 2])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_request_longer_than_one_token_tile(self):
        """Tokens are tiled in blocks of 64; a request must span tiles correctly."""
        got, ref = self._run([200, 17], [True, False])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_single_request(self):
        """One request exercises the degenerate binary search (a single step)."""
        got, ref = self._run([37], [True])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_chunked_prefill_boundary_carries_the_projected_value(self):
        """Two chunks of one request must equal a single chunk over both.

        ``_boundary_state_kernel`` parks the chunk's last lag row in the pool slot
        and reads it back next chunk as the row preceding the first token. That
        row is the PROJECTED val_proj2 value; the raw hidden state there is wrong
        with no shape or dtype error. Splitting and comparing against the unsplit
        run pins the carry in both directions, which the single-shot
        ``test_resumed_prefix_reads_carried_state`` cannot: it seeds the slot with
        random values rather than the kernel's own output.
        """
        from sglang.kernels.ops.attention import cca_conv1d

        dev, dt = "cuda", torch.bfloat16
        channels = self.GROUPS * self.CG
        taps = self.TOTAL_PADDING + 1
        s0, s1 = 5, 7
        total = s0 + s1
        torch.manual_seed(31)
        qk = torch.randn(total, channels, device=dev, dtype=dt)
        lag_now = torch.randn(total, self.LAG_DIM, device=dev, dtype=dt)
        weight = torch.randn(channels, self.CG, taps, device=dev, dtype=dt) * 0.1
        bias = torch.randn(channels, device=dev, dtype=dt) * 0.1
        slots = torch.tensor([1], dtype=torch.int64, device=dev)

        def _fresh_pools():
            return (
                torch.zeros(4, channels, self.TOTAL_PADDING, device=dev, dtype=dt),
                torch.zeros(4, self.LAG_DIM, 1, device=dev, dtype=dt),
            )

        def _run(qk_c, lag_c, cs, ls, prefix):
            cu = torch.tensor([0, qk_c.shape[0]], dtype=torch.int32, device=dev)
            pf = torch.tensor([prefix], dtype=torch.bool, device=dev)
            self.assertTrue(
                cca_conv1d.covered(
                    qk_c,
                    lag_c,
                    weight,
                    bias,
                    cs,
                    ls,
                    cu,
                    pf,
                    slots,
                    self.TOTAL_PADDING,
                    self.GROUPS,
                )
            )
            return cca_conv1d.cca_conv1d_fn(
                qk_c,
                lag_c,
                weight,
                bias,
                cs,
                ls,
                cu,
                pf,
                slots,
                self.TOTAL_PADDING,
                self.GROUPS,
            )

        cs_one, ls_one = _fresh_pools()
        one_qk, one_lag = _run(qk, lag_now, cs_one, ls_one, False)

        cs_two, ls_two = _fresh_pools()
        a_qk, a_lag = _run(qk[:s0], lag_now[:s0], cs_two, ls_two, False)
        b_qk, b_lag = _run(qk[s0:], lag_now[s0:], cs_two, ls_two, True)

        torch.testing.assert_close(
            torch.cat([a_qk, b_qk]).float(), one_qk.float(), rtol=2e-2, atol=2e-2
        )
        # The lag stream is pure data movement, so require exact equality: any
        # boundary bug shows as a wrong row, never as rounding.
        self.assertTrue(torch.equal(torch.cat([a_lag, b_lag]), one_lag))
        self.assertTrue(torch.equal(cs_two, cs_one))
        self.assertTrue(torch.equal(ls_two, ls_one))
        # The carried row really is this chunk's last projected value.
        self.assertTrue(torch.equal(ls_two[1].squeeze(-1), lag_now[-1]))

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_no_lag_stream_leaves_the_conv_half_identical(self):
        """A rank with no lag drops two of the four launches.

        Ranks whose K heads all come from ``val_proj1`` never read the lag, so
        they pass ``None`` and the shifted copy plus the boundary kernel are
        skipped. The conv half must be untouched by that, and the lag pool must
        stay exactly as handed over.
        """
        from sglang.kernels.ops.attention import cca_conv1d

        args = self._inputs([6, 3], [True, False], seed=12)
        cu = torch.tensor([0, 6, 9], dtype=torch.int32, device="cuda")
        slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        prefix = torch.tensor([True, False], dtype=torch.bool, device="cuda")

        cs_ref = args["conv_state"].clone()
        ls_ref = args["lag_state"].clone()
        ref_qk, _ = cca_conv1d.cca_conv1d_fn(
            args["qk"],
            args["lag_now"],
            args["weight"],
            args["bias"],
            cs_ref,
            ls_ref,
            cu,
            prefix,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )

        cs_got = args["conv_state"].clone()
        ls_got = args["lag_state"].clone()
        self.assertTrue(
            cca_conv1d.covered(
                args["qk"],
                None,
                args["weight"],
                args["bias"],
                cs_got,
                None,
                cu,
                prefix,
                slots,
                self.TOTAL_PADDING,
                self.GROUPS,
            )
        )
        got_qk, got_lag = cca_conv1d.cca_conv1d_fn(
            args["qk"],
            None,
            args["weight"],
            args["bias"],
            cs_got,
            None,
            cu,
            prefix,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )
        self.assertIsNone(got_lag)
        self.assertTrue(torch.equal(got_qk, ref_qk))
        self.assertTrue(torch.equal(cs_got, cs_ref))
        # Never handed the lag pool, so it cannot have moved.
        self.assertTrue(torch.equal(ls_got, args["lag_state"]))

    def test_covered_rejects_a_half_specified_lag_pair(self):
        """One lag tensor without the other is refused, not guessed at.

        Guessing "no lag" would skip the chunk-boundary carry, and the next chunk
        would silently resume from a stale slot. This runs on CPU: ``covered()``
        rejects the pair before it ever reaches the device checks.
        """
        from sglang.kernels.ops.attention import cca_conv1d

        qk = torch.randn(4, self.GROUPS * self.CG)
        lag = torch.randn(4, self.LAG_DIM)
        weight = torch.randn(self.GROUPS * self.CG, self.CG, self.TOTAL_PADDING + 1)
        bias = torch.randn(self.GROUPS * self.CG)
        cs = torch.randn(3, self.GROUPS * self.CG, self.TOTAL_PADDING)
        ls = torch.randn(3, self.LAG_DIM, 1)
        cu = torch.tensor([0, 4], dtype=torch.int32)
        pf = torch.zeros(1, dtype=torch.bool)
        slots = torch.zeros(1, dtype=torch.int64)
        for lag_now, lag_state in ((lag, None), (None, ls)):
            with self.subTest(lag_now=lag_now is not None):
                self.assertFalse(
                    cca_conv1d.covered(
                        qk,
                        lag_now,
                        weight,
                        bias,
                        cs,
                        lag_state,
                        cu,
                        pf,
                        slots,
                        self.TOTAL_PADDING,
                        self.GROUPS,
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_covered_rejects_the_unfolded_two_stage_conv(self):
        """Without the folded weight there is nothing for this kernel to apply."""
        from sglang.kernels.ops.attention import cca_conv1d

        args = self._inputs([4], [False])
        cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        self.assertFalse(
            cca_conv1d.covered(
                args["qk"],
                args["lag_now"],
                None,
                args["bias"],
                args["conv_state"],
                args["lag_state"],
                cu,
                torch.zeros(1, dtype=torch.bool, device="cuda"),
                torch.zeros(1, dtype=torch.int64, device="cuda"),
                self.TOTAL_PADDING,
                self.GROUPS,
            )
        )


class TestZayaMoDReachability(CustomTestCase):
    """``ZayaRouter.fold_mod_reachability`` decides whether MOD is live.

    ``balancing_biases`` is added to a *softmax probability*, not a logit, so the
    skip slot's score is bounded above by ``1 + b_skip`` while every real
    expert's is at least ``b_j``. When ``1 + b_skip < max_j b_j`` no tie-breaking
    rule can pick the skip slot and the MOD path is dead work. Wrong in one
    direction it is silent extra launches; wrong in the other, silently wrong
    output.
    """

    def _router(self, *, use_mod=True, topk=1):
        from sglang.srt.models.zaya import ZayaRouter

        config = _make_swa_config(num_hidden_layers=4, swa_layers=[0, 0, 0, 0])
        config.zaya_use_mod = use_mod
        return ZayaRouter(
            config=config,
            num_moe_experts=4,
            moe_router_topk=topk,
            mlp_expansion=8,
            layer_id=1,
        )

    def test_skip_bias_of_minus_one_makes_mod_dead(self):
        # The shipped ZAYA1-74B layout: skip at -1.0, real biases straddling 0.
        router = self._router()
        with torch.no_grad():
            router.balancing_biases.copy_(
                torch.tensor([0.03, -0.06, 0.02, -0.01, -1.0])
            )
        router.fold_mod_reachability()
        self.assertFalse(router.mod_reachable)

    def test_all_real_biases_negative_keeps_mod_live(self):
        # 1 + b_skip == 0 is NOT below max_j b_j here, so the proof does not hold
        # and the branch must stay -- the check is conservative by design.
        router = self._router()
        with torch.no_grad():
            router.balancing_biases.copy_(
                torch.tensor([-0.03, -0.06, -0.02, -0.01, -1.0])
            )
        router.fold_mod_reachability()
        self.assertTrue(router.mod_reachable)

    def test_a_reachable_skip_bias_keeps_mod_live(self):
        # Distinct from the case above: there the real biases fail the test, here
        # the SKIP bias does. Catches a predicate that ignores b_skip entirely.
        router = self._router()
        with torch.no_grad():
            router.balancing_biases.copy_(torch.tensor([0.03, 0.01, 0.02, 0.0, 0.5]))
        router.fold_mod_reachability()
        self.assertTrue(router.mod_reachable)

    def test_boundary_is_strict(self):
        # 1 + b_skip exactly equal to max_j b_j must stay live: torch.argmax
        # tie-breaking is unspecified, so equality is not a proof of anything.
        router = self._router()
        with torch.no_grad():
            router.balancing_biases.copy_(torch.tensor([0.25, 0.1, 0.0, 0.0, -0.75]))
        router.fold_mod_reachability()
        self.assertTrue(router.mod_reachable)

    def test_mod_disabled_in_config_is_never_reachable(self):
        # Without the use_mod early return there is no skip slot at all, so
        # ``biases[-1]`` is a real expert's bias and the comparison reads garbage
        # -- which could mark MOD live on a model that has no skip path.
        router = self._router(use_mod=False)
        router.fold_mod_reachability()
        self.assertFalse(router.mod_reachable)

    def test_topk_above_one_keeps_mod_live(self):
        # The bound is on the *maximum*: it proves the skip slot never wins an
        # argmax, and says nothing about the runner-up a topk > 1 selection also
        # takes. Applying it there would let a selected skip id reach FusedMoE
        # unclamped, so the predicate must refuse to fire above top-1 even on
        # the bias layout that is dead at top-1.
        router = self._router(topk=2)
        with torch.no_grad():
            router.balancing_biases.copy_(
                torch.tensor([0.03, -0.06, 0.02, -0.01, -1.0])
            )
        router.fold_mod_reachability()
        self.assertTrue(router.mod_reachable)


# ---------------------------------------------------------------------------
# ZayaRouter: the routing tail with an unreachable MOD skip slot
# ---------------------------------------------------------------------------


class TestZayaRoutingWithoutReachableMoD(CustomTestCase):
    """``_routing_reference`` drops the skip slot's dead work, not its answers.

    On a checkpoint whose skip bias puts the slot out of reach, two of the
    tail's launches are provably dead: the clamp of the expert ids (argmax
    cannot return the skip index) and the cast of ``route_prob`` to the model
    dtype (the MOD blend is its only consumer). Both now ride on
    ``mod_reachable``. What must NOT change is the routing *decision*, so every
    test here pins the unreachable path against the reachable one on the same
    router and the same logits.
    """

    NUM_MOE_EXPERTS = 4

    def _router(self, *, biases, topk=1):
        from sglang.srt.models.zaya import ZayaRouter

        config = _make_swa_config(num_hidden_layers=4, swa_layers=[0, 0, 0, 0])
        config.zaya_use_mod = True
        config.moe_router_topk = topk
        router = ZayaRouter(
            config=config,
            num_moe_experts=self.NUM_MOE_EXPERTS,
            moe_router_topk=topk,
            mlp_expansion=8,
            layer_id=1,
        )
        with torch.no_grad():
            router.balancing_biases.copy_(torch.tensor(biases))
        router.fold_mod_reachability()
        return router

    # The shipped ZAYA1-74B layout (skip at -1.0) versus one where the slot can
    # still win.
    DEAD_BIASES = [0.03, -0.06, 0.02, -0.01, -1.0]
    LIVE_BIASES = [0.03, 0.01, 0.02, 0.0, 0.5]

    def _logits(self, rows=6, seed=17, skip_bonus=0.0):
        torch.manual_seed(seed)
        logits = torch.randn(rows, self.NUM_MOE_EXPERTS + 1)
        logits[:, -1] += skip_bonus
        return logits

    def _route(self, router, logits, model_dtype=torch.bfloat16):
        hs_next = torch.zeros(logits.shape[0], router.mlp_expansion)
        return router._routing_reference(logits, model_dtype, hs_next)

    def test_decision_is_identical_with_and_without_the_dead_work(self):
        router = self._router(biases=self.DEAD_BIASES)
        self.assertFalse(router.mod_reachable)
        # A skip logit large enough that it wins the raw softmax outright: the
        # -1.0 bias is the only thing keeping it out of the selection, which is
        # exactly the property the removal leans on.
        logits = self._logits(skip_bonus=6.0)

        without = self._route(router, logits)
        router.mod_reachable = True  # the pre-change chain: clamp + cast
        with_dead_work = self._route(router, logits)

        torch.testing.assert_close(
            without.moe_ids, with_dead_work.moe_ids, rtol=0, atol=0
        )
        torch.testing.assert_close(
            without.moe_weight, with_dead_work.moe_weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            without.skip_ids, with_dead_work.skip_ids, rtol=0, atol=0
        )

    def test_ids_are_in_range_by_construction(self):
        # No clamp runs any more, so the guarantee has to come from the
        # selection itself: the skip column is not offered to the argmax.
        router = self._router(biases=self.DEAD_BIASES)
        routing = self._route(router, self._logits(skip_bonus=50.0))
        self.assertEqual(routing.moe_ids.dtype, torch.int32)
        self.assertLess(int(routing.moe_ids.max()), self.NUM_MOE_EXPERTS)
        self.assertGreaterEqual(int(routing.moe_ids.min()), 0)
        # ``skip_ids`` aliases the ids' values when there is nothing to clamp,
        # which is the contract ZayaRouting documents for this case.
        torch.testing.assert_close(
            routing.skip_ids.to(torch.int32), routing.moe_ids, rtol=0, atol=0
        )

    def test_nan_logits_cannot_produce_an_out_of_range_expert(self):
        # An out-of-range id is not a wrong answer in FusedMoE, it is an
        # out-of-bounds write in moe_align_block_size. A NaN row makes every
        # comparison false and is the one way an argmax over the full width
        # could land on the skip column, so pin that the narrowed selection
        # holds even there.
        router = self._router(biases=self.DEAD_BIASES)
        logits = self._logits()
        logits[2, -1] = float("nan")
        routing = self._route(router, logits)
        self.assertLess(int(routing.moe_ids.max()), self.NUM_MOE_EXPERTS)

    def test_reachable_skip_still_clamps_and_casts(self):
        # The other direction: a checkpoint that really can route to the skip
        # slot must keep both. Wrong here is silent -- an unclamped skip id
        # indexing FusedMoE, and a fp32 route_prob widening the MOD blend.
        router = self._router(biases=self.LIVE_BIASES)
        self.assertTrue(router.mod_reachable)
        routing = self._route(router, self._logits(skip_bonus=50.0))
        skip_slot = self.NUM_MOE_EXPERTS
        self.assertTrue(bool((routing.skip_ids == skip_slot).any()))
        self.assertLess(int(routing.moe_ids.max()), self.NUM_MOE_EXPERTS)
        self.assertEqual(routing.route_prob.dtype, torch.bfloat16)
        self.assertEqual(routing.moe_weight.dtype, torch.float32)

    def test_unreachable_skip_leaves_route_prob_uncast(self):
        # The cast exists for the MOD blend alone. Dropping it is what saves the
        # launch, so pin that it is really gone rather than moved: the fp32
        # softmax output comes straight back out.
        router = self._router(biases=self.DEAD_BIASES)
        self.assertTrue(router.router_softmax_fp32)
        routing = self._route(router, self._logits())
        self.assertEqual(routing.moe_weight.dtype, torch.float32)
        self.assertEqual(routing.route_prob.dtype, torch.float32)

    def test_no_mod_at_all_is_unchanged(self):
        # ``use_mod=False`` has no skip column to narrow away, so the same code
        # path must degrade to the plain top-1 it always was.
        from sglang.srt.models.zaya import ZayaRouter

        config = _make_swa_config(num_hidden_layers=4, swa_layers=[0, 0, 0, 0])
        config.zaya_use_mod = False
        router = ZayaRouter(
            config=config,
            num_moe_experts=self.NUM_MOE_EXPERTS,
            moe_router_topk=1,
            mlp_expansion=8,
            layer_id=1,
        )
        router.fold_mod_reachability()
        self.assertEqual(router.num_experts, self.NUM_MOE_EXPERTS)
        logits = torch.randn(5, self.NUM_MOE_EXPERTS)
        routing = self._route(router, logits)
        expected = logits.softmax(dim=-1).to(torch.float32)
        expected = (expected + router.balancing_biases).argmax(dim=-1, keepdim=True)
        torch.testing.assert_close(
            routing.moe_ids, expected.to(torch.int32), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
