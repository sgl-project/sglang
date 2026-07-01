"""Numerical and state-cache correctness tests for the ZAYA1 CCA module.

The CCA per-request conv-state cache must satisfy the following invariants,
which are each exercised by a dedicated test case:

1. A single-chunk extend forward (no prefix) is numerically equivalent to the
   reference torch implementation that processes the whole sequence at once.
2. Splitting a sequence into one prefill of ``S0`` tokens and ``S1`` single-
   token decode steps produces the same q / k / v tensors as the equivalent
   single-chunk run.
3. A batched two-request decode for request 0 yields identical q / k / v to a
   single-request decode of request 0 at the same step.
4. Multi-request prefills update only the conv state and ``prev_hs`` slots for
   each request and leave unused slots zero.
5. A simulated tensor-parallel (TP=2) CCA produces per-rank q / k / v slices
   that match the corresponding head slices of a TP=1 reference, both for
   prefill (``_forward_extend``) and for decode (``_forward_decode``).

All tests run on CPU with a tiny configuration so they stay fast and have no
GPU dependency. State is stored in a mock centralized pool that mirrors the
``HybridReqToTokenPool`` / ``MambaPool`` interface used at serving time.
"""

import os
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _ensure_dist_initialized() -> None:
    """Set up a minimal single-rank gloo distributed environment plus the
    SGLang model-parallel groups (TP=1, PP=1, EP=1). The CCA module reads
    ``get_tensor_model_parallel_rank()`` / ``get_tensor_model_parallel_world_size()``
    inside ``__init__`` to size its head-parallel projections, so the world
    group and model parallel groups must both be initialized before any CCA
    construction.
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
        # Pass arguments as kwargs because ``ensure_model_parallel_initialized``
        # forwards positional ``backend`` into the ``attention_data_parallel_size``
        # slot of ``initialize_model_parallel``, which then explodes on
        # ``int // str``. Using kwargs avoids that footgun.
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
    """Minimal stand-in for ``HybridReqToTokenPool`` providing the two methods
    that CCA calls: ``mamba2_layer_cache`` and ``get_mamba_indices``.

    For TP-aware tests, ``tp_size`` controls the per-rank ``in_out_ch`` of the
    ``conv[0]`` state. ``conv[1]`` (prev_hs) is replicated and stays at full
    ``hidden_size``.
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
            num_layers, pool_size + 1, cca_config.hidden_size, 1
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

    The CCA module reaches the conv-state plumbing via
    ``get_attn_backend().conv_state_metadata(...)`` and runs its own conv
    kernel. This mock exposes that accessor over a ``_MockReqToTokenPool``,
    mirroring ``ShortConvAttnBackend``: the req -> slot mapping (and, for extend,
    its host ``.tolist()`` mirror) is resolved once per step and shared across
    all conv layers, while the decode path stays entirely on-device.
    """

    def __init__(self, pool: "_MockReqToTokenPool"):
        self.req_to_token_pool = pool
        self.token_to_kv_pool = None
        # Per-forward-step memoization keyed on the ForwardBatch identity,
        # mirroring ShortConvAttnBackend.init_forward_metadata.
        self._step_indices = {}  # id(forward_batch) -> device index tensor
        self._step_slot_ids = {}  # id(forward_batch) -> host list (extend only)

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
        indices = self._resolve_indices(forward_batch)
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
        prev_hs_state = layer_cache.conv[1]

        self.assertTrue(torch.any(conv_state[2] != 0))
        self.assertTrue(torch.any(conv_state[5] != 0))

        torch.testing.assert_close(
            prev_hs_state[2].squeeze(-1).to(torch.float32),
            hs0[-1].to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            prev_hs_state[5].squeeze(-1).to(torch.float32),
            hs1[-1].to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )

        for idx in (0, 1, 3, 4):
            self.assertTrue(torch.all(conv_state[idx] == 0))
            self.assertTrue(torch.all(prev_hs_state[idx] == 0))

    def test_mamba_indices_resolved_once_per_forward_step(self):
        """The req -> MambaPool-slot mapping is identical for every CCA layer in
        a step, so it (and its GPU->CPU ``.tolist()`` sync) must be resolved once
        per forward step and shared across layers, not recomputed per layer.

        Regression guard for the per-layer mamba-sync fix: two CCA layers driven
        by a single ForwardBatch must trigger exactly one ``get_mamba_indices``
        lookup and one host materialization for the whole step.
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
            cca.forward(
                torch.randn(3, config.hidden_size, dtype=torch.float32) * 0.1,
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[3],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(3, dtype=torch.int64),
                ),
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


class TestZayaCCATensorParallel(CustomTestCase):
    """Head-parallel TP equivalence:

    For each TP rank, the CCA's q / k / v output must equal the head slice of
    the TP=1 reference's output that corresponds to that rank's heads. This
    verifies that the grouped-mean step and ``conv_qk.1`` (groups = num_q_heads
    + num_k_heads) are correctly partitioned across heads with no cross-rank
    leakage.
    """

    TP_SIZE = 2

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def _slice_full_state_dict_into_rank(self, ref_cca, tp_cca, tp_rank: int):
        """Copy the reference's full weights into the per-rank CCA, using the
        per-parameter ``weight_loader`` that the CCA installs on its own
        parameters during ``__init__``. This mirrors what
        ``ZayaForCausalLM.load_weights`` does at serving time and is the
        only way TP correctness is exercised end-to-end.
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


if __name__ == "__main__":
    unittest.main()
