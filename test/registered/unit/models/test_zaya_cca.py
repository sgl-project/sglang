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

All tests run on CPU with a tiny configuration so they stay fast and have no
GPU dependency. State is stored in a mock centralized pool that mirrors the
``HybridReqToTokenPool`` / ``MambaPool`` interface used at serving time.
"""

import os
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _ensure_dist_initialized() -> None:
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29632")
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)


# ---------------------------------------------------------------------------
# Mock centralized pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MockLayerCache:
    conv: List[torch.Tensor]
    temporal: torch.Tensor


class _MockReqToTokenPool:
    """Minimal stand-in for ``HybridReqToTokenPool`` providing the two methods
    that CCA calls: ``mamba2_layer_cache`` and ``get_mamba_indices``."""

    def __init__(self, pool_size: int, cca_config):
        in_out_ch = (cca_config.num_attention_heads + cca_config.num_key_value_heads) * cca_config.head_dim
        total_padding = (cca_config.cca_time0 - 1) + (cca_config.cca_time1 - 1)
        num_layers = len(cca_config.linear_layer_ids)

        self.conv_state = torch.zeros(num_layers, pool_size + 1, in_out_ch, total_padding)
        self.prev_hs_state = torch.zeros(num_layers, pool_size + 1, cca_config.hidden_size, 1)
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


@contextmanager
def _mock_pool_context(pool: _MockReqToTokenPool):
    """Install a mock ``ForwardContext`` whose ``req_to_token_pool`` is ``pool``."""
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        set_forward_context,
    )

    backend = SimpleNamespace(req_to_token_pool=pool, token_to_kv_pool=None)
    ctx = ForwardContext(attn_backend=backend)
    prev = set_forward_context(ctx)
    try:
        yield pool
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


def _make_tiny_config():
    from sglang.srt.configs.zaya import ZayaConfig

    return ZayaConfig(
        hidden_size=16,
        ffn_hidden_size=32,
        num_hidden_layers=2,
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


def _make_tiny_cca(seed: int = 0):
    from sglang.srt.models.zaya import CCA

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
        layer_id=0,
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


if __name__ == "__main__":
    unittest.main()
