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
GPU dependency.
"""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _ensure_dist_initialized() -> None:
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29632")
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)


def _make_forward_batch(
    *,
    is_decode: bool,
    extend_seq_lens_cpu,
    extend_prefix_lens_cpu,
    req_pool_indices,
    input_ids: torch.Tensor,
):
    """Build a minimal stand-in for ``ForwardBatch`` exposing just the fields
    that the CCA module accesses."""
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


def _make_tiny_cca(seed: int = 0):
    """Build a small CCA module on CPU with deterministic weights."""
    from sglang.srt.configs.zaya import ZayaConfig
    from sglang.srt.models.zaya import CCA

    torch.manual_seed(seed)
    config = ZayaConfig(
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

    # Re-initialize all learnable weights with bounded values to keep the
    # tests numerically stable in fp32 on CPU.
    with torch.no_grad():
        for p in cca.parameters():
            p.data.normal_(mean=0.0, std=0.05)
        # Zero the temperature so ``exp(temp) == 1`` whenever ``clamp_temp`` is
        # enabled, and the linear scaling stays 1 otherwise.
        cca.temp.data.zero_()

    return cca, config


class TestZayaCCA(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def test_single_chunk_matches_reference(self):
        """A single-chunk extend with empty prefix matches the no-state path."""
        cca, _ = _make_tiny_cca(seed=1)
        S = 5
        hs = torch.randn(S, cca.hidden_size, dtype=torch.float32) * 0.1
        # Use a second module so state is not shared between the two paths.
        cca_ref, _ = _make_tiny_cca(seed=1)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        # Reference: one-shot no-state forward over all ``S`` tokens.
        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        # SUT: prefill extend forward with empty prefix on a single request.
        fb = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[0],
            input_ids=torch.arange(S, dtype=torch.int64),
        )
        q, k, v = cca.forward(hs, fb)

        torch.testing.assert_close(q, q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k, k_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v, v_ref, atol=1e-5, rtol=1e-5)

    def test_prefill_then_decode_matches_full_sequence(self):
        """Prefill(S0) followed by ``S1`` single-token decode steps matches a
        one-shot reference over ``S0 + S1`` tokens.

        Verifies that both the ``conv_state`` pool and the ``prev_hs`` pool are
        threaded correctly across the prefill → decode boundary.
        """
        cca, _ = _make_tiny_cca(seed=2)
        cca_ref, _ = _make_tiny_cca(seed=2)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        S0, S1 = 4, 2
        S_total = S0 + S1
        torch.manual_seed(77)
        hs = torch.randn(S_total, cca.hidden_size, dtype=torch.float32) * 0.1

        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        fb_prefill = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S0],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[0],
            input_ids=torch.arange(S0, dtype=torch.int64),
        )
        q0, k0, v0 = cca.forward(hs[:S0], fb_prefill)

        # Drive ``S1`` decode steps, each for request id 0, and collect the
        # outputs to compare against the reference.
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

    def test_state_pool_lazy_resize(self):
        """The state pools start at size 1 and grow to at least
        ``max(req_pool_indices) + 1`` on the first non-trivial forward."""
        cca, _ = _make_tiny_cca(seed=3)
        self.assertEqual(cca.conv_state_pool.shape[0], 1)
        self.assertEqual(cca.prev_hs_pool.shape[0], 1)

        S = 3
        hs = torch.randn(S, cca.hidden_size, dtype=torch.float32) * 0.1
        fb = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[7],
            input_ids=torch.arange(S, dtype=torch.int64),
        )
        cca.forward(hs, fb)
        # Pool size must cover the highest seen index (7) and therefore be
        # at least 8.
        self.assertGreaterEqual(cca.conv_state_pool.shape[0], 8)
        self.assertGreaterEqual(cca.prev_hs_pool.shape[0], 8)

    def test_batched_decode_matches_single_decode(self):
        """A two-request batched decode of request 0 must produce the same
        q / k / v tensors as a single-request decode of request 0 at the same
        step.

        Guards against numerical drift in the decode conv path. An earlier
        implementation used a hand-rolled einsum that accumulated in the input
        dtype, while the prefill path delegates to ``nn.Conv1d`` whose backend
        kernels (cuDNN on NVIDIA, MIOpen on AMD) accumulate in fp32 even for
        bf16 inputs. Mixing the two led to silent per-token mismatches that
        only surfaced once more than one request was decoded in the same step.
        """
        cca_single, _ = _make_tiny_cca(seed=11)
        cca_batched, _ = _make_tiny_cca(seed=11)
        with torch.no_grad():
            cca_batched.load_state_dict(cca_single.state_dict())

        S0 = 4
        torch.manual_seed(202)
        hs0 = torch.randn(S0, cca_single.hidden_size, dtype=torch.float32) * 0.1
        hs1 = torch.randn(S0, cca_single.hidden_size, dtype=torch.float32) * 0.1
        decode0 = torch.randn(cca_single.hidden_size, dtype=torch.float32) * 0.1
        decode1 = torch.randn(cca_single.hidden_size, dtype=torch.float32) * 0.1

        # Single-request reference: prefill request 0 then one decode step on
        # request 0 alone.
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

        # Batched SUT: prefill both requests together, then decode one token
        # for each in a single batched step.
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
        own slots in the conv_state and prev_hs pools."""
        cca, _ = _make_tiny_cca(seed=4)

        S0, S1 = 3, 2
        hs0 = torch.randn(S0, cca.hidden_size, dtype=torch.float32) * 0.1
        hs1 = torch.randn(S1, cca.hidden_size, dtype=torch.float32) * 0.1
        hs = torch.cat([hs0, hs1], dim=0)

        fb = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S0, S1],
            extend_prefix_lens_cpu=[0, 0],
            req_pool_indices=[2, 5],
            input_ids=torch.arange(S0 + S1, dtype=torch.int64),
        )
        cca.forward(hs, fb)

        # Slot 2 must now hold the conv tail of request 0, slot 5 the tail
        # of request 1; both should be non-zero.
        self.assertTrue(torch.any(cca.conv_state_pool[2] != 0))
        self.assertTrue(torch.any(cca.conv_state_pool[5] != 0))

        # The prev_hs cache for each request must equal that request's last
        # input hidden_state.
        torch.testing.assert_close(
            cca.prev_hs_pool[2].to(torch.float32),
            hs0[-1].to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            cca.prev_hs_pool[5].to(torch.float32),
            hs1[-1].to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )

        # Slots covered by the lazy resize but not addressed by this batch
        # (here 0, 1, 3, 4) must stay zero-initialized.
        for idx in (0, 1, 3, 4):
            self.assertTrue(torch.all(cca.conv_state_pool[idx] == 0))
            self.assertTrue(torch.all(cca.prev_hs_pool[idx] == 0))


if __name__ == "__main__":
    unittest.main()
