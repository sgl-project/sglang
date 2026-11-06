"""The radix track path must snapshot the prefill state in fp32, not in bf16.

`hybrid_linear_attn_backend._track_mamba_state_extend` restores checkpoints whose
depth is not on the FLA chunk grid from the per-chunk ``h`` tensor, which is
allocated with the activation dtype (bf16) and is "already rounded to the
activation dtype".  Snapshot-capable kernels avoid that by writing the fp32
chunk-boundary state into ``track_state``; the backend only allocates the buffer
when the extend kernel declares ``supports_track_state_snapshot``.

GDN's Triton kernel used to declare ``supports_packed_decode`` and
``supports_strided_target_verify_qkv`` but not ``supports_track_state_snapshot``,
and ``fla/chunk.py`` dropped the arguments on the way to
``chunk_gated_delta_rule_fwd_h``.  These tests pin both halves: the declaration
and the fp32 bits actually landing in the buffer.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

# FLA chunk size: the radix track grid snaps donated checkpoints to this step.
CHUNK_SIZE = 64


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGdnTrackStateSnapshot(CustomTestCase):
    NUM_HEADS = 2
    HEAD_K_DIM = 64
    HEAD_V_DIM = 64
    # 3 chunks in total, checkpoint taken at the end of the second one.
    TOKENS = 3 * CHUNK_SIZE
    TRACKED_TOKENS = 2 * CHUNK_SIZE
    TRACK_CHUNK_IDX = TRACKED_TOKENS // CHUNK_SIZE

    def _inputs(self, tokens):
        torch.manual_seed(0)
        device, dtype = "cuda", torch.bfloat16
        q = torch.randn(
            1, tokens, self.NUM_HEADS, self.HEAD_K_DIM, dtype=dtype, device=device
        )
        k = torch.randn(
            1, tokens, self.NUM_HEADS, self.HEAD_K_DIM, dtype=dtype, device=device
        )
        v = torch.randn(
            1, tokens, self.NUM_HEADS, self.HEAD_V_DIM, dtype=dtype, device=device
        )
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, tokens, self.NUM_HEADS, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(
            torch.randn(1, tokens, self.NUM_HEADS, dtype=dtype, device=device)
        )
        return q, k, v, g, beta

    def _pool(self):
        torch.manual_seed(1)
        # Non-zero initial state: the checkpoint must carry it through.
        return (
            torch.randn(
                2,
                self.NUM_HEADS,
                self.HEAD_V_DIM,
                self.HEAD_K_DIM,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.1
        )

    def _cu_seqlens(self, tokens):
        return torch.tensor([0, tokens], dtype=torch.long, device="cuda")

    def _reference(self):
        """fp32 state after TRACKED_TOKENS tokens, written back by inplace_update."""
        pool = self._pool()
        q, k, v, g, beta = self._inputs(self.TOKENS)
        indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        cut = self.TRACKED_TOKENS
        chunk_gated_delta_rule(
            q=q[:, :cut],
            k=k[:, :cut],
            v=v[:, :cut],
            g=g[:, :cut],
            beta=beta[:, :cut],
            initial_state=pool,
            initial_state_indices=indices,
            cu_seqlens=self._cu_seqlens(cut),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        return pool[indices[0]]

    def _tracked_buf(self, extend):
        pool = self._pool()
        q, k, v, g, beta = self._inputs(self.TOKENS)
        indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        buf = torch.zeros(
            1,
            self.NUM_HEADS,
            self.HEAD_V_DIM,
            self.HEAD_K_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        extend(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=indices,
            cu_seqlens=self._cu_seqlens(self.TOKENS),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            track_state=buf,
            track_chunk_idx=torch.tensor(
                [self.TRACK_CHUNK_IDX], dtype=torch.int32, device="cuda"
            ),
        )
        return buf[0]

    def test_chunk_kernel_writes_fp32_checkpoint(self):
        expected = self._reference()
        actual = self._tracked_buf(chunk_gated_delta_rule)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(
            torch.equal(actual, expected),
            "track_state must hold the fp32 chunk-boundary state bit-for-bit; "
            f"max|diff|={(actual - expected).abs().max().item():.3e}",
        )

    def test_triton_extend_declares_track_state_snapshot(self):
        self.assertTrue(
            TritonGDNKernel.supports_track_state_snapshot,
            "TritonGDNKernel.extend forwards track_state, so it must opt into "
            "the fp32 snapshot buffer; otherwise the backend keeps falling back "
            "to the bf16 h copy for unaligned checkpoints.",
        )

        expected = self._reference()

        def extend_via_kernel(**kwargs):
            kernel = TritonGDNKernel()
            return kernel.extend(
                kwargs.pop("q"),
                kwargs.pop("k"),
                kwargs.pop("v"),
                kwargs.pop("g"),
                kwargs.pop("beta"),
                ssm_states=kwargs.pop("initial_state"),
                cache_indices=kwargs.pop("initial_state_indices"),
                query_start_loc=kwargs.pop("cu_seqlens"),
                **kwargs,
            )

        actual = self._tracked_buf(extend_via_kernel)
        self.assertTrue(torch.equal(actual, expected))

    def test_extend_without_track_state_is_unchanged(self):
        """The optional arguments must not perturb the ordinary extend path."""
        pool = self._pool()
        q, k, v, g, beta = self._inputs(self.TOKENS)
        indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        TritonGDNKernel().extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=pool,
            cache_indices=indices,
            query_start_loc=self._cu_seqlens(self.TOKENS),
        )
        reference = self._pool()
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=reference,
            initial_state_indices=indices,
            cu_seqlens=self._cu_seqlens(self.TOKENS),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        self.assertTrue(torch.equal(pool[indices[0]], reference[indices[0]]))


if __name__ == "__main__":
    unittest.main()
