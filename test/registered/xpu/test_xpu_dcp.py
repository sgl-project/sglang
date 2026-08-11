"""Decode context parallel (DCP) on Intel XPU.

DCP shards the KV cache across TP ranks by token position
(``pos % dcp_size == dcp_rank``), so each rank attends over only its shard and
produces a partial attention output plus an LSE; the partials are merged across
ranks by log-sum-exp. On XPU the collectives run over xccl through
``torch.distributed`` (there is no pynccl), and the merge/index kernels are plain
Triton.

Only the ``triton`` attention backend supports DCP here: the ``intel_xpu``
kernels leave ``softmax_lse`` unwritten (``flash_attn_with_kvcache``) or have no
LSE output at all (``flash_mla_decode``), which would make the cross-rank merge
silently wrong. ``ServerArgs`` rejects that combination; ``test_intel_xpu_backend_rejected``
pins it.

Usage:
python3 -m unittest test_xpu_dcp.TestXPUDCPDecode.test_dcp_decode_runs
"""

import unittest

import torch

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
    is_in_ci,
    run_bench_one_batch,
)

register_xpu_ci(est_time=600, suite="nightly-xpu-4-gpu", nightly=True)

# Qwen2.5-1.5B has 2 KV heads. get_num_kv_heads shards them over tp // dcp_size
# groups, so at TP=4/DCP=2 each rank keeps one KV head and both ranks of a DCP
# group hold the same one -- the layout the LSE merge assumes.
TP_SIZE = 4
DCP_SIZE = 2


def _server_args(dcp_size: int) -> list[str]:
    args = [
        "--device",
        "xpu",
        "--attention-backend",
        "triton",
        "--tp",
        str(TP_SIZE),
        "--dcp-size",
        str(dcp_size),
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.6",
        "--batch-size",
        "1",
    ]
    if is_in_ci():
        args += ["--input", "64", "--output", "4"]
    return args


class TestXPUDCPDecode(CustomTestCase):
    """DCP=2 / TP=2 completes prefill + decode on XPU.

    Guards the platform enablement as a whole: ``initialize_model_parallel``
    admitting XPU, the widened DCP paged allocator, the per-rank masked KV write,
    the sharded KV index build, and the xccl all-gather / reduce-scatter used by
    the LSE merge. Any of those regressing takes the run down rather than
    perturbing a number, so a completed decode is the assertion.
    """

    def test_dcp_decode_runs(self):
        _, decode_throughput, _ = run_bench_one_batch(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN, _server_args(DCP_SIZE)
        )
        self.assertGreater(
            decode_throughput, 0, "XPU DCP decode throughput must be > 0"
        )


class TestDCPGatheredQHeadOrder(CustomTestCase):
    """The DCP query all-gather must keep each query head on its own KV head.

    ``all_gather(dim=1)`` yields a RANK-major head layout, but the decode/extend
    kernels map query head ``h`` to KV head ``h // (num_q_heads / num_kv_heads)``
    and the gather multiplies ``num_q_heads`` by ``dcp_size`` while the local KV
    shard keeps its head count. Under that inflated divisor a rank-major layout
    points every query head at the wrong KV head -- wrong logits, no crash.
    ``_dcp_gather_q`` transposes to KV-head-major to restore the mapping and
    ``_dcp_ungather_heads`` inverts it.

    Regression guard: before the fix, a GQA run (Qwen3 tp=2 -> 2 KV heads/rank)
    produced attention output diverging by O(1) from each rank's standalone
    result. These cases fail on the rank-major layout and pass on the fix.
    """

    @classmethod
    def setUpClass(cls):
        if not (torch.cuda.is_available() or is_xpu()):
            raise unittest.SkipTest("CUDA or XPU required for the Triton kernels")
        cls.device = "cuda" if torch.cuda.is_available() else "xpu"

    def _stub_backend(self, *, q_heads_per_rank, kv_heads, dcp_size):
        """A TritonAttnBackend with only the fields the head helpers read.

        Built via __new__ because __init__ needs an initialized process group
        and a ModelRunner; the permutation logic depends on nothing else.
        """
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.dcp_size = dcp_size
        backend.num_kv_head = kv_heads
        backend.num_head = q_heads_per_rank * dcp_size
        backend.dcp_q_per_kv_head = q_heads_per_rank // max(1, kv_heads)
        return backend

    def _assert_gather_matches_standalone(
        self, *, q_heads_per_rank, kv_heads, dcp_size
    ):
        from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

        torch.manual_seed(0)
        device, head_dim, batch, seq = self.device, 128, 2, 24
        backend = self._stub_backend(
            q_heads_per_rank=q_heads_per_rank, kv_heads=kv_heads, dcp_size=dcp_size
        )

        k_buffer = torch.randn(
            seq * batch, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        v_buffer = torch.randn_like(k_buffer)
        kv_indptr = torch.tensor(
            [seq * i for i in range(batch + 1)], device=device, dtype=torch.int32
        )
        kv_indices = torch.arange(seq * batch, device=device, dtype=torch.int32)

        def attend(q):
            max_splits, heads = 4, q.shape[1]
            logits = torch.empty(
                batch, heads, max_splits, head_dim, device=device, dtype=torch.float32
            )
            lse = torch.full(
                (batch, heads, max_splits),
                -float("inf"),
                device=device,
                dtype=torch.float32,
            )
            out = torch.empty(
                batch, heads, head_dim, device=device, dtype=torch.float32
            )
            splits = torch.full((batch,), max_splits, device=device, dtype=torch.int32)
            decode_attention_fwd(
                q,
                k_buffer,
                v_buffer,
                out,
                kv_indptr,
                kv_indices,
                logits,
                lse,
                splits,
                max_splits,
                head_dim**-0.5,
                1.0,
                1.0,
            )
            return out, torch.logsumexp(lse, dim=-1)

        per_rank_q = [
            torch.randn(
                batch, q_heads_per_rank, head_dim, device=device, dtype=torch.bfloat16
            )
            for _ in range(dcp_size)
        ]
        # Ground truth: each rank attending over its own KV shard alone.
        standalone = [attend(q) for q in per_rank_q]

        class _ConcatGroup:
            """Stands in for the DCP GroupCoordinator: all_gather == concat."""

            def __init__(self, tensors):
                self.world_size = len(tensors)
                self._tensors = tensors

            def all_gather(self, tensor, dim):
                return torch.cat(self._tensors, dim=dim)

        gathered_q = backend._dcp_gather_q(per_rank_q[0], _ConcatGroup(per_rank_q))
        out, lse = attend(gathered_q)
        out = backend._dcp_ungather_heads(out)
        lse = backend._dcp_ungather_heads(lse)

        for rank in range(dcp_size):
            block = slice(rank * q_heads_per_rank, (rank + 1) * q_heads_per_rank)
            # Same kernel, same KV, same queries -> must be bit-identical, not
            # merely close: any head mis-mapping shows up as an O(1) difference.
            self.assertTrue(
                torch.equal(out[:, block], standalone[rank][0]),
                f"output mismatch for rank {rank} at q={q_heads_per_rank} "
                f"kv={kv_heads} dcp={dcp_size} (max diff "
                f"{(out[:, block] - standalone[rank][0]).abs().max().item()})",
            )
            self.assertTrue(
                torch.equal(lse[:, block], standalone[rank][1]),
                f"LSE mismatch for rank {rank} at q={q_heads_per_rank} "
                f"kv={kv_heads} dcp={dcp_size}",
            )

    def test_gqa_two_kv_heads(self):
        """The shape that was silently wrong before the fix (Qwen3 tp=2)."""
        self._assert_gather_matches_standalone(
            q_heads_per_rank=8, kv_heads=2, dcp_size=2
        )

    def test_gqa_four_kv_heads_dcp4(self):
        """More KV heads and a wider DCP group exercise a different stride."""
        self._assert_gather_matches_standalone(
            q_heads_per_rank=16, kv_heads=4, dcp_size=4
        )

    def test_mqa_single_kv_head_unchanged(self):
        """kv_heads == 1 (MLA absorb / MQA) takes the no-permutation fast path.

        This case was already correct before the fix, so it pins that the
        reorder did not regress it.
        """
        self._assert_gather_matches_standalone(
            q_heads_per_rank=8, kv_heads=1, dcp_size=2
        )

    def test_non_power_of_two_group(self):
        """q_per_kv_head = 3 checks the reshape does not assume a power of two."""
        self._assert_gather_matches_standalone(
            q_heads_per_rank=12, kv_heads=4, dcp_size=2
        )


class TestXPUDCPServerArgs(CustomTestCase):
    """Config-time contracts for DCP on XPU (no GPU work, no server launch)."""

    @staticmethod
    def _build(**kwargs):
        from sglang.srt.server_args import ServerArgs

        return ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
            device="xpu",
            tp_size=TP_SIZE,
            mem_fraction_static=0.6,
            **kwargs,
        )

    def test_intel_xpu_backend_rejected(self):
        """intel_xpu + DCP must fail loudly, not produce unnormalized output.

        Its decode kernels return no usable softmax LSE, so the DCP merge would
        weight every rank's partial by a zero/absent LSE and silently emit wrong
        logits -- the failure mode a coarse accuracy gate would not catch.
        """
        with self.assertRaises(ValueError) as cm:
            self._build(
                dcp_size=DCP_SIZE,
                prefill_attention_backend="triton",
                decode_attention_backend="intel_xpu",
            )
        self.assertIn("softmax LSE", str(cm.exception))

    def test_symm_mem_disabled(self):
        """Symmetric memory must be coerced off on XPU.

        It is a pynccl/ncclMemAlloc feature and XPU groups are always built with
        use_pynccl=False, so SymmetricMemoryContext would dereference a None
        comm. The DCP merge opts into it via use_symmetric_memory().
        """
        server_args = self._build(
            dcp_size=DCP_SIZE, attention_backend="triton", enable_symm_mem=True
        )
        self.assertFalse(server_args.enable_symm_mem)

    def test_decode_graph_disabled(self):
        """Decode graph capture must be off under DCP even when asked for.

        DCP issues per-layer collectives that are not capturable in an XPUGraph.
        Requesting 'full' explicitly is the only way decode capture turns on for
        XPU, so that is what this pins.
        """
        from sglang.srt.server_args import Backend, Phase

        server_args = self._build(
            dcp_size=DCP_SIZE,
            attention_backend="triton",
            cuda_graph_backend_decode="full",
        )
        self.assertEqual(
            server_args.cuda_graph_config[Phase.DECODE].backend, Backend.DISABLED
        )


if __name__ == "__main__":
    unittest.main()
