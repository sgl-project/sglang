"""Decode context parallel (DCP) on Intel XPU.

DCP shards the KV cache across TP ranks by token position
(``pos % dcp_size == dcp_rank``), so each rank attends over only its shard and
produces a partial attention output plus an LSE; the partials are merged across
ranks by log-sum-exp. On XPU the collectives run over xccl through
``torch.distributed`` (there is no pynccl), and the merge/index kernels are plain
Triton.

Three combinations are refused at startup rather than left to fail late, and
``TestXPUDCPServerArgs`` pins all three:

* ``intel_xpu`` as the attention backend. ``xpu_backend.py`` has no DCP path at
  all, and its kernels return no softmax LSE for the merge to weight by
  (``flash_attn_with_kvcache`` leaves it zero-filled, ``flash_mla_decode`` has
  none), so DCP there would be silently wrong rather than slow.
* MLA models. Triton's MLA KV write goes through the combined-row
  ``MLATokenToKVPool.set_kv_buffer``, which has no DCP owner-rule-aware kernel; DCP
  on XPU must use Triton, so there is nowhere else to go.
* GQA models with ``total_num_kv_heads > tp_size / dcp_size``. Every rank of a DCP
  group has to hold the same KV heads, and only ``models/qwen3_5.py`` shards its
  qkv projection that way; anything else builds fewer KV heads per rank than the
  pool was allocated for, and the masked KV write strides by the model's count and
  corrupts the cache silently.

docs/docs/hardware-platforms/xpu.mdx records what would lift each of them.

Usage:
python3 -m unittest test_xpu_dcp.TestXPUDCPLongContextDecode
"""

import unittest

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
    run_bench_one_batch,
)
from sglang.test.xpu.test_xpu_utils import write_results_to_github_step_summary

register_xpu_ci(est_time=1800, suite="nightly-xpu-4-gpu", nightly=True)

TP_SIZE = 4
DCP_SIZE = 2

# GLM-4.1V-9B-Thinking at a 4096-token prompt rather than a 1.5B model at 64: DCP
# only does anything once there is a KV cache worth sharding, so a short-prompt run
# on a small model measures model launch and not the feature. Of the KPI-sized dense
# models this is the one DCP admits at TP=4/DCP=2 -- 9B, 40 layers, 32 query heads
# and 2 KV heads, so total_num_kv_heads <= tp_size / dcp_size holds (see the module
# docstring). Reusing the constant rather than the literal: same checkpoint as the PP
# test, so the org rename lives in one place.
KPI_MODEL = DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP
INPUT_LEN = 4096
OUTPUT_LEN = 32
BATCH_SIZE = 4

# 2 KV heads over tp // dcp_size = 2 groups leaves one per rank, so this shape takes
# the no-permutation fast path in _dcp_gather_q. That is not a gap in coverage but a
# consequence of the admitted regime: for every model except Qwen3.5 the rule forces
# one KV head per rank, so the multi-KV-head permutation is only reachable through a
# DCP-aware qkv projection. TestDCPGatheredQHeadOrder covers it directly instead.

# Floor on DCP decode throughput as a fraction of the non-DCP baseline measured
# in the same run. DCP costs decode throughput rather than saving it -- measured
# 0.53 at this shape on 4x PCIe-connected Arc Pro B60 (54.37 -> 28.80 tok/s), and
# 0.77 at batch 1 / 32768 input, since the two extra collectives per layer outweigh
# the halved per-rank KV read. What DCP buys is context length, not speed. So this
# is a tripwire for the merge collapsing into something far worse (a serial or
# per-layer-synchronous path), deliberately loose enough not to fail on a host with
# a different interconnect.
DECODE_THROUGHPUT_FLOOR = 0.30


def _bench_args(dcp_size: int) -> list[str]:
    # run_bench_one_batch appends the auto-detected --device itself.
    #
    # 0.70, not the 0.85 a 9B model tolerates without DCP: the per-layer xccl
    # all-gather / reduce-scatter want Level Zero scratch from outside the static
    # pool, and at 0.85 the TP all-reduce dies with UR_RESULT_ERROR_OUT_OF_RESOURCES
    # during warmup decode. Both legs use it so the comparison stays fair.
    return [
        "--attention-backend",
        "triton",
        "--tp",
        str(TP_SIZE),
        "--dcp-size",
        str(dcp_size),
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.70",
        "--batch-size",
        str(BATCH_SIZE),
        "--input",
        str(INPUT_LEN),
        "--output",
        str(OUTPUT_LEN),
    ]


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestXPUDCPLongContextDecode(CustomTestCase):
    """DCP=2 vs non-DCP decode on a KPI model at 4096-token input, TP=4.

    Guards the platform enablement end to end -- initialize_model_parallel
    admitting XPU, the widened DCP paged allocator, the per-rank masked KV write,
    the sharded KV index build, and the xccl all-gather / reduce-scatter behind the
    LSE merge -- at a shape where the KV cache is big enough for sharding it to mean
    something.

    Both configurations are measured in the same run and the gate is their ratio,
    so it tracks the host it runs on rather than an absolute number baked in from
    one machine.
    """

    def test_dcp_long_context_decode(self):
        base_prefill, base_decode, base_latency = run_bench_one_batch(
            KPI_MODEL, _bench_args(1)
        )
        dcp_prefill, dcp_decode, dcp_latency = run_bench_one_batch(
            KPI_MODEL, _bench_args(DCP_SIZE)
        )
        floor = base_decode * DECODE_THROUGHPUT_FLOOR

        write_results_to_github_step_summary(
            {
                f"{KPI_MODEL} (dcp=1)": {
                    "server": f"tp={TP_SIZE} triton",
                    "client": f"bs={BATCH_SIZE} in={INPUT_LEN} out={OUTPUT_LEN}",
                    "output_throughput": base_decode,
                    "latency": base_latency,
                },
                f"{KPI_MODEL} (dcp={DCP_SIZE})": {
                    "server": f"tp={TP_SIZE} dcp={DCP_SIZE} triton",
                    "client": f"bs={BATCH_SIZE} in={INPUT_LEN} out={OUTPUT_LEN}",
                    "output_throughput": dcp_decode,
                    "output_throughput_threshold": round(floor, 2),
                    "latency": dcp_latency,
                },
            }
        )

        self.assertGreater(base_decode, 0, "non-DCP baseline decode must be > 0")
        self.assertGreaterEqual(
            dcp_decode,
            floor,
            f"DCP={DCP_SIZE} decode throughput {dcp_decode:.2f} tok/s fell below "
            f"{DECODE_THROUGHPUT_FLOOR:.0%} of the non-DCP baseline "
            f"{base_decode:.2f} tok/s at input {INPUT_LEN} "
            f"(decode latency {dcp_latency:.5f}s vs {base_latency:.5f}s, "
            f"prefill {dcp_prefill:.2f} vs {base_prefill:.2f} tok/s)",
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
    def _build(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN, **kwargs):
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(
            model_path=model_path,
            device="xpu",
            tp_size=TP_SIZE,
            mem_fraction_static=0.6,
            **kwargs,
        )
        # Construction only records what the caller asked for. Resolution
        # declares into a stash and never writes the fields, so the coercions
        # below are read with resolution_result rather than off the record.
        server_args.resolve_once()
        return server_args

    def test_intel_xpu_backend_rejected(self):
        """intel_xpu + DCP must fail loudly, not produce unnormalized output.

        The backend has no DCP path, and its decode kernels return no usable
        softmax LSE, so the merge would weight every rank's partial by a
        zero/absent LSE and silently emit wrong logits -- the failure mode a
        coarse accuracy gate would not catch. The message has to name both, since
        lifting the restriction needs an sgl-kernel-xpu change first.
        """
        for field in ("attention_backend", "prefill_attention_backend"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as cm:
                    self._build(dcp_size=DCP_SIZE, **{field: "intel_xpu"})
                message = str(cm.exception)
                self.assertIn("softmax LSE", message)
                self.assertIn("no DCP implementation", message)

    def test_intel_xpu_decode_backend_rejected(self):
        """A triton prefill does not rescue an intel_xpu decode."""
        with self.assertRaises(ValueError) as cm:
            self._build(
                dcp_size=DCP_SIZE,
                prefill_attention_backend="triton",
                decode_attention_backend="intel_xpu",
            )
        self.assertIn("softmax LSE", str(cm.exception))

    def test_mla_model_rejected(self):
        """MLA + DCP must be refused at startup, not deep in the model forward.

        MLATokenToKVPool.set_kv_buffer takes the combined row and has no
        owner-rule-aware kernel, so under DCP it asserts -- once from the Triton
        decode write and once from DeepSeek's MHA prefill write. Both fire minutes
        into a run, after the weights are loaded, which is why this is a config-time
        rejection. DCP on XPU cannot escape it: it must use triton, and the unified
        pool (the only holder of a resolved write loc) rejects triton under DCP.
        """
        with self.assertRaises(ValueError) as cm:
            self._build(
                model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                dcp_size=DCP_SIZE,
                attention_backend="triton",
                trust_remote_code=True,
            )
        self.assertIn("MLA models on Intel XPU", str(cm.exception))

    def test_kv_head_replication_rejected(self):
        """A GQA model with total_num_kv_heads > tp_size / dcp_size must be refused.

        DCP shards KV by position, so each rank of a DCP group needs the *same* KV
        heads: get_num_kv_heads -- which sizes the KV pool and the backend's
        num_kv_head -- divides by tp // dcp_size, and the attention module has to
        match by passing kv_tp_rank/kv_tp_size to QKVParallelLinear. Only
        models/qwen3_5.py does; everything else divides by the full tp_size and so
        builds fewer KV heads per rank than the pool holds.

        Nothing raises on that mismatch downstream, which is why the check belongs
        here: masked_set_kv_buffer_kernel strides by loc * H * D with H read off
        cache_k.shape, i.e. the model's count, so against a wider buffer every token
        is written at a fraction of its address and the upper heads never at all.
        Prefill and the first token are right and every decode step after is garbage
        -- Llama-3.1-8B at TP=4/DCP=2 scored 0.0 on GSM8K.

        Qwen3-0.6B stands in for the whole class (8 KV heads, so 4 per rank from the
        pool against the model's 2) because it is the cheapest config to resolve.
        """
        with self.assertRaises(ValueError) as cm:
            self._build(
                model_path="Qwen/Qwen3-0.6B",
                dcp_size=DCP_SIZE,
                attention_backend="triton",
            )
        message = str(cm.exception)
        self.assertIn("same KV heads", message)
        self.assertIn("4 KV heads per rank", message)
        self.assertIn("produces 2", message)
        # The message has to say what does work, not just what does not.
        self.assertIn("--dcp-size 1", message)

    def test_kpi_model_admitted(self):
        """The benchmarked model must pass the replication check it is chosen for.

        Pins KPI_MODEL against the guard so that swapping it for a model with more
        KV heads fails here, cheaply, rather than in the 30-minute bench above.
        """
        server_args = self._build(
            model_path=KPI_MODEL,
            dcp_size=DCP_SIZE,
            attention_backend="triton",
            trust_remote_code=True,
        )
        self.assertEqual(server_args.dcp_size, DCP_SIZE)

    def test_mla_model_allowed_without_dcp(self):
        """The rejection is DCP-scoped: MLA on XPU without DCP still builds."""
        server_args = self._build(
            model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            attention_backend="triton",
            trust_remote_code=True,
        )
        self.assertEqual(server_args.dcp_size, 1)

    def test_symm_mem_disabled(self):
        """Symmetric memory must be coerced off on XPU.

        It is a pynccl/ncclMemAlloc feature and XPU groups are always built with
        use_pynccl=False, so SymmetricMemoryContext would dereference a None
        comm. The DCP merge opts into it via use_symmetric_memory().
        """
        server_args = self._build(
            dcp_size=DCP_SIZE, attention_backend="triton", enable_symm_mem=True
        )
        self.assertFalse(resolution_result(server_args, "enable_symm_mem"))

    def test_decode_graph_disabled(self):
        """Decode graph capture must be off under DCP even when asked for.

        DCP issues per-layer collectives that are not capturable in an XPUGraph.
        Requesting 'full' explicitly is the only way decode capture turns on for
        XPU, so that is what this pins.
        """
        from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

        server_args = self._build(
            dcp_size=DCP_SIZE,
            attention_backend="triton",
            cuda_graph_backend_decode="full",
        )
        decided = resolution_result(server_args, "cuda_graph_config")
        self.assertEqual(decided[Phase.DECODE].backend, Backend.DISABLED)


if __name__ == "__main__":
    unittest.main()
