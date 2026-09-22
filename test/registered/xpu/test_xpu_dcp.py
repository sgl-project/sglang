import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.attention_hook import (
    XPU_DECODE_MAX_Q_GROUP_SIZE,
    _xpu_fmha_emits_softmax_lse,
    xpu_dcp_decode_q_group_error,
)
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

KPI_MODEL = DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP
INPUT_LEN = 4096
OUTPUT_LEN = 32
BATCH_SIZE = 4


DECODE_THROUGHPUT_FLOOR = 0.30


def _kernel_lse(emits: bool):
    return patch(
        "sglang.srt.arg_groups.attention_hook._xpu_fmha_emits_softmax_lse",
        return_value=emits,
    )


def _bench_args(backend: str, dcp_size: int) -> list[str]:
    return [
        "--attention-backend",
        backend,
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


_SKIP_REASON = None
if not torch.xpu.is_available():
    _SKIP_REASON = "Intel XPU not available (torch.xpu.is_available() returned False)"
elif not _xpu_fmha_emits_softmax_lse():
    _SKIP_REASON = "installed sgl-kernel-xpu does not emit softmax_lse"


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class TestXPUDCPLongContextDecode(CustomTestCase):
    def test_dcp_long_context_decode(self):
        results = {}
        for backend, dcp_size in (("intel_xpu", 1), ("intel_xpu", DCP_SIZE)):
            prefill, decode, latency = run_bench_one_batch(
                KPI_MODEL, _bench_args(backend, dcp_size)
            )
            results[(backend, dcp_size)] = (prefill, decode, latency)

        base_prefill, base_decode, base_latency = results[("intel_xpu", 1)]
        xpu_prefill, xpu_decode, xpu_latency = results[("intel_xpu", DCP_SIZE)]
        floor = base_decode * DECODE_THROUGHPUT_FLOOR

        write_results_to_github_step_summary(
            {
                f"{KPI_MODEL} ({be} dcp={dcp})": {
                    "server": f"tp={TP_SIZE} dcp={dcp} {be}",
                    "client": f"bs={BATCH_SIZE} in={INPUT_LEN} out={OUTPUT_LEN}",
                    "input_throughput": pf,
                    "output_throughput": dec,
                    "latency": lat,
                }
                for (be, dcp), (pf, dec, lat) in results.items()
            }
        )

        self.assertGreater(base_decode, 0, "non-DCP intel_xpu baseline must be > 0")
        self.assertGreaterEqual(
            xpu_decode,
            floor,
            f"intel_xpu dcp={DCP_SIZE} decode throughput {xpu_decode:.2f} tok/s fell "
            f"below {DECODE_THROUGHPUT_FLOOR:.0%} of the non-DCP intel_xpu baseline "
            f"{base_decode:.2f} tok/s at input {INPUT_LEN} "
            f"(decode latency {xpu_latency:.5f}s vs {base_latency:.5f}s, "
            f"prefill {xpu_prefill:.2f} vs {base_prefill:.2f} tok/s)",
        )


class TestDCPGatheredQHeadOrder(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not (torch.cuda.is_available() or is_xpu()):
            raise unittest.SkipTest("CUDA or XPU required for the Triton kernels")
        cls.device = "cuda" if torch.cuda.is_available() else "xpu"

    def _stub_backend(self, *, q_heads_per_rank, kv_heads, dcp_size):
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
        standalone = [attend(q) for q in per_rank_q]

        class _ConcatGroup:
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
        self._assert_gather_matches_standalone(
            q_heads_per_rank=8, kv_heads=2, dcp_size=2
        )

    def test_gqa_four_kv_heads_dcp4(self):
        self._assert_gather_matches_standalone(
            q_heads_per_rank=16, kv_heads=4, dcp_size=4
        )

    def test_mqa_single_kv_head_unchanged(self):
        self._assert_gather_matches_standalone(
            q_heads_per_rank=8, kv_heads=1, dcp_size=2
        )

    def test_non_power_of_two_group(self):
        self._assert_gather_matches_standalone(
            q_heads_per_rank=12, kv_heads=4, dcp_size=2
        )


class TestDCPEmptyLSERows(CustomTestCase):
    def _rows(self, *, prefix_lens, seq_lens=None):
        from sglang.srt.layers.dcp import dcp_empty_lse_rows

        return dcp_empty_lse_rows(
            torch.tensor(prefix_lens, dtype=torch.int32),
            None if seq_lens is None else torch.tensor(seq_lens, dtype=torch.int32),
        )

    def test_decode_mask_is_already_per_token(self):
        rows = self._rows(prefix_lens=[7, 0, 3])
        self.assertEqual(rows.tolist(), [False, True, False])

    def test_extend_expands_over_the_varlen_layout(self):
        rows = self._rows(prefix_lens=[4, 0], seq_lens=[3, 2])
        self.assertEqual(rows.tolist(), [False, False, False, True, True])

    def test_one_flag_per_query_row(self):
        rows = self._rows(prefix_lens=[64, 0], seq_lens=[101, 7])
        self.assertEqual(rows.numel(), 108)
        self.assertFalse(rows.is_xpu or rows.is_cuda)

    def test_masked_fill_accepts_the_result(self):
        rows = self._rows(prefix_lens=[64, 0], seq_lens=[101, 7])
        lse = torch.zeros(108, 4)
        lse.masked_fill_(rows.unsqueeze(-1), -float("inf"))
        self.assertTrue(torch.isinf(lse[101:108]).all())
        self.assertFalse(torch.isinf(lse[:101]).any())

    def test_rank_owning_nothing_masks_every_row(self):
        rows = self._rows(prefix_lens=[0, 0], seq_lens=[5, 3])
        self.assertTrue(rows.all())


class TestDCPMetadataBuild(CustomTestCase):
    PAGE_SIZE = 1
    MAX_TOKENS = 64

    def _backend(self, *, dcp_rank, batch_size):
        from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend

        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.dcp_size = 2
        backend.dcp_rank = dcp_rank
        backend.page_size = self.PAGE_SIZE
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(
                batch_size * self.MAX_TOKENS, dtype=torch.int32
            ).view(batch_size, self.MAX_TOKENS)
        )
        return backend

    def _build(self, *, decode, prefix_lens, seq_lens=None, dcp_rank=0):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        lens = torch.tensor(prefix_lens, dtype=torch.int32)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
            batch_size=len(prefix_lens),
            req_pool_indices=torch.arange(len(prefix_lens)),
            seq_lens=lens,
            seq_lens_cpu=lens,
            extend_prefix_lens=lens,
            extend_prefix_lens_cpu=prefix_lens,
            extend_seq_lens=torch.tensor(seq_lens or [], dtype=torch.int32),
            extend_seq_lens_cpu=seq_lens or [],
        )
        backend = self._backend(dcp_rank=dcp_rank, batch_size=len(prefix_lens))
        backend._init_forward_metadata_dcp(batch)
        return backend.dcp_metadata

    def test_decode_mask_covers_one_row_per_request(self):
        dcp = self._build(decode=True, prefix_lens=[8, 1, 0])
        # rank 1 owns no token of a length-1 sequence, and none of an empty one
        self.assertEqual(dcp.empty_lse_rows.tolist(), [False, False, True])
        self.assertEqual(
            self._build(
                decode=True, prefix_lens=[8, 1, 0], dcp_rank=1
            ).empty_lse_rows.tolist(),
            [False, True, True],
        )

    def test_extend_mask_covers_every_query_row(self):
        dcp = self._build(decode=False, prefix_lens=[5, 0], seq_lens=[3, 2])
        self.assertEqual(dcp.empty_lse_rows.numel(), 5)
        self.assertEqual(dcp.empty_lse_rows.tolist(), [False] * 3 + [True] * 2)

    def test_mask_lands_on_the_batch_device(self):
        dcp = self._build(decode=True, prefix_lens=[8, 0])
        self.assertEqual(dcp.empty_lse_rows.device, torch.tensor(0).device)


class _StubModelConfig:
    def __init__(self, q_heads: int, kv_heads: int):
        self._q_heads = q_heads
        self._kv_heads = kv_heads
        self.hf_config = SimpleNamespace(architectures=["StubForCausalLM"])

    def get_max_num_attention_heads(self) -> int:
        return self._q_heads

    def get_num_kv_heads(self, tensor_parallel_size: int, dcp_size: int = 1) -> int:
        return max(1, self._kv_heads // max(1, tensor_parallel_size // dcp_size))


class TestXPUDCPDecodeQGroup(CustomTestCase):
    @staticmethod
    def _error(q_heads, kv_heads, attn_tp_size, dcp_size):
        return xpu_dcp_decode_q_group_error(
            _StubModelConfig(q_heads, kv_heads),
            attn_tp_size=attn_tp_size,
            dcp_size=dcp_size,
        )

    def test_group_at_the_cap_is_admitted(self):
        self.assertIsNone(self._error(32, 2, attn_tp_size=4, dcp_size=2))

    def test_group_over_the_cap_is_rejected(self):
        message = self._error(48, 1, attn_tp_size=4, dcp_size=2)
        self.assertIsNotNone(message)
        self.assertIn("24", message)
        self.assertIn("q_group_size", message)
        self.assertIn("--dcp-size 1", message)

    def test_dcp_can_break_a_config_that_works_without_it(self):
        for dcp_size, fits in ((1, True), (2, True), (4, False)):
            with self.subTest(dcp_size=dcp_size):
                message = self._error(64, 1, attn_tp_size=8, dcp_size=dcp_size)
                self.assertEqual(message is None, fits)

    def test_no_remedy_claimed_when_none_exists(self):
        message = self._error(64, 1, attn_tp_size=2, dcp_size=2)
        self.assertIsNotNone(message)
        self.assertIn("No --dcp-size fits", message)

    def test_kpi_model_sits_exactly_at_the_cap(self):
        self.assertIsNone(self._error(32, 2, attn_tp_size=TP_SIZE, dcp_size=DCP_SIZE))
        self.assertEqual((32 // TP_SIZE) * DCP_SIZE, XPU_DECODE_MAX_Q_GROUP_SIZE)


class TestXPUDCPServerArgs(CustomTestCase):
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
        server_args.resolve_once()
        return server_args

    def test_intel_xpu_rejected_without_kernel_lse(self):
        for field in (
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
        ):
            with self.subTest(field=field):
                with _kernel_lse(False):
                    with self.assertRaises(ValueError) as cm:
                        self._build(dcp_size=DCP_SIZE, **{field: "intel_xpu"})
                message = str(cm.exception)
                self.assertIn("softmax LSE", message)
                self.assertIn("sgl-kernel-xpu", message)

    def test_intel_xpu_decode_backend_rejected(self):
        with _kernel_lse(False):
            with self.assertRaises(ValueError) as cm:
                self._build(
                    dcp_size=DCP_SIZE,
                    prefill_attention_backend="triton",
                    decode_attention_backend="intel_xpu",
                )
        self.assertIn("softmax LSE", str(cm.exception))

    def test_intel_xpu_admitted_with_kernel_lse(self):
        for field in (
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
        ):
            with self.subTest(field=field):
                with _kernel_lse(True):
                    server_args = self._build(dcp_size=DCP_SIZE, **{field: "intel_xpu"})
                self.assertEqual(server_args.dcp_size, DCP_SIZE)

    def test_intel_xpu_allowed_without_dcp_on_old_kernel(self):
        with _kernel_lse(False):
            server_args = self._build(attention_backend="intel_xpu")
        self.assertEqual(server_args.dcp_size, 1)

    def test_mla_model_rejected(self):
        for backend in ("triton", "intel_xpu"):
            with self.subTest(backend=backend):
                with _kernel_lse(True):
                    with self.assertRaises(ValueError) as cm:
                        self._build(
                            model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                            dcp_size=DCP_SIZE,
                            attention_backend=backend,
                            trust_remote_code=True,
                        )
                self.assertIn("MLA models on Intel XPU", str(cm.exception))

    def test_kv_head_replication_rejected(self):
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
        self.assertIn("--dcp-size 1", message)

    def test_kpi_model_admitted(self):
        for backend in ("intel_xpu", "triton"):
            with self.subTest(backend=backend):
                with _kernel_lse(True):
                    server_args = self._build(
                        model_path=KPI_MODEL,
                        dcp_size=DCP_SIZE,
                        attention_backend=backend,
                        trust_remote_code=True,
                    )
                self.assertEqual(server_args.dcp_size, DCP_SIZE)

    def test_speculative_decoding_rejected_on_intel_xpu(self):
        for algorithm in ("EAGLE", "NEXTN"):
            with self.subTest(algorithm=algorithm):
                with _kernel_lse(True):
                    with self.assertRaises(ValueError) as cm:
                        self._build(
                            model_path=KPI_MODEL,
                            dcp_size=DCP_SIZE,
                            attention_backend="intel_xpu",
                            trust_remote_code=True,
                            speculative_algorithm=algorithm,
                        )
                message = str(cm.exception)
                self.assertIn("TARGET_VERIFY", message)
                self.assertIn("--attention-backend triton", message)

    def test_mla_model_allowed_without_dcp(self):
        server_args = self._build(
            model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            attention_backend="triton",
            trust_remote_code=True,
        )
        self.assertEqual(server_args.dcp_size, 1)

    def test_decode_graph_disabled(self):
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
