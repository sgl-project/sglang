"""DSV4.1 low-latency indexer inputs: weights_proj and wk on ll_bf16 at 1-8 rows,
and the decode index query built on a side stream.

CPU: the row routing, the prepared query reaching the DeepGEMM decode data, the
side-stream joins, and the positions the backend packs with. GPU (SM100 family
with CuTe DSL): kernel parity at the served shapes, and the side-stream query
equals the stock serial one bit for bit, eagerly and replayed from a CUDA graph.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.gemm import small_gemm_bf16
from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
from sglang.srt.layers.attention.dsv4.v41_indexer.types import DecodeInputs
from sglang.srt.models.dsv41_low_latency import indexer as ll_indexer
from sglang.srt.models.dsv41_low_latency import ll_bf16
from sglang.srt.models.dsv41_low_latency.low_m_gemm import cutedsl_low_m_supported
from sglang.test.test_utils import CustomTestCase

HIDDEN = 5120
Q_LORA = 1024
HEAD_DIM = 512
INDEX_HEADS = 64


def _indexer(*, hidden=64, q_lora=32, owns_k=True, installed=True):
    config = SimpleNamespace(
        index_n_heads=INDEX_HEADS,
        index_head_dim=128,
        qk_rope_head_dim=64,
        index_topk=512,
        kv_source_layer_ids=[2] if owns_k else [],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        q_lora_rank=q_lora,
        hidden_size=hidden,
        rms_norm_eps=1e-6,
    )
    indexer = DeepseekV41Indexer(
        config, layer_id=2, head_dim=HEAD_DIM, quant_config=None, prefix="indexer"
    )
    if installed:
        with (
            patch.object(ll_indexer, "ll_bf16_supported", return_value=True),
            patch.object(ll_indexer, "cutedsl_low_m_supported", return_value=True),
        ):
            ll_indexer.install_projections(
                SimpleNamespace(compressor=None, indexer=indexer)
            )
    return indexer


class TestProjectionRouting(CustomTestCase):
    def setUp(self):
        # CPU tensors: keep the stock wk off the JIT CUDA GEMM in a CUDA image.
        patcher = patch.object(
            small_gemm_bf16, "can_use_n128k512_gemm", return_value=False
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_uninstalled_keeps_the_stock_gemms(self):
        indexer = _indexer(installed=False)
        with patch.object(ll_indexer, "_ll_bf16_gemm") as ll:
            k = indexer.forward_wk(torch.randn(6, HEAD_DIM, dtype=torch.bfloat16))
            w = indexer.head_weights_raw(torch.randn(6, 64, dtype=torch.bfloat16))
        ll.assert_not_called()
        self.assertEqual((k.dtype, k.shape), (torch.bfloat16, (6, 128)))
        self.assertEqual((w.dtype, w.shape), (torch.bfloat16, (6, INDEX_HEADS)))

    def test_switch_at_1_to_8_contiguous_rows_and_round_to_bf16(self):
        indexer = _indexer()

        def fake(x, w):
            return torch.full((x.shape[0], w.shape[0]), 1.0 + 2**-12)

        bf16 = torch.bfloat16
        with patch.object(ll_indexer, "_ll_bf16_gemm", side_effect=fake) as ll:
            k = indexer.forward_wk(torch.zeros(8, HEAD_DIM, dtype=bf16))
            w = indexer.head_weights_raw(torch.zeros(1, 64, dtype=bf16))
            # FP32 accumulation comes back rounded to the layer's BF16 output.
            for out in (k, w):
                self.assertEqual(out.dtype, bf16)
                self.assertTrue(torch.all(out == 1.0))
            self.assertEqual(ll.call_count, 2)
            indexer.forward_wk(torch.zeros(9, HEAD_DIM, dtype=bf16))
            indexer.forward_wk(torch.zeros(HEAD_DIM, 4, dtype=bf16).T)
            indexer.head_weights_raw(torch.zeros(9, 64, dtype=bf16))
            indexer.head_weights_raw(torch.zeros(0, 64, dtype=bf16))
            indexer.head_weights_raw(torch.zeros(64, 6, dtype=bf16).T)
        self.assertEqual(ll.call_count, 2)

    def test_narrow_weights_proj_dispatches_to_the_dot_product(self):
        for m in range(1, 9):
            self.assertEqual(
                ll_bf16.dispatch(M=m, K=HIDDEN, N=INDEX_HEADS).backend, "dotprod"
            )


def _decode_inputs(indexer, prepared_q=None) -> DecodeInputs:
    return DecodeInputs(
        indexer=indexer,
        layer_id=2,
        compress_ratio=2,
        freqs_cis=torch.zeros(16, 32, dtype=torch.complex64),
        x=torch.zeros(6, 64, dtype=torch.bfloat16),
        q_lora=torch.zeros(6, 32, dtype=torch.bfloat16),
        positions=torch.arange(6),
        req_rows=torch.zeros(6, dtype=torch.int64),
        paged_metadata=SimpleNamespace(compressed_page_size=64),
        is_verify=True,
        out_page_indices=torch.empty(6, 512, dtype=torch.int32),
        prepared_q=prepared_q,
    )


def _packed(fill):
    return (
        torch.full((6, INDEX_HEADS, 64), fill, dtype=torch.int8),
        torch.full((6, INDEX_HEADS), fill, dtype=torch.int32),
        torch.full((6, INDEX_HEADS), float(fill)),
    )


class TestPreparedIndexQuery(CustomTestCase):
    def _pool(self):
        pool = MagicMock()
        pool.get_index_k_with_scale_buffer.return_value = torch.zeros(4, 64 * 68)
        return pool

    def test_decode_data_takes_the_prepared_query(self):
        prepared = _packed(1)
        with patch.object(scoring, "_index_q_and_weights") as stock:
            data = scoring.get_deep_gemm_decode_data(
                _decode_inputs(_indexer(), prepared), self._pool()
            )
        stock.assert_not_called()
        self.assertEqual(data.q_fp4.data_ptr(), prepared[0].data_ptr())
        self.assertEqual(data.q_fp4.shape, (6, 1, INDEX_HEADS, 64))
        self.assertIs(data.weights, prepared[2])

    def test_without_it_the_scorer_projects_itself(self):
        with patch.object(
            scoring, "_index_q_and_weights", return_value=_packed(0)
        ) as stock:
            scoring.get_deep_gemm_decode_data(_decode_inputs(_indexer()), self._pool())
        stock.assert_called_once()

    def test_serial_projection_keeps_the_stock_order(self):
        indexer = _indexer()
        calls = []

        class _WqB(torch.nn.Module):
            def forward(self, q):
                calls.append("wq_b")
                return torch.zeros(6, 1), None

        indexer.wq_b = _WqB()
        with (
            patch.object(
                DeepseekV41Indexer,
                "head_weights_raw",
                side_effect=lambda x: calls.append("weights_proj") or "w",
            ),
            patch.object(scoring, "index_q_rope_pack", return_value="packed") as pack,
        ):
            got = scoring._index_q_and_weights(
                indexer=indexer, x="x", q_lora="q_lora", freqs_cis="f", positions="p"
            )
        self.assertEqual(got, "packed")
        self.assertEqual(calls, ["wq_b", "weights_proj"])
        self.assertEqual(pack.call_args.kwargs["head_weights_raw"], "w")


class _Stream:
    """Logs the joins and the work issued while it is current."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits {other.name}")


class TestIndexQSideStream(CustomTestCase):
    def setUp(self):
        self.log = []
        self.main = _Stream("main", self.log)
        self.side = _Stream("side", self.log)

        @contextlib.contextmanager
        def on(stream):
            self.log.append(f"enter {stream.name}")
            yield
            self.log.append(f"exit {stream.name}")

        for patcher in (
            patch.object(torch.cuda, "current_stream", return_value=self.main),
            patch.object(torch.cuda, "stream", side_effect=on),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _layer(self):
        indexer = SimpleNamespace(
            head_weights_raw=lambda x: self.log.append("weights_proj") or "raw",
            wq_b=lambda q_lora: self.log.append("wq_b") or ("q", None),
        )
        return SimpleNamespace(indexer=indexer)

    def test_begin_needs_an_indexer_and_a_prepared_query_reader(self):
        side_stream = ll_indexer.IndexQSideStream(self.side)
        reader = SimpleNamespace(low_ratio_decode_reads_prepared_q=True)
        torch_scorer = SimpleNamespace(low_ratio_decode_reads_prepared_q=False)
        self.assertIsNone(side_stream.begin(SimpleNamespace(indexer=None), "x", reader))
        self.assertIsNone(side_stream.begin(self._layer(), "x", torch_scorer))
        self.assertEqual(self.log, [])

    def test_side_stream_joins_both_ways(self):
        backend = SimpleNamespace(
            low_ratio_decode_reads_prepared_q=True,
            low_ratio_decode_index_q=lambda **kw: self.log.append("pack") or kw,
        )
        sources = _Stream("sources", self.log)
        layer = self._layer()
        index_q = ll_indexer.IndexQSideStream(self.side).begin(layer, "x", backend)
        self.assertIs(index_q.stream, self.side)
        prepared = index_q.finish(
            layer=layer,
            q_lora="q_lora",
            positions="pos",
            forward_batch="fb",
            attn_backend=backend,
            stream_sources=sources,
        )
        # x and q_lora are ready on main before the side stream reads them, and
        # the scoring stream waits for the packed query.
        self.assertEqual(
            self.log,
            [
                "side waits main",
                "enter side",
                "weights_proj",
                "exit side",
                "side waits main",
                "enter side",
                "wq_b",
                "pack",
                "exit side",
                "sources waits side",
            ],
        )
        self.assertEqual(prepared["q"], "q")
        self.assertEqual(prepared["head_weights_raw"], "raw")
        self.assertIs(prepared["layer"], layer)

    def test_install_uses_the_idle_alt_stream(self):
        # The low-ratio prepare runs kv on alt_streams[0] and sources on alt_streams[-1].
        streams = [_Stream(name, self.log) for name in ("kv", "index_q", "sources")]
        layer = self._layer()
        attn = SimpleNamespace(indexer=layer.indexer, alt_streams=streams)
        ll_indexer.install_index_q_side_stream(attn)
        reader = SimpleNamespace(low_ratio_decode_reads_prepared_q=True)
        index_q = attn.low_latency_index_q.begin(layer, "x", reader)
        self.assertEqual(index_q.stream.name, "index_q")
        for attn in (
            SimpleNamespace(indexer=None, alt_streams=streams),
            SimpleNamespace(indexer=object(), alt_streams=None),
        ):
            attn.low_latency_index_q = None
            ll_indexer.install_index_q_side_stream(attn)
            self.assertIsNone(attn.low_latency_index_q)


class TestBackendIndexQuery(CustomTestCase):
    def test_packs_with_the_positions_the_stock_scorer_uses(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        layer = SimpleNamespace(indexer="indexer", freqs_cis="freqs")
        live = torch.arange(6)
        hoisted = torch.arange(6) + 100
        for meta_pos, want in ((hoisted, hoisted), (torch.arange(8), live)):
            with self.subTest(hoisted=meta_pos.shape[0] == live.shape[0]):
                backend.forward_metadata = SimpleNamespace(
                    low_ratio_req_indices=torch.zeros_like(meta_pos),
                    low_ratio_pos_i64=meta_pos,
                )
                with (
                    patch.object(
                        deepseek_v4_backend, "index_q_rope_pack", return_value="packed"
                    ) as pack,
                    patch.object(deepseek_v4_backend, "token_req_indices"),
                ):
                    got = backend.low_ratio_decode_index_q(
                        layer=layer,
                        q="q",
                        head_weights_raw="raw",
                        positions=live,
                        forward_batch="fb",
                    )
                self.assertEqual(got, "packed")
                kwargs = pack.call_args.kwargs
                self.assertIs(kwargs["positions"], want)
                self.assertEqual(
                    (kwargs["indexer"], kwargs["q"], kwargs["head_weights_raw"]),
                    ("indexer", "q", "raw"),
                )
                self.assertEqual(kwargs["freqs_cis"], "freqs")


def _gpu_ok() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and ll_bf16.is_available()
        and cutedsl_low_m_supported()
    )


@unittest.skipUnless(_gpu_ok(), "needs an SM100-family GPU with CuTe DSL")
class TestLowLatencyIndexerGpu(CustomTestCase):
    def test_kernel_parity_at_the_served_shapes(self):
        # wk: N = 128, K = 512; weights_proj: N = 64 heads, K = hidden.
        for n, k in ((128, HEAD_DIM), (INDEX_HEADS, HIDDEN)):
            for m in range(1, 9):
                with self.subTest(n=n, m=m):
                    torch.manual_seed(n + m)
                    x = torch.randn(m, k, device="cuda").bfloat16()
                    w = (torch.randn(n, k, device="cuda") / k**0.5).bfloat16()
                    out = ll_bf16.ll_bf16_gemm(x, w)
                    self.assertEqual((out.dtype, out.shape), (torch.float32, (m, n)))
                    torch.testing.assert_close(
                        out, x.float() @ w.float().T, atol=1e-4, rtol=1e-4
                    )

    def test_installed_projections_match_the_stock_ones(self):
        torch.manual_seed(0)
        stock = _indexer(hidden=HIDDEN, installed=False).cuda()
        for p in stock.parameters():
            torch.nn.init.normal_(p, std=0.02)
        fast = _indexer(hidden=HIDDEN, installed=False).cuda()
        fast.load_state_dict(stock.state_dict())
        ll_indexer.install_projections(SimpleNamespace(compressor=None, indexer=fast))
        self.assertIsNotNone(fast.low_m_gemm)
        for m in (1, 6, 8):
            x = torch.randn(m, HIDDEN, device="cuda").bfloat16()
            latent = torch.randn(m, HEAD_DIM, device="cuda").bfloat16()
            # Only the K summation order differs: one BF16 ulp at most.
            torch.testing.assert_close(
                fast.head_weights_raw(x),
                stock.head_weights_raw(x),
                atol=1e-2,
                rtol=8e-3,
            )
            torch.testing.assert_close(
                fast.forward_wk(latent), stock.forward_wk(latent), atol=1e-2, rtol=8e-3
            )

    def test_side_stream_query_matches_the_serial_one(self):
        torch.manual_seed(0)
        indexer = _indexer(hidden=HIDDEN, q_lora=Q_LORA, installed=False).cuda()
        for p in indexer.parameters():
            torch.nn.init.normal_(p, std=0.02)
        m = 6
        x = torch.randn(m, HIDDEN, device="cuda").bfloat16()
        q_lora = torch.randn(m, Q_LORA, device="cuda").bfloat16()
        freqs = torch.polar(
            torch.ones(64, 32, device="cuda"), torch.rand(64, 32, device="cuda")
        )
        pos = torch.arange(m, device="cuda") + 40
        layer = SimpleNamespace(indexer=indexer, freqs_cis=freqs)

        def pack(*, layer, q, head_weights_raw, positions, forward_batch):
            torch.cuda._sleep(1_000_000)  # exposes a missing join
            return scoring.index_q_rope_pack(
                indexer=layer.indexer,
                q=q,
                head_weights_raw=head_weights_raw,
                freqs_cis=layer.freqs_cis,
                positions=positions,
            )

        backend = SimpleNamespace(
            low_ratio_decode_reads_prepared_q=True, low_ratio_decode_index_q=pack
        )
        side_stream = ll_indexer.IndexQSideStream(torch.cuda.Stream())
        sources = torch.cuda.Stream()

        def serial():
            return scoring._index_q_and_weights(
                indexer=indexer, x=x, q_lora=q_lora, freqs_cis=freqs, positions=pos
            )

        def split():
            main = torch.cuda.current_stream()
            sources.wait_stream(main)
            index_q = side_stream.begin(layer, x, backend)
            prepared = index_q.finish(
                layer=layer,
                q_lora=q_lora,
                positions=pos,
                forward_batch=None,
                attn_backend=backend,
                stream_sources=sources,
            )
            with torch.cuda.stream(sources):
                out = [t.clone() for t in prepared]
            main.wait_stream(sources)
            main.wait_stream(index_q.stream)
            return out

        expect = [t.clone() for t in serial()]
        for a, b in zip(split(), expect):
            self.assertTrue(torch.equal(a, b))

        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            split()
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = split()
        for _ in range(3):
            x.normal_()
            q_lora.normal_()
            expect = [t.clone() for t in serial()]
            graph.replay()
            torch.cuda.synchronize()
            for a, b in zip(graph_out, expect):
                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
