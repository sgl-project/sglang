"""Native MLA prefix merging and breakable-graph replay regressions."""

from types import SimpleNamespace as NS
from unittest import main, skipUnless
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAPrefillMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")


def _backend():
    backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
    backend.data_type = torch.bfloat16
    backend.q_data_type = torch.bfloat16
    backend.workspace_buffer = torch.empty(
        64 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    backend._kv_shard_pool = None
    backend.forward_prefill_metadata = None
    backend.disable_chunked_prefix_cache = False
    return backend


def _extend_batch():
    return NS(
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.tensor([8, 6], device="cuda"),
        extend_prefix_lens=torch.zeros(2, dtype=torch.int64, device="cuda"),
        extend_prefix_lens_cpu=[0, 0],
        extend_seq_lens_cpu=[5, 3],
        batch_size=2,
    )


@skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillCpuLensTest(CustomTestCase):
    def test_native_prefix_merge_matches_dense_reference(self):
        from sglang.srt.layers.attention.merge_state import merge_state

        torch.manual_seed(19)
        tokens, prefix, heads = 32, 128, 12
        q = torch.randn(tokens, heads, 192, device="cuda", dtype=torch.bfloat16) * 0.5
        k = (
            torch.randn(
                prefix + tokens, heads, 192, device="cuda", dtype=torch.bfloat16
            )
            * 0.5
        )
        v = (
            torch.randn(
                prefix + tokens, heads, 128, device="cuda", dtype=torch.bfloat16
            )
            * 0.5
        )
        layer = NS(
            scaling=192**-0.5,
            tp_q_head_num=heads,
            tp_k_head_num=heads,
            head_dim=192,
            v_head_dim=128,
        )
        cpu_q_lens = torch.tensor([tokens], dtype=torch.int32)
        cum_q = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                backend = _backend()
                backend.data_type = dtype
                q_ref, k_ref, v_ref = (x.to(dtype).float() for x in (q, k, v))
                scores = torch.einsum("qhd,khd->hqk", q_ref, k_ref) * layer.scaling
                mask = (
                    torch.arange(prefix + tokens, device="cuda")[None, :]
                    > (prefix + torch.arange(tokens, device="cuda"))[:, None]
                )
                scores.masked_fill_(mask, float("-inf"))
                expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v_ref)
                states = []
                for start, end, causal in (
                    (0, prefix, False),
                    (prefix, prefix + tokens, True),
                ):
                    length = end - start
                    cpu_kv_lens = torch.tensor([length], dtype=torch.int32)
                    backend.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
                        tokens, cum_q, cpu_q_lens.cuda(), seq_lens_cpu=cpu_q_lens
                    )
                    batch = NS(
                        forward_mode=ForwardMode.EXTEND,
                        batch_size=1,
                        attn_attend_prefix_cache=not causal,
                        mha_return_lse=True,
                        prefix_chunk_idx=0,
                        prefix_chunk_has_zero_kv=[False],
                        prefix_chunk_seq_lens=[cpu_kv_lens.cuda()],
                        prefix_chunk_seq_lens_cpu=[cpu_kv_lens],
                        prefix_chunk_cu_seq_lens=[
                            torch.tensor([0, length], dtype=torch.int32, device="cuda")
                        ],
                        prefix_chunk_max_seq_lens=[length],
                    )
                    import flashinfer.prefill

                    with patch(
                        "flashinfer.prefill.trtllm_ragged_attention_deepseek",
                        wraps=flashinfer.prefill.trtllm_ragged_attention_deepseek,
                    ) as kernel:
                        output, lse = backend.forward_extend(
                            q,
                            k[start:end],
                            v[start:end],
                            layer,
                            batch,
                            save_kv_cache=False,
                        )
                    self.assertIs(kernel.call_args.kwargs["q_seq_lens_cpu"], cpu_q_lens)
                    expected_kv_lens = cpu_q_lens if causal else cpu_kv_lens
                    self.assertIs(
                        kernel.call_args.kwargs["kv_seq_lens_cpu"], expected_kv_lens
                    )
                    expected_lse = scores[:, :, start:end].logsumexp(-1).T
                    torch.testing.assert_close(
                        lse, expected_lse, rtol=0.001, atol=0.002
                    )
                    states.append((output, lse))
                merged, _ = merge_state(*states[0], *states[1])
                relative_rms = (
                    (merged.float() - expected).square().mean()
                    / expected.square().mean()
                ).sqrt()
                self.assertLess(
                    relative_rms.item(), 0.04 if dtype == torch.float8_e4m3fn else 0.005
                )

    def test_metadata_cpu_lengths_and_dcp(self):
        for dcp in (False, True):
            with (
                self.subTest(dcp=dcp),
                patch(
                    "sglang.srt.layers.attention.trtllm_mla_backend.get_parallel",
                    return_value=NS(dcp_enabled=dcp),
                ),
            ):
                backend = _backend()
                backend.init_forward_metadata(_extend_batch())
                lengths = backend.forward_prefill_metadata.seq_lens_cpu
                if dcp:
                    self.assertIsNone(lengths)
                else:
                    self.assertEqual(
                        (lengths.dtype, lengths.device.type, lengths.tolist()),
                        (torch.int32, "cpu", [5, 3]),
                    )


class BreakableGraphDispatchTest(CustomTestCase):
    def test_dispatch(self):
        from sglang.srt.models.deepseek_common import attention_backend_handler as h
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
            AttnForwardMethod,
        )

        mla, mha = AttnForwardMethod.MLA, AttnForwardMethod.MHA_CHUNKED_KV
        cases = (
            (h.handle_attention_trtllm_mla, False, False, mha),
            (h.handle_attention_trtllm_mla, True, False, mla),
            (h.handle_attention_trtllm_mla, False, True, mla),
            (h.handle_attention_tokenspeed_mla, False, False, mla),
        )
        with patch.object(h, "is_in_breakable_cuda_graph", return_value=True):
            for handler, disabled, piecewise, expected in cases:
                with patch.object(
                    h, "is_in_tc_piecewise_cuda_graph", return_value=piecewise
                ):
                    for prefixes in ([0, 0], [4096, 0]):
                        with self.subTest(
                            handler=handler.__name__,
                            disabled=disabled,
                            piecewise=piecewise,
                            prefixes=prefixes,
                        ):
                            batch = NS(
                                forward_mode=ForwardMode.EXTEND,
                                extend_prefix_lens_cpu=prefixes,
                            )
                            self.assertIs(
                                handler(
                                    NS(disable_chunked_prefix_cache=disabled), batch
                                ),
                                expected,
                            )

    def test_chunk_metadata_hook_fallback_only_for_tokenspeed(self):
        from sglang.srt.layers.attention import trtllm_mla_backend as trtllm
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )
        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        batch = NS(extend_prefix_lens_cpu=[0, 0])
        for cls, disabled, expect_super in (
            (TRTLLMMLABackend, False, False),
            (TRTLLMMLABackend, True, True),
            (TokenspeedMLABackend, False, True),
        ):
            backend = cls.__new__(cls)
            backend.disable_chunked_prefix_cache = disabled
            with (
                patch.object(trtllm, "is_in_breakable_cuda_graph", return_value=True),
                patch.object(
                    trtllm, "is_in_tc_piecewise_cuda_graph", return_value=False
                ),
                patch.object(
                    FlashInferMLAAttnBackend, "init_mha_chunk_metadata"
                ) as super_init,
            ):
                cls.init_mha_chunk_metadata(backend, batch)
            self.assertEqual(super_init.called, expect_super, cls.__name__)


@skipUnless(torch.cuda.is_available(), "CUDA required")
class BreakableChunkedPrefixReplayTest(CustomTestCase):
    def test_live_prefix_topology_and_padding_across_replays(self):
        from sglang.srt.layers.radix_attention import (
            _force_eager_attn,
            force_eager_attention,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            enable_breakable_cuda_graph,
        )
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            forward_mha as mha,
        )

        # Exercise the real core's prefix control flow with tiny deterministic
        # attention operations. The serving harness covers the actual kernels.
        class Attention(mha.DeepseekMHAForwardMixin):
            num_local_heads = 1
            v_head_dim = 2

            def attn_mha(self, q, k, v, batch, **kwargs):
                assert _force_eager_attn.get(), "nested RadixAttention break"
                assert batch.out_cache_loc.shape[0] == q.shape[0]
                assert batch.positions.shape[0] == q.shape[0]
                out = q + k + v
                return (out, out[..., 0]) if batch.mha_return_lse else out

            def _chunked_prefix_attn_mha(
                self, q, accum_output, accum_lse, forward_batch
            ):
                return accum_output + sum(forward_batch.extend_prefix_lens_cpu)

            def _apply_gated(self, output, gate):
                return output * gate

            def o_proj(self, output):
                # K3 consumes this Python attribute during capture, not replay.
                # Moving o_proj inside the eager callback would drop its gate.
                gate_input = self.gate_hidden_states
                self.gate_hidden_states = None
                if gate_input is not None:
                    output = output * gate_input
                return output, None

        def batch(lens, prefixes):
            result = NS(
                extend_seq_lens_cpu=lens,
                extend_prefix_lens_cpu=prefixes,
                num_prefix_chunks=None,
                global_num_token_non_padded_cpu=sum(lens),
                out_cache_loc=torch.arange(8, device="cuda"),
                positions=torch.arange(8, device="cuda"),
            )
            result.set_attn_attend_prefix_cache = lambda flag: setattr(
                result, "attn_attend_prefix_cache", flag
            )
            result.prepare_chunked_prefix_cache_info = lambda *a, **kw: setattr(
                result, "num_prefix_chunks", (max(prefixes) + 63) // 64
            )
            return result

        attn = Attention()
        capture_batch = batch([8], [0])
        context = NS(forward_batch=capture_batch)
        source = torch.ones(8, 1, 2, device="cuda")
        gate = torch.ones(8, 2, device="cuda")

        def run():
            attn.gate_hidden_states = gate
            q = source + 1
            return attn.forward_normal_chunked_kv_core(
                q, q, q, capture_batch, gate
            ).clone()

        with (
            patch.object(mha, "get_tc_piecewise_forward_context", return_value=context),
            patch.object(mha, "get_attn_backend", return_value=NS()),
            patch.object(mha, "resolve_attn_backend", return_value=NS()),
            patch.object(mha, "get_parallel", return_value=NS(dcp_enabled=False)),
            enable_breakable_cuda_graph(),
        ):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
                graph = BreakableCUDAGraph()
                with BreakableCUDAGraphCapture(graph, stream=stream):
                    output = run()
            torch.cuda.current_stream().wait_stream(stream)
            self.assertEqual(len(graph._break_fns), 1)
            for i, (lens, prefixes) in enumerate(
                (
                    ([8], [0]),
                    ([3, 2], [64, 0]),
                    ([2], [512]),
                    ([2, 2, 4], [128, 0, 320]),
                    ([8], [0]),
                )
            ):
                live = batch(lens, prefixes)
                context.forward_batch = live
                original_loc, original_positions = live.out_cache_loc, live.positions
                source.fill_(i + 2)
                gate.fill_(i + 1)
                graph.replay()
                n = sum(lens)
                q = (source + 1)[:n]
                with (
                    force_eager_attention(),
                    patch.object(mha, "is_in_breakable_cuda_graph", return_value=False),
                ):
                    reference_batch = batch(lens, prefixes)
                    reference_batch.out_cache_loc = reference_batch.out_cache_loc[:n]
                    reference_batch.positions = reference_batch.positions[:n]
                    attn.gate_hidden_states = gate[:n]
                    expected = attn.forward_normal_chunked_kv_core(
                        q, q, q, reference_batch, gate[:n]
                    )
                torch.testing.assert_close(output[:n], expected, atol=0, rtol=0)
                self.assertFalse(output[n:].any())
                self.assertIs(live.out_cache_loc, original_loc)
                self.assertIs(live.positions, original_positions)
                self.assertEqual(live.mha_return_lse, any(prefixes))

            with (
                patch.object(
                    attn,
                    "_forward_normal_chunked_kv_core",
                    side_effect=RuntimeError("test"),
                ),
                self.assertRaisesRegex(RuntimeError, "test"),
            ):
                graph.replay()
            self.assertIs(live.out_cache_loc, original_loc)
            self.assertIs(live.positions, original_positions)
            self.assertFalse(_force_eager_attn.get())


if __name__ == "__main__":
    main()
