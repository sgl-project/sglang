import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.kda_input_pack import contiguous_kda_inputs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def make_inputs(tokens, heads=96, dim=128, dtype=torch.bfloat16):
    # The real K3 convolution and merged f_a/beta projection layouts.
    qkv = torch.empty(tokens, 3 * heads * dim, dtype=dtype, device="cuda")
    qkv.view(torch.uint8).random_(0, 256)
    q, k, v = (x.reshape(1, tokens, heads, dim) for x in qkv.chunk(3, -1))
    g = torch.zeros_like(q, memory_format=torch.contiguous_format)
    bfa = torch.empty(tokens, 128 + heads, dtype=dtype, device="cuda")
    bfa.view(torch.uint8).random_(0, 256)
    beta = bfa[:, 128:].unsqueeze(0)
    return q, k, v, g, beta


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestKDAInputPack(CustomTestCase):
    def assert_packed(self, inputs, outputs):
        for source, output in zip(inputs, outputs):
            self.assertTrue(output.is_contiguous())
            self.assertEqual(source.dtype, output.dtype)
            self.assertTrue(
                torch.equal(
                    source.contiguous().view(torch.uint8), output.view(torch.uint8)
                )
            )
            if source.is_contiguous():
                self.assertEqual(source.data_ptr(), output.data_ptr())

    def test_real_layout_bits_and_tails(self):
        # Random raw bits cover NaNs, infinities, subnormals and signed zero.
        for tokens, heads, dim, dtype in (
            (16384, 96, 128, torch.bfloat16),
            (257, 6, 128, torch.bfloat16),
            (17, 3, 65, torch.float32),
            (0, 3, 65, torch.bfloat16),
        ):
            with self.subTest(tokens=tokens, dtype=dtype):
                inputs = make_inputs(tokens, heads, dim, dtype)
                self.assert_packed(inputs, contiguous_kda_inputs(*inputs))

    def test_grouped_qkv_and_contiguous_inputs(self):
        inputs = make_inputs(257, 6)
        grouped = tuple(t.contiguous() for t in inputs[:4]) + inputs[4:]
        self.assert_packed(grouped, contiguous_kda_inputs(*grouped))
        dense = tuple(t.contiguous() for t in inputs)
        outputs = contiguous_kda_inputs(*dense)
        self.assertTrue(all(a is b for a, b in zip(dense, outputs)))

    def test_cuda_graph_replays_current_values(self):
        inputs = make_inputs(257, 6)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                contiguous_kda_inputs(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = contiguous_kda_inputs(*inputs)
        for value in (0.0, -0.0, float("inf"), float("nan")):
            for t in inputs:
                t.fill_(value)
            graph.replay()
            self.assert_packed(inputs, outputs)

    def test_flashinfer_output_and_state(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("FlashInfer KDA requires Blackwell")
        from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (
            FlashInferKDAPrefillKernel,
        )

        inputs = make_inputs(257, 6)
        for t in inputs:
            t.normal_(0, 0.1)
        dense = tuple(t.contiguous() for t in inputs)
        kernel = FlashInferKDAPrefillKernel()
        states = torch.zeros(2, 6, 128, 128, dtype=torch.bfloat16, device="cuda")
        kwargs = dict(
            cache_indices=torch.tensor([1], device="cuda"),
            query_start_loc=torch.tensor([0, 257], dtype=torch.int32, device="cuda"),
            A_log=torch.zeros(6, device="cuda"),
            dt_bias=torch.zeros(768, device="cuda"),
            lower_bound=-5.0,
        )
        expected_states = states.clone()
        expected = kernel.extend(*dense, ssm_states=expected_states, **kwargs)
        actual = kernel.extend(*inputs, ssm_states=states, **kwargs)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_states, states))

    def test_flashinfer_graph_adapter_replay(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("FlashInfer KDA requires Blackwell")
        from sglang.srt.layers.attention.linear.kda_flashinfer_prefill_graph import (
            FlashInferKDAPrefillGraph,
        )
        from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (
            FlashInferKDAPrefillKernel,
        )

        kernel = FlashInferKDAPrefillKernel()
        if kernel.prefill_graph_wrapper_cls is None:
            self.skipTest("FlashInfer does not provide the prefill graph wrapper")
        inputs = make_inputs(257, 6)
        for t in inputs:
            t.normal_(0, 0.1)
        states = torch.zeros(2, 6, 128, 128, dtype=torch.bfloat16, device="cuda")
        layer = SimpleNamespace(
            layer_id=0,
            A_log=torch.zeros(6, device="cuda"),
            dt_bias=torch.zeros(768, device="cuda"),
            lower_bound=-5.0,
        )
        adapter = FlashInferKDAPrefillGraph(
            wrapper_cls=kernel.prefill_graph_wrapper_cls,
            num_tokens=257,
            max_seqs=1,
            track=False,
            device="cuda",
        )
        adapter.cu_seqlens.copy_(torch.tensor([0, 257], device="cuda"))
        adapter.cache_indices.fill_(1)

        def run():
            return adapter.run(
                layer=layer,
                **dict(zip(("q", "k", "v", "g", "beta"), inputs)),
                ssm_states=states,
                h_track=None,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        for _ in range(2):
            for t in inputs:
                t.normal_(0, 0.1)
            states.zero_()
            expected_states = states.clone()
            expected = kernel.extend(
                *(t.contiguous() for t in inputs),
                ssm_states=expected_states,
                cache_indices=adapter.cache_indices,
                query_start_loc=adapter.cu_seqlens,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                lower_bound=layer.lower_bound,
            )
            graph.replay()
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(expected_states, states))


if __name__ == "__main__":
    unittest.main()
