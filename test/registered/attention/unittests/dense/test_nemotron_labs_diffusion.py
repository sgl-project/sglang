import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.runner_utils.capture_mode as capture_mode
from sglang.srt.layers.attention.nemotron_labs_diffusion_backend import (
    NemotronLabsDiffusionAttentionBackend,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    build_replay_fb_view,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    MockModelRunner,
    build_dense_attention_fixture,
    dense_attention_reference_with_custom_mask,
    replace_backend,
    run_dense_fixture_eager,
    run_dense_forward,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


class TestNemotronLabsDiffusionAttention(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
        "FA4 paged attention requires SM100",
    )
    @torch.inference_mode()
    def test_draft_and_causal_graphs_refresh_lengths_and_request_slots(self):
        case = DenseAttentionCase(
            name="nemotron_diffusion_paged_graph",
            backend="fa4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=2,
            page_size=32,
            prefix_lens=(0, 31, 64, 96),
            extend_lens=(32, 32, 32, 32),
        )
        fixture = build_dense_attention_fixture(
            self, case, head_dim=128, hidden_size=128, max_context_len=256
        )
        with patch.object(
            MockModelRunner, "decode_num_tokens_per_req", return_value=32
        ):
            backend = NemotronLabsDiffusionAttentionBackend(fixture.runner)
        backend.is_linear_spec = True
        fixture = replace_backend(fixture, backend)
        fixture.actual_module.attn.attn_type = AttentionType.ENCODER_ONLY
        batch = fixture.forward_batch
        batch.forward_mode = ForwardMode.DLLM_EXTEND

        def reference(prefix_hidden, input_hidden, causal):
            masks = [
                (
                    torch.arange(prefix.shape[0] + 32, device="cuda")[None, :]
                    <= prefix.shape[0] + torch.arange(32, device="cuda")[:, None]
                    if causal
                    else torch.ones(
                        (32, prefix.shape[0] + 32), device="cuda", dtype=torch.bool
                    )
                )
                for prefix in prefix_hidden
            ]
            return dense_attention_reference_with_custom_mask(
                fixture.reference_module, case, prefix_hidden, input_hidden, masks
            )

        inputs = {"input_hidden": fixture.input_hidden.clone()}
        graphs, outputs = {}, {}
        # Capture uses a static batch; the label supplies its attention phase.
        batch.dllm_is_causal = False
        for causal in (False, True):
            fixture.actual_module.attn.attn_type = (
                AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY
            )
            expected = reference(fixture.prefix_hidden, fixture.input_hidden, causal)
            eager = run_dense_fixture_eager(fixture)
            torch.testing.assert_close(eager, expected, atol=3e-3, rtol=3e-2)
            with patch.object(
                capture_mode,
                "_capture_attention_variant",
                "verify" if causal else "draft",
            ):
                backend.init_forward_metadata_out_graph(batch, in_capture=True)
            for _ in range(3):
                with forward_context(ForwardContext(attn_backend=backend)):
                    run_dense_forward(fixture, batch, inputs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with (
                forward_context(ForwardContext(attn_backend=backend)),
                torch.cuda.graph(graph),
            ):
                outputs[causal] = run_dense_forward(fixture, batch, inputs)
            graphs[causal] = graph

        for slots, prefix_lens in (
            ((0, 1, 2, 3), (0, 31, 64, 96)),
            ((2, 0, 3, 1), (33, 0, 31, 0)),
        ):
            hidden = torch.cat(
                [fixture.input_hidden.reshape(4, 32, 128)[slot] for slot in slots]
            )
            prefixes = [
                fixture.prefix_hidden[slot][:length]
                for slot, length in zip(slots, prefix_lens)
            ]
            batch.req_pool_indices.copy_(torch.tensor(slots, device="cuda"))
            batch.seq_lens.copy_(
                torch.tensor([length + 32 for length in prefix_lens], device="cuda")
            )
            table = fixture.runner.req_to_token_pool.req_to_token
            batch.out_cache_loc.copy_(
                torch.cat(
                    [
                        table[slot, length : length + 32]
                        for slot, length in zip(slots, prefix_lens)
                    ]
                )
            )
            inputs["input_hidden"].copy_(hidden)
            for causal in (False, True, False):
                with self.subTest(causal=causal, slots=slots, prefix_lens=prefix_lens):
                    batch.dllm_is_causal = causal
                    fixture.actual_module.attn.attn_type = (
                        AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY
                    )
                    backend.init_forward_metadata(batch)
                    buffers = SimpleNamespace(
                        input_ids=batch.input_ids,
                        positions=batch.positions,
                        req_pool_indices=batch.req_pool_indices,
                        seq_lens=batch.seq_lens,
                        seq_lens_cpu=batch.seq_lens_cpu,
                        mamba_track_indices=None,
                    )
                    replay = build_replay_fb_view(
                        forward_batch=batch,
                        buffers=buffers,
                        bs=4,
                        raw_bs=4,
                        num_tokens=128,
                        global_num_tokens_cpu=None,
                        seq_len_fill_value=0,
                        capture_forward_mode=ForwardMode.DLLM_EXTEND,
                        is_encoder_decoder=False,
                    )
                    backend.init_forward_metadata_out_graph(replay)
                    graphs[causal].replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        outputs[causal],
                        reference(prefixes, hidden, causal),
                        atol=3e-3,
                        rtol=3e-2,
                    )


if __name__ == "__main__":
    unittest.main()
