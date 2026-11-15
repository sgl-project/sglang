"""Tracked KDA graph calls obey FlashInfer's checkpoint/state dtype contract.

CPU only: FakeTensor models CUDA tensor metadata. The real SGLang graph path,
FlashInfer packed wrapper and eligibility checks run; GPU launches are mocked.
This does not validate kernel numerics or CUDA graph replay.
"""

import unittest
from types import SimpleNamespace
from unittest import TestCase, mock

import torch
from flashinfer import kda, kda_prefill_cute
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang.srt.layers.attention.linear import kda_prefill_graph as graph
from sglang.test.ci.ci_register import register_cuda_ci

# Needs the flashinfer package; GPU launches are mocked.
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class CheckpointDtypeTest(TestCase):
    def _run(self, dtype, *, flashinfer):
        with FakeTensorMode():

            def tensor(shape, dtype=torch.bfloat16):
                return torch.empty(shape, dtype=dtype, device="cuda:0")

            tokens, heads, dim = 8192, 32, 128
            width = heads * dim
            state = tensor((256, heads, dim, dim), dtype)
            layer = SimpleNamespace(
                layer_id=0,
                q_dim=width,
                k_dim=width,
                v_dim=width,
                head_q_dim=dim,
                head_k_dim=dim,
                head_v_dim=dim,
                conv_weights=tensor((3 * width, 4)),
                bias=None,
                A_log=tensor((heads,), torch.float32),
                dt_bias=tensor((heads, dim), torch.float32),
                lower_bound=-5.0,
            )
            meta = graph.KDAPrefillGraphMetadata.allocate(
                num_tokens=tokens,
                max_seqs_cap=64,
                track=True,
                cache_index_dtype=torch.int32,
                device=state.device,
                flashinfer_wrapper_cls=(
                    kda.RecurrentKDAPrefillWrapper if flashinfer else None
                ),
            )
            dispatcher = mock.Mock()
            dispatcher.extend.return_value = tensor((1, tokens, heads, dim))

            def launch(**kwargs):
                return kwargs["output"], None

            with (
                mock.patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ),
                mock.patch.object(kda, "get_compute_capability", return_value=(10, 3)),
                mock.patch.object(
                    kda_prefill_cute, "get_compute_capability", return_value=(10, 3)
                ),
                mock.patch.object(
                    kda_prefill_cute,
                    "_is_cute_dsl_kda_runtime_available",
                    return_value=True,
                ),
                mock.patch.object(
                    kda_prefill_cute, "_run_cute_dsl_kda_prefill", side_effect=launch
                ) as compiled,
                mock.patch.object(
                    graph,
                    "causal_conv1d_fn",
                    side_effect=lambda *args, output_split_size, **kwargs: (
                        tensor((3, tokens, width))
                        if output_split_size
                        else tensor((tokens, 3 * width)).transpose(0, 1)
                    ),
                ),
                mock.patch.object(graph, "kda_track_conv_window") as conv_track,
                mock.patch.object(graph, "kda_track_ssm_state") as state_track,
            ):
                result = graph.kda_extend_in_graph(
                    layer=layer,
                    meta=meta,
                    conv_pool=tensor((256, 3, 3 * width)),
                    ssm_states=state,
                    kernel_dispatcher=dispatcher,
                    mixed_qkv=tensor((tokens, 3 * width)),
                    a=tensor((1, tokens, width)),
                    b=tensor((1, tokens, heads)),
                )

            self.assertEqual(result.shape, (1, tokens, heads, dim))
            conv_track.assert_called_once()
            state_track.assert_called_once()
            checkpoint = state_track.call_args.kwargs["h_track"]
            self.assertIs(state_track.call_args.kwargs["ssm_pool"], state)
            if flashinfer:
                compiled.assert_called_once()
                call = compiled.call_args.kwargs
                self.assertIs(call["state_checkpoints"], checkpoint)
                self.assertIs(call["initial_state"], state)
                self.assertEqual(call["lower_bound"], -5.0)
                self.assertEqual(call["checkpoint_every_n_tokens"], 64)
                self.assertFalse(call["output_final_state"])
                self.assertEqual(checkpoint.dtype, state.dtype)
                dispatcher.extend.assert_not_called()
            else:
                compiled.assert_not_called()
                dispatcher.extend.assert_called_once()
                self.assertIs(
                    dispatcher.extend.call_args.kwargs["track_state"], checkpoint
                )
                self.assertEqual(checkpoint.dtype, torch.float32)

    def test_flashinfer_bf16_and_fp32_state_pools(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self._run(dtype, flashinfer=True)

    def test_triton_keeps_fp32_chunk_snapshots(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self._run(dtype, flashinfer=False)


if __name__ == "__main__":
    unittest.main()
