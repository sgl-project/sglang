"""Real pool/mixer/acceptance integration tests; no server or model load."""

import unittest
from types import SimpleNamespace

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
from test_mamba2_flashinfer_replay import check_numerics


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMamba2SpecReplay(CustomTestCase):
    @torch.inference_mode()
    def test_mixer_and_backend_commit(self):
        from sglang.kernels.ops.mamba.triton_ops.ssu_dispatch import (
            initialize_mamba_selective_state_update_backend,
        )
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateDType,
            Mamba2StateShape,
        )
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
        from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata
        from sglang.srt.mem_cache.memory_pool import MambaPool
        from sglang.srt.runtime_context import get_context, get_parallel

        context = get_context().override_server_args(
            enable_mamba_cache_stochastic_rounding=True,
            mamba_cache_philox_rounds=5,
        )
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        torch.manual_seed(910)
        initialize_mamba_selective_state_update_backend(
            SimpleNamespace(
                mamba_backend="flashinfer",
                enable_mamba_cache_stochastic_rounding=True,
                mamba_cache_philox_rounds=5,
            )
        )
        batch, width, layers = 4, 4, 2
        params = Mamba2CacheParams(
            shape=Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=8192,
                n_groups=8,
                num_heads=128,
                head_dim=64,
                state_size=128,
                conv_kernel=4,
            ),
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float16),
            layers=list(range(layers)),
        )

        def make_pool(replay):
            return MambaPool(
                size=16,
                spec_state_size=batch,
                cache_params=params,
                mamba_layer_ids=list(range(layers)),
                device="cuda",
                speculative_num_draft_tokens=width,
                speculative_eagle_topk=1,
                enable_mamba2_spec_replay=replay,
                mamba2_replay_dtype=torch.bfloat16,
            )

        baseline, replay = make_pool(False), make_pool(True)
        slots = torch.tensor([3, 1, 5, -1], device="cuda", dtype=torch.int32)
        tracks = torch.tensor([8, 9, 10, -1], device="cuda", dtype=torch.int32)
        last = torch.tensor([0, 1, 3, -1], device="cuda", dtype=torch.int32)
        track_steps = torch.tensor([0, -1, 1, -1], device="cuda", dtype=torch.int32)
        metadata = Mamba2Metadata(
            query_start_loc=torch.arange(
                0, batch * width + 1, width, device="cuda", dtype=torch.int32
            ),
            mamba_cache_indices=slots,
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=batch,
            is_target_verify=True,
            draft_token_num=width,
        )
        with get_parallel().override(
            tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
        ):
            mixers = [
                MambaMixer2(
                    params,
                    hidden_size=64,
                    use_conv_bias=True,
                    use_bias=False,
                    n_groups=8,
                ).to(device="cuda", dtype=torch.bfloat16)
                for _ in range(layers)
            ]

        # Keep this a kernel/lifecycle test: TP=1 output projection has no
        # collective work, so bypass only its distributed-group lookup.
        class LocalProjection(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.weight = original.weight

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight), None

        for mixer in mixers:
            mixer.out_proj = LocalProjection(mixer.out_proj)
            for p in mixer.parameters():
                p.normal_(std=0.1)
            mixer.A.data = -torch.rand(128, device="cuda", dtype=torch.float32) - 0.1
            mixer.dt_bias.fill_(-2)
        initial_state = torch.randn_like(baseline.mamba_cache.temporal) * 0.1
        initial_conv = torch.randn_like(baseline.mamba_cache.conv[0]) * 0.1
        for pool in (baseline, replay):
            pool.mamba_cache.temporal.copy_(initial_state)
            pool.mamba_cache.conv[0].copy_(initial_conv)

        def commit(pool):
            backend = SimpleNamespace(
                linear_attn_backend=SimpleNamespace(
                    _translate_mamba_indices=lambda x: x,
                    forward_metadata=metadata,
                    accept_lens_pool=None,
                    req_to_token_pool=SimpleNamespace(
                        mamba_pool=pool,
                        get_speculative_mamba2_params_all_layers=pool.get_speculative_mamba2_params_all_layers,
                    ),
                ),
                _update_ple_state_after_mtp_verify=lambda *a: None,
            )
            HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
                backend, last, tracks, track_steps, model=None
            )

        inputs = torch.empty(batch * width, 64, device="cuda", dtype=torch.bfloat16)
        inputs.normal_(std=0.1)

        def run(pool):
            layer_outputs = []
            for layer, mixer in enumerate(mixers):
                # Identical layer inputs isolate rollback correctness.
                # Chaining numerically different outputs would legitimately
                # change the next layer's convolution state.
                hidden, _, _ = mixer(
                    hidden_states=inputs,
                    layer_cache=pool.mamba2_layer_cache(layer),
                    metadata=metadata,
                    use_triton_causal_conv=True,
                )
                layer_outputs.append(hidden[:-width])
            commit(pool)
            return torch.stack(layer_outputs)

        # Production has one target pool. Its existing conv metadata cache is
        # single-entry: warm/capture each test pool separately and retain that
        # cache's tensors while both comparison graphs remain alive.
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            _conv_multi_meta_cache,
        )

        graphs, outputs, conv_metadata = [], [], []
        for pool in (baseline, replay):
            run(pool)
            conv_metadata.extend(_conv_multi_meta_cache.values())
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs.append(run(pool))
            graphs.append(graph)
        for pool in (baseline, replay):
            pool.mamba_cache.temporal.copy_(initial_state)
            pool.mamba_cache.conv[0].copy_(initial_conv)
        plan_stream, forward_stream = torch.cuda.Stream(), torch.cuda.Stream()
        forward_stream.wait_stream(torch.cuda.current_stream())
        for iteration in range(32):
            # Reproduce plan/forward event ordering without a scheduler/server.
            plan_stream.wait_stream(forward_stream)
            with torch.cuda.stream(plan_stream):
                inputs.normal_(std=0.1)
                last[:3].copy_((torch.arange(3, device="cuda") + iteration) % width)
                track_steps[:3].copy_(last[:3] // 2)
                if iteration % 4 == 0:
                    slots[:3].copy_(slots[:3].roll(1))
            forward_stream.wait_stream(plan_stream)
            with torch.cuda.stream(forward_stream):
                for graph in graphs:
                    graph.replay()
            torch.cuda.current_stream().wait_stream(forward_stream)
            check_numerics(
                "mixer_outputs",
                outputs[1],
                outputs[0],
                iteration=iteration,
            )
            check_numerics(
                "mixer_states",
                replay.mamba_cache.temporal,
                baseline.mamba_cache.temporal,
                iteration=iteration,
            )
            torch.testing.assert_close(
                baseline.mamba_cache.conv[0], replay.mamba_cache.conv[0], rtol=0, atol=0
            )
        self.assertEqual(replay.mamba_cache.mamba2_replay_pending.count_nonzero(), 0)
        self.assertEqual(replay.mamba_cache.mamba2_replay_ring_start.count_nonzero(), 0)

    def test_pool_allocation_accounting(self):
        from sglang.srt.configs.mamba2_spec_replay import Mamba2ReplaySizing
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateDType,
            Mamba2StateShape,
        )
        from sglang.srt.mem_cache.memory_pool import MambaPool

        params = Mamba2CacheParams(
            shape=Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=8192,
                n_groups=8,
                num_heads=128,
                head_dim=64,
                state_size=128,
                conv_kernel=4,
            ),
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float16),
            layers=[0, 2],
        )
        pool = MambaPool(
            size=32,
            spec_state_size=8,
            cache_params=params,
            mamba_layer_ids=[0, 2],
            device="cuda",
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
            enable_mamba2_spec_replay=True,
            mamba2_replay_dtype=torch.bfloat16,
        )
        self.assertIsNone(pool.mamba_cache.intermediate_ssm)
        costs = Mamba2ReplaySizing.from_params(
            params, layers=2, width=4, activation_bytes=2
        )
        self.assertEqual(round(pool.mem_usage * (1 << 30)), costs.bytes_for(32, 8, 4))
        self.assertEqual(
            pool.mamba2_layer_cache(1).mamba2_replay_x.shape, (33, 128, 8, 64)
        )
        self.assertEqual(
            {entry[0] for entry in pool._iter_transfer_state_entries()},
            {"conv", "temporal"},
        )
        # Copy/host restore preserve complete checkpoints, never stale records.
        source = torch.tensor([1], device="cuda", dtype=torch.int64)
        dest = torch.tensor([3], device="cuda", dtype=torch.int64)
        pool.mamba_cache.temporal[:, source] = 0.125
        pool.mamba_cache.conv[0][:, source] = 0.25
        pool.copy_from(source, dest)
        saved = pool.get_cpu_copy(source)
        pool.clear_slots(dest)
        pool.load_cpu_copy(saved, dest)
        torch.testing.assert_close(
            pool.mamba_cache.temporal[:, source],
            pool.mamba_cache.temporal[:, dest],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            pool.mamba_cache.conv[0][:, source],
            pool.mamba_cache.conv[0][:, dest],
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
