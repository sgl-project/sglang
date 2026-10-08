"""Production SSM cache geometry, dynamic parent trees and accepted-prefix replay."""

import unittest
from unittest.mock import patch

import torch
from sglang.kernels.ops.mamba.triton_ops import mamba_ssm as ssm
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class TestStaticVerify(CustomTestCase):
    def test_prefix_rollback_and_live_graph(self):
        torch.set_num_threads(2)
        torch.manual_seed(8871)
        for batch in (1, 3):
            for tree in (False, True):
                with self.subTest(batch=batch, tree=tree):
                    self._check_batch(batch, tree)

    def _check_batch(self, batch, tree):
        state = torch.randn(5, 128, 64, 128, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(batch, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
        dt = torch.randn(batch, 4, 128, device="cuda", dtype=torch.bfloat16)
        dt = dt.unsqueeze(-1).expand(-1, -1, -1, 64)
        A = (-torch.rand(128, device="cuda"))[:, None, None].expand(-1, 64, 128)
        B = torch.randn(batch, 4, 8, 128, device="cuda", dtype=torch.bfloat16)
        C = torch.randn_like(B)
        D = torch.randn(128, device="cuda")[:, None].expand(-1, 64)
        bias = torch.randn_like(D[:, 0])[:, None].expand(-1, 64)
        indices = torch.arange(batch, device="cuda", dtype=torch.int32) + 1
        cache_indices = torch.arange(batch, device="cuda", dtype=torch.int64).flip(0)
        parents = torch.tensor([[-1, 0, 1, 2]] * batch, device="cuda")
        cache_shape = (batch, 4, 128, 64, 128)
        caches = [
            torch.zeros(cache_shape, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        outputs = [torch.empty_like(x) for _ in range(2)]

        def run(which):
            ssm.selective_state_update(
                state,
                x,
                dt,
                A,
                B,
                C,
                D,
                dt_bias=bias,
                dt_softplus=True,
                state_batch_indices=indices,
                out=outputs[which],
                disable_state_update=True,
                intermediate_states_buffer=caches[which],
                cache_steps=4,
                retrieve_parent_token=parents if tree else None,
                intermediate_state_indices=cache_indices,
            )

        graphs = []
        for which in (0, 1):
            with envs.SGLANG_SM120_MAMBA_STATIC_VERIFY.override(bool(which)):
                run(which)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(which)
                graphs.append(graph)
        for case in range(24):
            x.normal_()
            B.normal_()
            C.normal_()
            state.normal_()
            pattern = (
                [-1, 0, 1, 2],
                [-1, 0, 0, 1],
                [-1, -1, -1, -1],
                [-1, 0, 1, 0],
            )[case % 4]
            parents.copy_(torch.tensor([pattern] * batch, device="cuda"))
            indices.copy_(
                torch.arange(batch, device="cuda", dtype=torch.int32) + 1 + case % 2
            )
            if case % 5 == 0:
                indices[0] = -1
            cache_indices.copy_(
                torch.roll(torch.arange(batch, device="cuda"), case % batch)
            )
            snapshot = state.clone()
            for graph in graphs:
                graph.replay()
            torch.testing.assert_close(state, snapshot, atol=0, rtol=0)
            torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
            torch.testing.assert_close(caches[1], caches[0], atol=0, rtol=0)
            # Commit each legal accepted prefix and use that exact state
            # for three ordinary decode transitions after rejection.
            for accepted in range(1, 5):
                states = [snapshot.clone(), snapshot.clone()]
                for which in (0, 1):
                    for row in range(batch):
                        slot = int(indices[row])
                        if slot >= 0:
                            states[which][slot].copy_(
                                caches[which][int(cache_indices[row]), accepted - 1]
                            )
                for step in range(3):
                    decoded = []
                    for which in (0, 1):
                        out = torch.empty_like(x[:, step])
                        with envs.SGLANG_SM120_MAMBA_STATIC_VERIFY.override(
                            bool(which)
                        ):
                            ssm.selective_state_update(
                                states[which],
                                x[:, step],
                                dt[:, step],
                                A,
                                B[:, step],
                                C[:, step],
                                D,
                                dt_bias=bias,
                                dt_softplus=True,
                                state_batch_indices=indices,
                                out=out,
                            )
                        decoded.append(out)
                    torch.testing.assert_close(states[1], states[0], atol=0, rtol=0)
                    torch.testing.assert_close(decoded[1], decoded[0], atol=0, rtol=0)

    def test_dispatch_guard(self):
        # Inspect the real launch arguments; a successful fallback is not proof
        # that the specialized kernel ran.
        kernel = ssm._selective_scan_update_kernel
        calls = []

        class ObservedKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    calls.append(kwargs.get("STATIC_SHAPES_STRIDES"))
                    return kernel[grid](*args, **kwargs)

                return launch

        state = torch.zeros(2, 128, 64, 128, device="cuda", dtype=torch.bfloat16)
        x = torch.ones(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
        dt = torch.ones(1, 4, 128, 1, device="cuda", dtype=torch.bfloat16).expand_as(x)
        A = (-torch.ones(128, 1, 1, device="cuda")).expand(128, 64, 128)
        B = torch.ones(1, 4, 8, 128, device="cuda", dtype=torch.bfloat16)
        D = torch.ones(128, 1, device="cuda").expand(128, 64)
        kwargs = {
            "dt_bias": D,
            "out": torch.empty_like(x),
            "disable_state_update": True,
            "state_batch_indices": torch.tensor([1], device="cuda", dtype=torch.int32),
            "intermediate_states_buffer": torch.zeros(
                1, 4, 128, 64, 128, device="cuda", dtype=torch.bfloat16
            ),
            "cache_steps": 4,
            "retrieve_parent_token": torch.tensor([[-1, 0, 1, 2]], device="cuda"),
            "intermediate_state_indices": torch.tensor([0], device="cuda"),
        }
        with patch.object(ssm, "_selective_scan_update_kernel", ObservedKernel()):
            for enabled in (False, True):
                with envs.SGLANG_SM120_MAMBA_STATIC_VERIFY.override(enabled):
                    ssm.selective_state_update(state, x, dt, A, B, B, D, **kwargs)
            with envs.SGLANG_SM120_MAMBA_STATIC_VERIFY.override(True):
                kwargs["retrieve_parent_token"] = None
                ssm.selective_state_update(state, x, dt, A, B, B, D, **kwargs)
                kwargs["disable_state_update"] = False
                ssm.selective_state_update(state, x, dt, A, B, B, D, **kwargs)
        self.assertIsNone(calls[0])
        self.assertIsNotNone(calls[1])
        self.assertEqual(calls[1][:6], (1, 4, 128, 64, 128, 16))
        self.assertIsNotNone(calls[2])
        self.assertIsNone(calls[3])


if __name__ == "__main__":
    unittest.main()
