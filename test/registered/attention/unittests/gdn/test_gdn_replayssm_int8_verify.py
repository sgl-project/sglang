"""Check the INT8 checkpoint projection with the existing D/K residual ring."""

import unittest

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class TestGDNReplaySSMInt8Verify(unittest.TestCase):
    def test_matches_exactly_representable_bf16_checkpoint(self):
        device = "cuda"
        dtype = torch.bfloat16
        gen = torch.Generator(device=device).manual_seed(134)
        requests, slots, heads, value_heads = 2, 4, 2, 6
        dim, ring, window = 128, 32, 8
        shape = (slots, value_heads, dim, dim)

        def randn(*shape, dtype=dtype):
            return torch.randn(shape, dtype=dtype, device=device, generator=gen)

        def indices(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        quantized = torch.randint(
            -127, 128, shape, dtype=torch.int8, device=device, generator=gen
        )
        # Layer slices of a slot-major scale pool retain a strided slot axis.
        exponents = torch.randint(
            -9, -4, (slots, 2, value_heads, dim), device=device, generator=gen
        )
        scales = torch.exp2(exponents.float())[:, 1]
        self.assertFalse(scales.is_contiguous())
        # Powers of two make dequantization exact, so any output difference
        # would come from applying the scales in the wrong part of the recurrence.
        bf16_state = (quantized.float() * scales.unsqueeze(-1)).to(dtype)
        tokens = requests * window
        args = {
            "q": randn(tokens, heads, dim),
            "k": randn(tokens, heads, dim),
            "v": randn(tokens, value_heads, dim),
            "a": randn(tokens, value_heads),
            "b": randn(tokens, value_heads),
            "A_log": randn(value_heads, dtype=torch.float32),
            "dt_bias": randn(value_heads, dtype=torch.float32),
            "query_start_loc": indices([0, window, tokens]),
            "ssm_state_indices": indices([3, 1]),
            "replay_indices": indices([1, 3]),
            "write_pos": indices([0, 5, 0, 3]),
            "cache_base": indices([0, 28, 0, 7]),
            "is_flush": indices([0] * slots).to(torch.int8),
            "max_cache_len": ring,
            "max_spec_len": window,
            "launch_mode": "verify",
            "null_block_id": -1,
            "num_warps": 2,
            "num_stages": 1,
        }
        rings = {
            "d_cache": randn(slots, value_heads, ring, dim) * 0.25,
            "k_cache": randn(slots, heads, ring, dim) * 0.125,
            "g_cache": -randn(slots, value_heads, ring, dtype=torch.float32).abs(),
            "rawv_cache": randn(slots, value_heads, ring, dim) * 0.001,
            "rawk_cache": randn(slots, heads, ring, dim) * 0.001,
        }

        def verify(state, row_scales):
            updated = {name: buf.clone() for name, buf in rings.items()}
            out = torch.empty_like(args["v"])
            gdn_replayssm_spec_decode(
                **args,
                **updated,
                out=out,
                checkpoint_state=state,
                checkpoint_scale=row_scales,
                beta_cache=None,
            )
            return out, updated

        checkpoint_before = quantized.clone()
        expected, expected_rings = verify(bf16_state, None)
        actual, actual_rings = verify(quantized, scales)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(quantized, checkpoint_before, rtol=0, atol=0)
        for name, buf in expected_rings.items():
            # The low part of D exposes small FP32 accumulator differences
            # that vanish in the BF16 forward output and the high part of D.
            atol = 4e-6 if name == "rawv_cache" else 0
            torch.testing.assert_close(actual_rings[name], buf, rtol=0, atol=atol)
        for name in ("rawv_cache", "rawk_cache"):
            self.assertFalse(torch.equal(actual_rings[name], rings[name]))

    def make_case(self):
        device = "cuda"
        gen = torch.Generator(device=device).manual_seed(43)
        slots, heads, value_heads, dim, ring = 6, 2, 6, 128, 32

        def randn(*shape, dtype=torch.bfloat16):
            return torch.randn(shape, dtype=dtype, device=device, generator=gen)

        def index(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        state = torch.randint(
            -127,
            128,
            (slots, value_heads, dim, dim),
            device=device,
            dtype=torch.int8,
            generator=gen,
        )
        # Arbitrary positive scales with pitched slot and value-head strides.
        scale = (
            torch.rand((slots, value_heads, 2, dim), device=device, generator=gen)
            * 0.005
            + 0.001
        )[:, :, 1]
        return dict(
            checkpoint_state=state,
            checkpoint_scale=scale,
            q=randn(18, heads, dim) * 0.2,
            k=randn(18, heads, dim) * 0.2,
            v=randn(18, value_heads, dim) * 0.2,
            a=randn(18, value_heads) * 0.1,
            b=randn(18, value_heads) * 0.1,
            A_log=randn(value_heads, dtype=torch.float32) * 0.1 - 1,
            dt_bias=randn(value_heads, dtype=torch.float32) * 0.1,
            d_cache=randn(slots, value_heads, ring, dim) * 0.2,
            k_cache=randn(slots, heads, ring, dim) * 0.1,
            g_cache=-randn(slots, value_heads, ring, dtype=torch.float32).abs() * 0.1,
            rawv_cache=randn(slots, value_heads, ring, dim) * 0.001,
            rawk_cache=randn(slots, heads, ring, dim) * 0.001,
            beta_cache=None,
            out=torch.full(
                (18, value_heads, dim), 99, dtype=torch.bfloat16, device=device
            ),
            query_start_loc=index([0, 8, 16, 18]),
            ssm_state_indices=index([1, 3, -1]),
            replay_indices=index([2, 0, -1]),
            write_pos=index([0, 0, 5, 0, 0, 0]),
            cache_base=index([7, 0, 28, 0, 0, 0]),
            is_flush=torch.zeros(slots, dtype=torch.int8, device=device),
            max_cache_len=ring,
            max_spec_len=8,
            launch_mode="verify",
            null_block_id=-1,
        )

    @staticmethod
    def ring_names():
        return ("d_cache", "k_cache", "g_cache", "rawv_cache", "rawk_cache")

    @staticmethod
    def clone_case(case):
        return {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in case.items()
        }

    def test_arbitrary_scales_against_fp64_recurrence(self):
        case = self.make_case()
        initial = self.clone_case(case)
        gdn_replayssm_spec_decode(**case)
        metrics = {"out": [], "D": [], "K": []}
        ring = case["max_cache_len"]
        dim = case["k"].shape[-1]
        group = case["v"].shape[1] // case["k"].shape[1]

        def unit(x):
            x = x.double()
            return x * (x.square().sum(dim=-1, keepdim=True) + 1e-6).rsqrt()

        for request in (0, 1):
            slot = int(initial["ssm_state_indices"][request])
            row = int(initial["replay_indices"][request])
            base = int(initial["cache_base"][row])
            history = int(initial["write_pos"][row])
            state = initial["checkpoint_state"][slot].double() * initial[
                "checkpoint_scale"
            ][slot].double().unsqueeze(-1)
            # Historical replay consumes stored D/K high parts and FP32 gates.
            # The speculative window also writes the D/K low parts for commit.
            for i in range(history):
                pos = (base + i) % ring
                d = initial["d_cache"][row, :, pos].double()
                k = (
                    initial["k_cache"][row, :, pos]
                    .double()
                    .repeat_interleave(group, dim=0)
                )
                decay = initial["g_cache"][row, :, pos].double().exp()
                state = decay[:, None, None] * state + d[:, :, None] * k[:, None, :]
            expected_output, expected_d, expected_k = [], [], []
            for tok in range(request * 8, (request + 1) * 8):
                k = unit(initial["k"][tok])
                query = unit(initial["q"][tok]).repeat_interleave(group, dim=0) / (
                    dim**0.5
                )
                key = k.repeat_interleave(group, dim=0)
                x = initial["a"][tok].double() + initial["dt_bias"].double()
                decay = (
                    -initial["A_log"].double().exp() * torch.nn.functional.softplus(x)
                ).exp()
                state = state * decay[:, None, None]
                beta = initial["b"][tok].double().sigmoid()
                delta = beta[:, None] * (
                    initial["v"][tok].double() - torch.einsum("hvk,hk->hv", state, key)
                )
                state = state + delta[:, :, None] * key[:, None, :]
                expected_output.append(torch.einsum("hvk,hk->hv", state, query))
                expected_d.append(delta)
                expected_k.append(k)
            indices = [(base + history + i) % ring for i in range(8)]
            actual = {
                "out": case["out"][request * 8 : (request + 1) * 8].double(),
                "D": (
                    case["d_cache"][row, :, indices].double()
                    + case["rawv_cache"][row, :, indices].double()
                ).transpose(0, 1),
                "K": (
                    case["k_cache"][row, :, indices].double()
                    + case["rawk_cache"][row, :, indices].double()
                ).transpose(0, 1),
            }
            for name, values in zip(
                ("out", "D", "K"), (expected_output, expected_d, expected_k)
            ):
                ref = torch.stack(values)
                diff = actual[name] - ref
                rel = float(
                    torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref)
                )
                metrics[name].append((rel, float(diff.abs().max())))
                self.assertLess(
                    rel, 0.012 if name != "K" else 1e-5, (name, history, rel)
                )
        for name, stats in metrics.items():
            print(
                f"FP64 {name}: max_rel_l2={max(x[0] for x in stats):.8f} max_abs={max(x[1] for x in stats):.8f}"
            )

    def test_nonfinite_tail_and_write_boundaries(self):
        original = self.make_case()
        clean = self.clone_case(original)
        gdn_replayssm_spec_decode(**clean)
        # Packed ranges are 8/8/2, with the 2-token row a padding sentinel.
        written = torch.zeros((6, 32), dtype=torch.bool, device="cuda")
        written[2, 1:9] = True
        written[0, 7:15] = True
        for value in (float("nan"), float("inf")):
            for source in ("draft", "outside_history"):
                with self.subTest(value=value, source=source):
                    dirty = self.clone_case(original)
                    if source == "draft":
                        for name in ("q", "k", "v", "a", "b"):
                            dirty[name][5:8] = value
                    else:
                        for name in self.ring_names():
                            dirty[name][2, :, 14] = value
                    before = self.clone_case(dirty)
                    gdn_replayssm_spec_decode(**dirty)
                    torch.testing.assert_close(
                        dirty["out"][:5], clean["out"][:5], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        dirty["out"][8:], clean["out"][8:], rtol=0, atol=0
                    )
                    self.assertEqual(int(torch.count_nonzero(dirty["out"][16:])), 0)
                    for name in self.ring_names():
                        torch.testing.assert_close(
                            dirty[name][2, :, 1:6],
                            clean[name][2, :, 1:6],
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            dirty[name].transpose(1, 2)[~written],
                            before[name].transpose(1, 2)[~written],
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )
                    for name in (
                        "checkpoint_state",
                        "checkpoint_scale",
                        "write_pos",
                        "cache_base",
                        "is_flush",
                    ):
                        torch.testing.assert_close(
                            dirty[name], before[name], rtol=0, atol=0
                        )

    def test_int8_interface_validation_and_pending_flush(self):
        case = self.make_case()
        scales = case["checkpoint_scale"]
        invalid = (
            ("launch_mode", "both", "launch_mode"),
            ("launch_mode", "flush", "launch_mode"),
            ("checkpoint_scale", None, "checkpoint_scale"),
            ("checkpoint_scale", scales.float()[1:], "checkpoint_scale"),
            ("checkpoint_scale", scales.bfloat16(), "checkpoint_scale"),
            ("checkpoint_scale", scales.cpu(), "checkpoint_scale"),
            (
                "checkpoint_scale",
                torch.stack((scales, scales), dim=-1)[..., 0],
                "checkpoint_scale",
            ),
        )
        for name, value, message in invalid:
            with self.subTest(name=name, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    gdn_replayssm_spec_decode(**(case | {name: value}))
        case["is_flush"][2] = 1
        before = self.clone_case(case)
        gdn_replayssm_spec_decode(**case)
        torch.testing.assert_close(case["out"][:8], before["out"][:8], rtol=0, atol=0)
        for name in self.ring_names():
            torch.testing.assert_close(case[name][2], before[name][2], rtol=0, atol=0)

    def test_cuda_graph_replays_updated_state_and_cursors(self):
        template = self.make_case()
        graph_case = self.make_case()
        gdn_replayssm_spec_decode(**graph_case)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gdn_replayssm_spec_decode(**graph_case)
        for step, (history, base, pending) in enumerate(
            ((0, 0, 0), (5, 28, 0), (3, 30, 1))
        ):
            fresh = self.clone_case(template)
            fresh["write_pos"][2] = history
            fresh["cache_base"][2] = base
            fresh["is_flush"][2] = pending
            fresh["q"].mul_(1 + step / 8)
            fresh["checkpoint_scale"].mul_(1 + step / 5)
            for name, tensor in fresh.items():
                if isinstance(tensor, torch.Tensor):
                    graph_case[name].copy_(tensor)
            gdn_replayssm_spec_decode(**fresh)
            graph.replay()
            for name in (
                "out",
                *self.ring_names(),
                "checkpoint_state",
                "checkpoint_scale",
                "write_pos",
                "cache_base",
                "is_flush",
            ):
                torch.testing.assert_close(
                    graph_case[name], fresh[name], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
