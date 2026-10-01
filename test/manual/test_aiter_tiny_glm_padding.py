"""MI355X correctness of the SGLang tiny-GLM inactive padding adapter.

Run: python3 test/manual/test_aiter_tiny_glm_padding.py
Requires both opt-in AITER factories. No synthetic performance claim is made.
"""


def main():
    import torch
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import GateMode, fused_moe
    from aiter.ops.flydsl.kernels.mxmoe_tiny_m4 import make_operator as m4
    from aiter.ops.flydsl.mxmoe_tiny_m8 import make_operator as m8

    from sglang.kernels.ops.moe.fill_padded_rows import _fill_padded_rows

    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"
    device = "cuda:0"
    w1 = torch.full((257, 512, 3072), 0x22, dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    w2 = torch.full((257, 6144, 128), 0x22, dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    w1.is_shuffled = w2.is_shuffled = True
    weights = {
        "w1": w1,
        "w2": w2,
        "w1_scale": torch.full(
            (257, 512, 192), 120, dtype=torch.uint8, device=device
        ).view(torch.float8_e8m0fnu),
        "w2_scale": torch.full(
            (257, 6144, 8), 120, dtype=torch.uint8, device=device
        ).view(torch.float8_e8m0fnu),
    }
    for rows, make in [(4, m4), (8, m8)]:
        # Two handles represent independent graph/stream scratch owners.
        operators = [make(weights=weights, rows=rows) for _ in range(2)]
        x = torch.empty((rows, 6144), dtype=torch.bfloat16, device=device)
        ids = torch.empty((rows, 9), dtype=torch.int32, device=device)
        rw = torch.empty((rows, 9), dtype=torch.float32, device=device)
        n = torch.tensor(rows, dtype=torch.int32, device=device)
        live = torch.tensor(
            [0, 7, 255, 1, 2, 3, 4, 5], dtype=torch.int32, device=device
        )

        def fill(active):
            n.fill_(active)
            x.fill_(float("nan"))
            x[:active].fill_(1 / 128)
            ids[:, :8].fill_(0)
            ids[:active, :8].copy_(live.expand(active, 8))
            ids[:, 8].fill_(256)
            rw.fill_(0.125)
            rw[:, 8].fill_(0.5)

        def adapted(op):
            _fill_padded_rows(ids[:, :8], n, 0, unique_ids=True, hidden_states=x)
            _fill_padded_rows(rw, n, 0.0)
            return op.run(x, ids, rw)

        def reference():
            return fused_moe(
                x,
                **weights,
                topk_ids=ids,
                topk_weight=rw,
                quant_type=QuantType.per_1x32,
                activation=ActivationType.Silu,
                gate_mode=GateMode.SEPARATED.value,
            )

        for op in operators:
            fill(rows)
            adapted(op)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = adapted(op)
            for active in list(range(rows + 1)) + [rows, 1, 0, rows]:
                fill(active)
                before = x[:active].clone()
                beforeids = ids[:active].clone()
                beforerw = rw[:active].clone()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(x[:active], before, atol=0, rtol=0)
                torch.testing.assert_close(ids[:active], beforeids, atol=0, rtol=0)
                torch.testing.assert_close(rw[:active], beforerw, atol=0, rtol=0)
                torch.testing.assert_close(out, reference(), atol=0.02, rtol=0.02)
                torch.testing.assert_close(
                    out[active:], torch.zeros_like(out[active:]), atol=0, rtol=0
                )
                assert torch.isfinite(out).all()
                assert op.state_check()["passed"]
        print(
            f"PASS M{rows}: all active prefixes, dirty inactive rows, shared weights, two private graph handles"
        )


if __name__ == "__main__":
    main()
