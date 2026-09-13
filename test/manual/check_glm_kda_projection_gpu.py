"""Manual ROCm KDA loader/projection/graph check; not endpoint qualification."""

import json
from types import SimpleNamespace

import torch


def main():
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path="dummy", device="cuda"), role="test")
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.models import glm5_next as model

    print(
        json.dumps(
            {
                "source": model.__file__,
                "torch": torch.__version__,
                "hip": torch.version.hip,
            }
        ),
        flush=True,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(20260913)

    names = [
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "f_b_proj",
        "g_a_proj",
        "g_b_proj",
    ]
    prefix = "model.layers.1.self_attn"
    quant = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        ignored_layers=[f"{prefix}.{n}" for n in names],
    )
    cfg = SimpleNamespace(
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
        },
        dtype=torch.bfloat16,
    )
    with (
        get_parallel().override(tp_rank=0, tp_size=1, attn_tp_rank=0, attn_tp_size=1),
        torch.device("cuda"),
        torch.inference_mode(),
    ):
        module = model.Glm5NextLinearAttention(
            1, 512, cfg, quant_config=quant, prefix=prefix
        )
        assert module.do_fuse_qkvbfg
        assert module.fused_qkvbfg_a_proj.weight.dtype == torch.bfloat16
        assert module.fused_fg_b_proj.weight.dtype == torch.bfloat16
        weights = [
            torch.randn(n, 512, device="cuda") * 0.01
            for n in (512, 512, 512, 4, 128, 128)
        ]
        bweights = [torch.randn(512, 128, device="cuda") * 0.01 for _ in range(2)]
        # Use the real shard loader, including repeated f/g-a rows and batched b projections.
        for shard, weight in enumerate(weights):
            layer = module.fused_qkvbfg_a_proj
            layer.weight_loader(layer.weight, weight, shard)
        for shard, weight in enumerate(bweights):
            layer = module.fused_fg_b_proj
            layer.weight_loader(layer.weight, weight, shard)
        for count in (1, 8, 16, 257):
            x = torch.randn(count, 512, device="cuda")
            expected = (
                torch.cat([x @ w.T for w in weights[:3]], -1),
                x @ weights[3].T,
                (x @ weights[4].T) @ bweights[0].T,
                (x @ weights[5].T) @ bweights[1].T,
            )
            actual = module.forward_qkvbfg_fused(x, None)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0.003, rtol=0.02)
            if count <= 16:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        module.forward_qkvbfg_fused(x, None)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = module.forward_qkvbfg_fused(x, None)
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                for a, b in zip(captured, expected):
                    torch.testing.assert_close(a, b, atol=0.003, rtol=0.02)
            print(
                json.dumps({"tokens": count, "loader_projection_graph": "pass"}),
                flush=True,
            )
        print("G03_REAL_MODULE_LOADER_GRAPH_PASS_NO_ENDPOINT_QUALIFICATION", flush=True)


if __name__ == "__main__":
    main()
