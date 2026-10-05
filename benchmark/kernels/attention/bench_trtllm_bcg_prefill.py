"""K3 per-rank attention-path microbenchmark; excludes input/output projections and TP collectives."""

import argparse
import gc
import json
import random
import statistics
import time

import flashinfer
import torch

from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    bcg_mla_bmm_then_unified_attention,
)
from sglang.srt.runtime_context import get_context
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    MockMLAModelRunner,
    TinyMLAModelConfig,
    _make_forward_batch,
)


class Attention(DeepseekMHAForwardMixin):
    num_local_heads = 12
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    use_dsa = False
    kv_cache_dtype = "fp8_e4m3"

    def __init__(self, weight):
        self.weight = weight
        self.attn_mha = RadixAttention(
            12, 192, 192**-0.5, num_kv_heads=12, layer_id=0, v_head_dim=128
        )
        self.attn_mqa = RadixAttention(
            12, 576, 192**-0.5, num_kv_heads=1, layer_id=0, v_head_dim=512
        )

    def kv_b_proj(self, x):
        return x @ self.weight.T, None

    def o_proj(self, x):
        return x, None


@torch.inference_mode()
def run_case(tokens, prefix, pool_tokens, iterations, rounds):
    if tokens <= 0 or prefix < 0 or tokens + prefix > pool_tokens:
        raise ValueError(
            "Require tokens > 0, prefix >= 0, and tokens + prefix <= pool_tokens"
        )
    torch.manual_seed(42)
    case = MLAAttentionCase(
        "k3", "trtllm_mla", ForwardMode.EXTEND, 12, 64, (prefix,), (tokens,)
    )
    config = TinyMLAModelConfig(
        num_heads=12,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        hidden_size=7168,
        context_len=pool_tokens,
    )
    config.qk_nope_head_dim = config.hf_config.qk_nope_head_dim = 128
    config.v_head_dim = config.hf_config.v_head_dim = 128
    config.scaling = 192**-0.5
    runner = MockMLAModelRunner(
        case=case,
        model_config=config,
        dtype=torch.bfloat16,
        device="cuda",
        max_context_len=pool_tokens,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fp8_kv_cache=True,
    )
    try:
        with get_context().override_server_args(
            disable_chunked_prefix_cache=False, flashinfer_mla_disable_ragged=True
        ):
            backend = TRTLLMMLABackend(runner)
            batch = _make_forward_batch(
                case, runner, max_context_len=pool_tokens, device="cuda"
            )
            batch.global_num_token_non_padded_cpu = tokens
            batch.req_to_token_pool = runner.req_to_token_pool
            batch.mha_one_shot = False
            pool = runner.token_to_kv_pool.get_key_buffer(0)
            pool.copy_(
                (torch.randn(pool.shape, device="cuda", dtype=torch.bfloat16) * 0.5).to(
                    torch.float8_e4m3fn
                )
            )
            q = torch.randn(tokens, 12, 192, device="cuda", dtype=torch.bfloat16) * 0.5
            weight = (
                torch.randn(3072, 512, device="cuda", dtype=torch.bfloat16) / 512**0.5
            )
            attn = Attention(weight)
            wk = weight.view(12, 256, 512)[:, :128].contiguous()
            wv = weight.view(12, 256, 512)[:, 128:].transpose(1, 2).contiguous()
            latent = pool[64 + prefix : 64 + prefix + tokens].to(torch.bfloat16)
            latent_kv = latent[:, 0, :512]
            k_rope = latent[:, :, 512:]

            absorbed_buffer = torch.empty(
                12, tokens, 512, device="cuda", dtype=torch.bfloat16
            )
            absorbed_view = absorbed_buffer.transpose(0, 1)
            old_output_buffer = torch.empty(
                tokens, 12 * 512, device="cuda", dtype=torch.bfloat16
            )

            # Reproduce the pre-change BCG absorbed-MLA path with the existing helper.
            def old_forward():
                bcg_mla_bmm_then_unified_attention(
                    q[:, :, :128].transpose(0, 1),
                    wk,
                    absorbed_buffer,
                    absorbed_view,
                    latent_kv.unsqueeze(1),
                    old_output_buffer,
                    False,
                    0,
                    q[:, :, 128:],
                    k_rope,
                )
                out = old_output_buffer.view(tokens, 12, 512)
                return (
                    torch.bmm(out.transpose(0, 1), wv)
                    .transpose(0, 1)
                    .reshape(tokens, -1)
                )

            def new_forward():
                kv = attn.kv_b_proj(latent_kv)[0].view(tokens, 12, 256)
                k = torch.cat((kv[:, :, :128], k_rope.expand(-1, 12, -1)), dim=-1)
                return attn.forward_normal_chunked_kv_core(q, k, kv[:, :, 128:], batch)

            def prepare(mode):
                backend.fallback_mla_under_breakable_graph = mode == "without"
                batch.num_prefix_chunks = batch.prefix_chunk_len = None
                batch.attn_attend_prefix_cache = None
                backend.init_forward_metadata(batch)

            graphs, outputs = {}, {}
            with (
                forward_context(ForwardContext(attn_backend=backend)),
                set_tc_piecewise_forward_context(
                    batch,
                    [attn.attn_mqa],
                    None,
                    [],
                    [],
                    mha_companion_layers=[attn.attn_mha],
                ),
                enable_breakable_cuda_graph(),
            ):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                for mode, fn in (("without", old_forward), ("with", new_forward)):
                    prepare(mode)
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    graph = BreakableCUDAGraph()
                    with BreakableCUDAGraphCapture(graph, stream=stream):
                        outputs[mode] = fn().clone()
                    graphs[mode] = graph
                # A changed input catches stale captured outputs before timing.
                q.mul_(0.9375)
                for mode, fn in (("without", old_forward), ("with", new_forward)):
                    prepare(mode)
                    graphs[mode].replay()
                    torch.cuda.synchronize()
                    reference = fn()
                    torch.testing.assert_close(
                        outputs[mode], reference, rtol=0.01, atol=0.001
                    )
                samples = {mode: [] for mode in graphs}
                rng = random.Random(123)
                for _ in range(rounds):
                    order = list(graphs)
                    rng.shuffle(order)
                    for mode in order:
                        prepare(mode)
                        graph = graphs[mode]
                        for _ in range(3):
                            graph.replay()
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        for _ in range(iterations):
                            graph.replay()
                        torch.cuda.synchronize()
                        samples[mode].append(
                            (time.perf_counter() - before) * 1000 / iterations
                        )
                before, after = outputs["without"].float(), outputs["with"].float()
                if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                    raise RuntimeError("Nonfinite attention output")
                error = (
                    ((before - after).square().mean() / before.square().mean())
                    .sqrt()
                    .item()
                )
                result = {
                    "tokens": tokens,
                    "prefix": prefix,
                    "pool_tokens": pool_tokens,
                    "relative_rms_difference": error,
                    "samples_ms": samples,
                    "median_ms": {
                        mode: statistics.median(values)
                        for mode, values in samples.items()
                    },
                }
                print(json.dumps(result), flush=True)
    finally:
        runner._server_args_override.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="256:0:131072,256:16384:131072,265:89600:131072,4096:16384:131072,16384:0:131072,16384:16384:131072,256:16384:1048576",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if args.iterations <= 0 or args.rounds <= 0:
        parser.error("iterations and rounds must be positive")
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "flashinfer": flashinfer.__version__,
                "args": vars(args),
            }
        ),
        flush=True,
    )
    for case in args.cases.split(","):
        run_case(*(int(x) for x in case.split(":")), args.iterations, args.rounds)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
