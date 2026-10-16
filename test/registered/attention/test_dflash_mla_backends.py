"""Real compressed DFlash MLA backends, including bidirectional CUDA graphs."""

import copy
import gc
import importlib.util
import json
import math
import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.mem_cache.memory_pool import (
    KVWriteLoc,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.dflash import DFlashMLAAttention
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    MockMLAModelRunner,
    TinyMLAModelConfig,
    _make_forward_batch,
)
from sglang.test.parallel_groups import publish

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

HIDDEN, HEADS, RANK, NOPE, ROPE, VALUE, BLOCK = 128, 32, 512, 128, 64, 128, 8
CAPACITY = 256


def _rope(x, positions, interleaved):
    freq = positions.float()[:, None] / (
        10000 ** (torch.arange(0, ROPE, 2, device=x.device).float() / ROPE)
    )
    if interleaved:
        cos, sin = (
            freq.cos().repeat_interleave(2, -1),
            freq.sin().repeat_interleave(2, -1),
        )
        a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack((-b, a), -1).flatten(-2)
    else:
        cos, sin = freq.cos().repeat(1, 2), freq.sin().repeat(1, 2)
        a, b = x.chunk(2, -1)
        rotated = torch.cat((-b, a), -1)
    return x * cos[:, None] + rotated * sin[:, None]


def _expanded_reference(
    module, prefix, hidden, positions, *, interleaved, causal=False
):
    """Independent fp32 SpecForge equations; materialize per-head K and V."""
    weights = {name: p.float() for name, p in module.named_parameters()}

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name):
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            * weights[name + ".weight"]
        )

    hidden = hidden.float()
    if module.q_lora_rank is None:
        q = linear(hidden, "q_proj")
    else:
        q = linear(norm(linear(hidden, "q_a_proj"), "q_a_layernorm"), "q_b_proj")
    q_nope, q_pe = q.view(-1, HEADS, NOPE + ROPE).split((NOPE, ROPE), -1)
    q = torch.cat((q_nope, _rope(q_pe, positions, interleaved)), -1)
    outputs = []
    for i, ctx in enumerate(prefix):
        x = torch.cat((ctx.float(), hidden[i * BLOCK : (i + 1) * BLOCK]))
        c, k_pe = linear(x, "kv_a_proj_with_mqa").split((RANK, ROPE), -1)
        k_nope, v = (
            linear(norm(c, "kv_a_layernorm"), "kv_b_proj")
            .view(-1, HEADS, NOPE + VALUE)
            .split((NOPE, VALUE), -1)
        )
        k_pe = _rope(
            k_pe[:, None], torch.arange(x.shape[0], device=x.device), interleaved
        )
        k = torch.cat((k_nope, k_pe.expand(-1, HEADS, -1)), -1)
        scores = (
            torch.einsum("qhd,khd->hqk", q[i * BLOCK : (i + 1) * BLOCK], k)
            * (NOPE + ROPE) ** -0.5
        )
        if causal:
            allowed = (
                torch.arange(x.shape[0], device=x.device)[None]
                <= (len(ctx) + torch.arange(BLOCK, device=x.device))[:, None]
            )
            scores.masked_fill_(~allowed[None], -torch.inf)
        output = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).flatten(1)
        outputs.append(linear(output, "o_proj"))
    return torch.cat(outputs)


def _tokenspeed_fp8_reference(
    module, prefix, hidden, positions, *, interleaved, causal
):
    """Independent Torch oracle for TokenSpeed's E4M3 Q/KV/P quantization.

    This oracle covers one 128-token KV tile, with FP32 accumulation and
    normalization. It reads weights only, without calling production model
    projections or attention, or reading the served KV cache.
    """
    weights = dict(module.named_parameters())
    dtype = hidden.dtype

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name):
        x = x.float()
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            * weights[name + ".weight"].float()
        ).to(dtype)

    def round_fp8(x):
        return x.to(torch.float8_e4m3fn).float()

    def kv(x, pos):
        c, k_pe = linear(x, "kv_a_proj_with_mqa").split((RANK, ROPE), -1)
        c = round_fp8(norm(c, "kv_a_layernorm"))
        k_pe = round_fp8(_rope(k_pe[:, None], pos, interleaved).to(dtype))
        return c, k_pe.squeeze(1)

    if module.q_lora_rank is None:
        q = linear(hidden, "q_proj")
    else:
        q = linear(norm(linear(hidden, "q_a_proj"), "q_a_layernorm"), "q_b_proj")
    q_nope, q_pe = q.view(-1, HEADS, NOPE + ROPE).split((NOPE, ROPE), -1)
    w_k, w_v = (
        weights["kv_b_proj.weight"]
        .view(HEADS, NOPE + VALUE, RANK)
        .split((NOPE, VALUE), 1)
    )
    q_abs = torch.bmm(q_nope.transpose(0, 1), w_k).transpose(0, 1)
    q_pe = _rope(q_pe, positions, interleaved).to(dtype)
    q = round_fp8(torch.cat((q_abs, q_pe), -1))
    block_c, block_pe = kv(hidden, positions)
    outputs = []
    for i, ctx in enumerate(prefix):
        ctx_pos = torch.arange(len(ctx), device=hidden.device)
        ctx_c, ctx_pe = kv(ctx, ctx_pos)
        block = slice(i * BLOCK, (i + 1) * BLOCK)
        c = torch.cat((ctx_c, block_c[block]))
        assert len(c) <= 128, "FP8 oracle assumes one 128-token KV tile"
        k = torch.cat((c, torch.cat((ctx_pe, block_pe[block]))), -1)
        scores = torch.einsum("qhd,kd->hqk", q[block], k) * (NOPE + ROPE) ** -0.5
        if causal:
            allowed = (
                torch.arange(len(c), device=hidden.device)[None]
                <= (len(ctx) + torch.arange(BLOCK, device=hidden.device))[:, None]
            )
            scores.masked_fill_(~allowed[None], -torch.inf)
        # These short sequences fit one kernel KV tile. TokenSpeed computes
        # the normalization denominator before rounding P for the FP8 P@V MMA.
        probabilities = (scores - scores.amax(-1, keepdim=True)).exp()
        denominator = probabilities.sum(-1, keepdim=True)
        quantized_probabilities = round_fp8(probabilities) / denominator
        outputs.append(torch.einsum("hqk,kd->qhd", quantized_probabilities, c))
    latent_output = torch.cat(outputs).to(dtype)
    output = (
        torch.bmm(latent_output.transpose(0, 1), w_v.transpose(1, 2))
        .transpose(0, 1)
        .flatten(1)
    )
    return linear(output, "o_proj").float()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDFlashMLABackends(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")

    def _check(
        self,
        backend_name,
        *,
        fp8=False,
        interleaved=True,
        q_lora_rank=None,
        causal=False,
    ):
        torch.manual_seed(20261016)
        case = MLAAttentionCase(
            name="dflash_causal" if causal else "dflash_bidirectional",
            backend=backend_name,
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=HEADS,
            page_size=32,
            prefix_lens=(0, 31),
            extend_lens=(BLOCK, BLOCK),
        )
        config = Qwen3Config(
            hidden_size=HIDDEN,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=HEADS,
            num_key_value_heads=1,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=RANK,
            qk_nope_head_dim=NOPE,
            qk_rope_head_dim=ROPE,
            v_head_dim=VALUE,
            rope_interleave=interleaved,
            rms_norm_eps=1e-6,
            rope_theta=10000,
            max_position_embeddings=CAPACITY,
            is_causal=causal,
            layer_types=["full_attention"],
            dflash_config={"attention_mode": "mla", "block_size": BLOCK},
        )
        model_config = TinyMLAModelConfig(
            num_heads=HEADS,
            kv_lora_rank=RANK,
            qk_rope_head_dim=ROPE,
            hidden_size=HIDDEN,
            context_len=CAPACITY,
        )
        model_config.is_dflash_mla_draft = True
        model_config.hf_config = model_config.hf_text_config = config
        model_config.qk_nope_head_dim = NOPE
        model_config.head_dim = NOPE + ROPE
        model_config.v_head_dim = VALUE
        model_config.scaling = (NOPE + ROPE) ** -0.5
        model_config.dflash_mla_causal = causal
        runner = MockMLAModelRunner(
            case=case,
            model_config=model_config,
            dtype=torch.bfloat16,
            device="cuda",
            max_context_len=CAPACITY,
            kv_lora_rank=RANK,
            qk_rope_head_dim=ROPE,
            disable_cuda_graph=False,
            fp8_kv_cache=fp8,
        )
        runner.is_draft_worker = True
        runner.spec_algorithm = SpeculativeAlgorithm.DFLASH
        # Runtime namespaces snapshot resolved arguments; mutating server_args
        # here would not update the values read by the attention backends.
        get_context().override(
            "dflash_mla_backend_fixture",
            speculative_algorithm="DFLASH",
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=BLOCK,
            speculative_num_steps=BLOCK - 1,
        )
        runner.draft_attention_backend = backend_name
        backend = ATTENTION_BACKENDS[backend_name](runner)
        # Normally stamped by ModelRunner.init_attention_backend, which this
        # isolated kernel fixture deliberately bypasses.
        backend.prefill_attention_backend_str = backend_name
        backend.decode_attention_backend_str = backend_name
        self.assertIsInstance(runner.token_to_kv_pool, MLATokenToKVPool)
        self.assertEqual(
            runner.token_to_kv_pool.get_key_buffer(0).shape[1:],
            (1, RANK + ROPE),
        )
        module = DFlashMLAAttention(config, layer_id=0)
        module.rotary_emb = copy.deepcopy(module.rotary_emb)
        # Model loading casts weights, while the CUDA RoPE kernel keeps its
        # table in FP32. Preserve the original table before Module.to casts
        # floating buffers, rather than widening an already-rounded BF16 copy.
        rope_cache = module.rotary_emb.cos_sin_cache
        self.assertEqual(rope_cache.dtype, torch.float32)
        module = module.to(device="cuda", dtype=torch.bfloat16).eval()
        module.rotary_emb.cos_sin_cache = rope_cache.to(device="cuda")
        with torch.no_grad():
            for name, param in module.named_parameters():
                if "layernorm" in name:
                    param.fill_(1)
                else:
                    param.normal_(std=1 / math.sqrt(param.shape[-1]))

        def make_inputs(prefix_lens):
            current = MLAAttentionCase(
                name=case.name,
                backend=backend_name,
                forward_mode=case.forward_mode,
                num_heads=HEADS,
                page_size=case.page_size,
                prefix_lens=prefix_lens,
                extend_lens=(BLOCK,) * len(prefix_lens),
            )
            batch = _make_forward_batch(
                current, runner, max_context_len=CAPACITY, device="cuda"
            )
            batch.seq_lens = torch.tensor(prefix_lens, device="cuda", dtype=torch.int32)
            # Match DFlashWorkerV2: device lengths are committed prefixes;
            # host planning lengths include the in-flight verify block.
            # FlashInfer's DFlash replay fast plan consumes the latter directly.
            batch.seq_lens_cpu = torch.tensor(prefix_lens, dtype=torch.int32) + BLOCK
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            batch.spec_info = DFlashVerifyInput(
                draft_token=batch.input_ids,
                positions=batch.positions,
                draft_token_num=BLOCK,
                topk=1,
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            # The generic MLA fixture plans total sequence lengths before it
            # knows about speculation. DFlash instead plans committed prefix
            # lengths plus its BLOCK-sized read/write window.
            runner.kv_index_translator.bind_own_plan(batch)
            self.assertIs(batch.kv_loc_plan.seq_lens, batch.seq_lens)
            self.assertEqual(batch.kv_loc_plan.read_extent, BLOCK)
            prefixes = [
                torch.randn(n, HIDDEN, device="cuda", dtype=torch.bfloat16)
                for n in prefix_lens
            ]
            hidden = torch.randn(
                len(prefix_lens) * BLOCK, HIDDEN, device="cuda", dtype=torch.bfloat16
            )
            for i, ctx in enumerate(prefixes):
                if not len(ctx):
                    continue
                c, k_pe = module.kv_proj_only(ctx)
                k_pe = module.apply_k_rope(torch.arange(len(ctx), device="cuda"), k_pe)
                locs = runner.req_to_token_pool.req_to_token[i, : len(ctx)].long()
                runner.token_to_kv_pool.set_mla_kv_buffer(
                    module.attn,
                    KVWriteLoc(locs, physical=True),
                    c.unsqueeze(1),
                    k_pe.unsqueeze(1),
                )
            return batch, prefixes, hidden

        def quantized_reference(prefix, hidden, positions):
            if backend_name != "tokenspeed_mla":
                return None
            return _tokenspeed_fp8_reference(
                module,
                prefix,
                hidden,
                positions,
                interleaved=interleaved,
                causal=causal,
            )

        def check_output(actual, expected, label, quantized_expected=None):
            error = (actual.float() - expected).abs()
            relative_l2 = error.norm() / expected.norm()
            quantization_metrics = {}
            if quantized_expected is not None:
                quantized_error = (actual.float() - quantized_expected).abs()
                quantized_relative_l2 = (
                    quantized_error.norm() / quantized_expected.norm()
                )
                quantization_metrics = {
                    "fp8_reference_max_abs_error": quantized_error.max().item(),
                    "fp8_reference_relative_l2": quantized_relative_l2.item(),
                    "fp8_reference_model_relative_l2": (
                        (quantized_expected - expected).norm() / expected.norm()
                    ).item(),
                }
            print(
                json.dumps(
                    {
                        "backend": backend_name,
                        "fp8": fp8,
                        "causal": causal,
                        "phase": label,
                        "max_abs_error": error.max().item(),
                        "relative_l2": relative_l2.item(),
                        **quantization_metrics,
                    }
                ),
                flush=True,
            )
            self.assertTrue(torch.isfinite(actual).all())
            if quantized_expected is not None:
                # Bound total FP8 model error separately from the much smaller
                # kernel residual against the independent quantized oracle.
                self.assertLess(relative_l2.item(), 0.05)
                self.assertLess(quantized_relative_l2.item(), 0.003)
                torch.testing.assert_close(
                    actual.float(), quantized_expected, atol=0.01, rtol=0.01
                )
            else:
                self.assertLess(relative_l2.item(), 0.045 if fp8 else 0.012)
                torch.testing.assert_close(
                    actual.float(),
                    expected,
                    atol=0.07 if fp8 else 0.025,
                    rtol=0.06 if fp8 else 0.025,
                )

        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            batch, prefix, hidden = make_inputs(case.prefix_lens)
            expected = _expanded_reference(
                module,
                prefix,
                hidden,
                batch.positions,
                interleaved=interleaved,
                causal=causal,
            )
            opposite_mask = _expanded_reference(
                module,
                prefix,
                hidden,
                batch.positions,
                interleaved=interleaved,
                causal=not causal,
            )
            self.assertGreater((expected[0] - opposite_mask[0]).norm().item(), 0.1)
            quantized_expected = quantized_reference(prefix, hidden, batch.positions)
            backend.init_forward_metadata(batch)
            check_output(
                module(batch.positions, hidden, batch),
                expected,
                "eager",
                quantized_expected,
            )

            # Exercise actual CUDA capture and replay, not only graph metadata.
            # Production captures a fresh static ForwardBatch. An eager batch
            # can carry a per-batch backend metadata override (TokenSpeed/TRTLLM)
            # that would take precedence over the persistent graph buffers.
            batch, _, hidden = make_inputs(case.prefix_lens)
            self.assertIsNone(getattr(batch, "decode_trtllm_mla_metadata", None))
            backend.init_cuda_graph_state(max_bs=2, max_num_tokens=2 * BLOCK)
            runner.kv_index_translator.bind_runner_slots(batch)
            capture_plan = batch.kv_loc_plan
            captured_buffers = {
                name: getattr(batch, name).data_ptr()
                for name in (
                    "positions",
                    "out_cache_loc",
                    "seq_lens",
                    "req_pool_indices",
                )
            }
            self.assertIs(capture_plan.write_physical, batch.out_cache_loc)
            backend.init_forward_metadata_out_graph(batch, in_capture=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    backend.init_forward_metadata_in_graph(batch)
                    module(batch.positions, hidden, batch)
            torch.cuda.current_stream().wait_stream(stream)
            backend.on_after_cuda_graph_warmup()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                backend.init_forward_metadata_in_graph(batch)
                output = module(batch.positions, hidden, batch)
            for prefix_lens in ((0, 31), (63, 32), (1, 0)):
                replay, prefix, next_hidden = make_inputs(prefix_lens)
                expected = _expanded_reference(
                    module,
                    prefix,
                    next_hidden,
                    replay.positions,
                    interleaved=interleaved,
                    causal=causal,
                )
                quantized_expected = quantized_reference(
                    prefix, next_hidden, replay.positions
                )
                hidden.copy_(next_hidden)
                batch.positions.copy_(replay.positions)
                batch.out_cache_loc.copy_(replay.out_cache_loc)
                batch.seq_lens.copy_(replay.seq_lens)
                batch.seq_lens_cpu.copy_(replay.seq_lens_cpu)
                batch.seq_lens_sum = replay.seq_lens_sum
                batch.req_pool_indices.copy_(replay.req_pool_indices)
                self.assertIs(batch.kv_loc_plan, capture_plan)
                for name, address in captured_buffers.items():
                    self.assertEqual(getattr(batch, name).data_ptr(), address)
                # Refresh the backend's captured metadata buffers from this
                # iteration's live plan. The graph still writes through its
                # original runner-slot tensor, whose contents were copied above.
                backend.init_forward_metadata_out_graph(replay)
                graph.replay()
                graph_output = output.clone()
                # A normal execution with the same graph metadata distinguishes
                # bad planning from stale pointers baked into the captured graph.
                # Run it after replay so it cannot fill missing KV writes for it.
                metadata_output = module(batch.positions, hidden, batch)
                with self.subTest(
                    phase="graph_metadata_eager", prefix_lens=prefix_lens
                ):
                    check_output(
                        metadata_output,
                        expected,
                        f"graph_metadata_eager_{prefix_lens}",
                        quantized_expected,
                    )
                with self.subTest(phase="graph", prefix_lens=prefix_lens):
                    check_output(
                        graph_output,
                        expected,
                        f"graph_{prefix_lens}",
                        quantized_expected,
                    )

    def test_triton_bf16(self):
        self._check("triton")

    def test_triton_fp8_q_lora_neox(self):
        self._check("triton", fp8=True, interleaved=False, q_lora_rank=64)

    def test_flashinfer_bf16(self):
        self._check("flashinfer")

    def test_tokenspeed_fp8(self):
        if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
            self.skipTest("TokenSpeed MLA requires Blackwell SM100/SM103")
        if importlib.util.find_spec("tokenspeed_mla") is None:
            self.skipTest("tokenspeed_mla is not installed")
        self._check("tokenspeed_mla", fp8=True)

    def test_causal_backends(self):
        for backend_name in ("triton", "flashinfer", "tokenspeed_mla"):
            with self.subTest(backend=backend_name):
                if backend_name == "tokenspeed_mla":
                    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
                        self.skipTest("TokenSpeed MLA requires Blackwell SM100/SM103")
                    if importlib.util.find_spec("tokenspeed_mla") is None:
                        self.skipTest("tokenspeed_mla is not installed")
                self._check(
                    backend_name,
                    fp8=backend_name == "tokenspeed_mla",
                    causal=True,
                )

    def test_gpu_cache_memory(self):
        size, layers, page = 32768, 5, 32
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            measurements = {}
            for layout, cls, kwargs in (
                (
                    "compressed",
                    MLATokenToKVPool,
                    {"kv_lora_rank": RANK, "qk_rope_head_dim": ROPE},
                ),
                (
                    "expanded",
                    MHATokenToKVPool,
                    {
                        "head_num": HEADS,
                        "head_dim": NOPE + ROPE,
                        "v_head_dim": VALUE,
                    },
                ),
            ):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                pool = cls(
                    size=size,
                    page_size=page,
                    dtype=dtype,
                    layer_num=layers,
                    device="cuda",
                    enable_memory_saver=False,
                    **kwargs,
                )
                torch.cuda.synchronize()
                buffer_bytes = pool.get_kv_size_bytes()
                if isinstance(buffer_bytes, tuple):
                    buffer_bytes = sum(buffer_bytes)
                buffer_bytes = int(buffer_bytes)
                measurements[layout] = {
                    "buffer_bytes": buffer_bytes,
                    "allocated_bytes": torch.cuda.memory_allocated() - before,
                }
                expected = (
                    (size + page)
                    * layers
                    * dtype.itemsize
                    * (
                        (RANK + ROPE)
                        if layout == "compressed"
                        else HEADS * (NOPE + ROPE + VALUE)
                    )
                )
                self.assertEqual(buffer_bytes, expected)
                del pool
            ratio = (
                measurements["expanded"]["buffer_bytes"]
                / measurements["compressed"]["buffer_bytes"]
            )
            self.assertAlmostEqual(ratio, HEADS * (NOPE + ROPE + VALUE) / (RANK + ROPE))
            self.assertGreater(
                measurements["expanded"]["allocated_bytes"]
                / measurements["compressed"]["allocated_bytes"],
                17,
            )
            print(
                json.dumps(
                    {"dtype": str(dtype), "cache_memory": measurements, "ratio": ratio}
                ),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
