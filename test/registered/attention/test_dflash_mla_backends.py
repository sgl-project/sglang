"""DFlash MLA drafts on real MLA backends, bidirectional and causal, eager and graph."""

import copy
import math
import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.dflash import DFlashMLAAttention
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import BumpAllocator
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    MockMLAModelRunner,
    TinyMLAModelConfig,
    _make_forward_batch,
)
from sglang.test.parallel_groups import publish

register_cuda_ci(est_time=40, stage="base-b", runner_config="4-gpu-b200")

HIDDEN, HEADS, RANK, NOPE, ROPE, VALUE, BLOCK = 128, 32, 512, 128, 64, 128, 8
CAPACITY = 256
BACKENDS = ("triton", "flashinfer")


def _rope(x, positions):
    """Interleaved-pair RoPE in FP32, the DeepSeek convention."""
    freq = positions.float()[:, None] / (
        10000 ** (torch.arange(0, ROPE, 2, device=x.device).float() / ROPE)
    )
    cos, sin = freq.cos().repeat_interleave(2, -1), freq.sin().repeat_interleave(2, -1)
    a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
    return x * cos[:, None] + torch.stack((-b, a), -1).flatten(-2) * sin[:, None]


def _expanded_reference(module, prefix, hidden, positions, *, causal):
    """Independent FP32 SpecForge equations with per-head K and V."""
    w = {name: p.float() for name, p in module.named_parameters()}

    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w[name]

    hidden = hidden.float()
    q_nope, q_pe = (
        F.linear(hidden, w["q_proj.weight"])
        .view(-1, HEADS, NOPE + ROPE)
        .split((NOPE, ROPE), -1)
    )
    q = torch.cat((q_nope, _rope(q_pe, positions)), -1)
    outputs = []
    for i, ctx in enumerate(prefix):
        x = torch.cat((ctx.float(), hidden[i * BLOCK : (i + 1) * BLOCK]))
        c, k_pe = F.linear(x, w["kv_a_proj_with_mqa.weight"]).split((RANK, ROPE), -1)
        k_nope, v = (
            F.linear(norm(c, "kv_a_layernorm.weight"), w["kv_b_proj.weight"])
            .view(-1, HEADS, NOPE + VALUE)
            .split((NOPE, VALUE), -1)
        )
        k_pe = _rope(k_pe[:, None], torch.arange(len(x), device=x.device))
        k = torch.cat((k_nope, k_pe.expand(-1, HEADS, -1)), -1)
        scores = torch.einsum("qhd,khd->hqk", q[i * BLOCK : (i + 1) * BLOCK], k)
        if causal:
            allowed = (
                torch.arange(len(x), device=x.device)[None]
                <= (len(ctx) + torch.arange(BLOCK, device=x.device))[:, None]
            )
            scores.masked_fill_(~allowed[None], -torch.inf)
        probs = (scores * (NOPE + ROPE) ** -0.5).softmax(-1)
        output = torch.einsum("hqk,khd->qhd", probs, v).flatten(1)
        outputs.append(F.linear(output, w["o_proj.weight"]))
    return torch.cat(outputs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDFlashMLABackends(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")

    def _make_runner(self, backend_name, config):
        model_config = TinyMLAModelConfig(
            num_heads=HEADS,
            kv_lora_rank=RANK,
            qk_rope_head_dim=ROPE,
            hidden_size=HIDDEN,
            context_len=CAPACITY,
        )
        model_config.hf_config = model_config.hf_text_config = config
        model_config.qk_nope_head_dim = NOPE
        model_config.head_dim = NOPE + ROPE
        model_config.v_head_dim = VALUE
        model_config.scaling = (NOPE + ROPE) ** -0.5
        case = MLAAttentionCase(
            name="dflash",
            backend=backend_name,
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=HEADS,
            page_size=32,
            prefix_lens=(0, 31),
            extend_lens=(BLOCK, BLOCK),
        )
        runner = MockMLAModelRunner(
            case=case,
            model_config=model_config,
            dtype=torch.bfloat16,
            device="cuda",
            max_context_len=CAPACITY,
            kv_lora_rank=RANK,
            qk_rope_head_dim=ROPE,
            disable_cuda_graph=False,
        )
        runner.is_draft_worker = True
        runner.spec_algorithm = SpeculativeAlgorithm.DFLASH
        runner.draft_attention_backend = backend_name
        # Backends read the resolved runtime namespaces, not server_args.
        get_context().override(
            "dflash_mla_backend_fixture",
            speculative_algorithm="DFLASH",
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=BLOCK,
            speculative_num_steps=BLOCK - 1,
        )
        return runner

    def _make_backend(self, backend_name, runner, module):
        # Backends read the mask from the attn_type of the model's layers.
        runner.model = module
        backend = ATTENTION_BACKENDS[backend_name](runner)
        # Normally stamped by ModelRunner, which this fixture bypasses.
        backend.prefill_attention_backend_str = backend_name
        backend.decode_attention_backend_str = backend_name
        return backend

    def _check(self, backend_name, *, causal):
        torch.manual_seed(20261016)
        config = Qwen3Config(
            hidden_size=HIDDEN,
            num_hidden_layers=1,
            num_attention_heads=HEADS,
            num_key_value_heads=1,
            kv_lora_rank=RANK,
            qk_nope_head_dim=NOPE,
            qk_rope_head_dim=ROPE,
            v_head_dim=VALUE,
            rms_norm_eps=1e-6,
            rope_theta=10000,
            max_position_embeddings=CAPACITY,
            is_causal=causal,
            layer_types=["full_attention"],
            dflash_config={"attention_mode": "mla", "block_size": BLOCK},
        )
        runner = self._make_runner(backend_name, config)
        module = DFlashMLAAttention(config, layer_id=0)
        # get_rope() shares one cached instance, and the CUDA RoPE kernel reads
        # an FP32 table that Module.to would round to BF16.
        module.rotary_emb = copy.deepcopy(module.rotary_emb)
        rope_cache = module.rotary_emb.cos_sin_cache
        module = module.to(device="cuda", dtype=torch.bfloat16).eval()
        module.rotary_emb.cos_sin_cache = rope_cache.to(device="cuda")
        with torch.no_grad():
            for name, param in module.named_parameters():
                if "layernorm" in name:
                    param.fill_(1)
                else:
                    param.normal_(std=1 / math.sqrt(param.shape[-1]))
        module.absorb_kv_b_proj()
        backend = self._make_backend(backend_name, runner, module)
        zero_allocator = BumpAllocator(
            buffer_size=2, dtype=torch.float32, device="cuda"
        )

        def make_inputs(prefix_lens):
            current = MLAAttentionCase(
                name="dflash",
                backend=backend_name,
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=HEADS,
                page_size=32,
                prefix_lens=prefix_lens,
                extend_lens=(BLOCK,) * len(prefix_lens),
            )
            batch = _make_forward_batch(
                current, runner, max_context_len=CAPACITY, device="cuda"
            )
            # As in DFlashWorkerV2: device lengths are committed prefixes, host
            # planning lengths include the in-flight block.
            batch.seq_lens = torch.tensor(prefix_lens, device="cuda", dtype=torch.int32)
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
            runner.runner_ids.bind_own_plan(batch)
            prefix = [
                torch.randn(n, HIDDEN, device="cuda", dtype=torch.bfloat16)
                for n in prefix_lens
            ]
            hidden = torch.randn(
                len(prefix_lens) * BLOCK, HIDDEN, device="cuda", dtype=torch.bfloat16
            )
            for i, ctx in enumerate(prefix):
                if len(ctx):
                    module.write_context_kv(
                        pool=runner.token_to_kv_pool,
                        positions=torch.arange(len(ctx), device="cuda"),
                        hidden_states=ctx,
                        cache_loc=runner.req_to_token_pool.req_to_token[
                            i, : len(ctx)
                        ].long(),
                    )
            expected = _expanded_reference(
                module, prefix, hidden, batch.positions, causal=causal
            )
            return batch, hidden, expected

        def check(actual, expected):
            self.assertTrue(torch.isfinite(actual).all())
            relative_l2 = (actual.float() - expected).norm() / expected.norm()
            self.assertLess(relative_l2.item(), 0.012)
            torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.025)

        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            batch, hidden, expected = make_inputs((0, 31))
            backend.init_forward_metadata(batch)
            check(module(batch.positions, hidden, batch, zero_allocator), expected)

            # Capture on a fresh batch, then replay with different prefixes so a
            # mask or plan baked in at capture would show up.
            batch, hidden, _ = make_inputs((0, 31))
            backend.init_cuda_graph_state(max_bs=2, max_num_tokens=2 * BLOCK)
            runner.runner_ids.bind_runner_slots(batch)
            backend.init_forward_metadata_out_graph(batch, in_capture=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    backend.init_forward_metadata_in_graph(batch)
                    module(batch.positions, hidden, batch, zero_allocator)
            torch.cuda.current_stream().wait_stream(stream)
            backend.on_after_cuda_graph_warmup()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                backend.init_forward_metadata_in_graph(batch)
                output = module(batch.positions, hidden, batch, zero_allocator)

            replay, next_hidden, expected = make_inputs((63, 32))
            hidden.copy_(next_hidden)
            for name in ("positions", "out_cache_loc", "seq_lens", "req_pool_indices"):
                getattr(batch, name).copy_(getattr(replay, name))
            batch.seq_lens_cpu.copy_(replay.seq_lens_cpu)
            batch.seq_lens_sum = replay.seq_lens_sum
            backend.init_forward_metadata_out_graph(replay)
            graph.replay()
            check(output, expected)

    def _check_backends(self, *, causal):
        for backend_name in BACKENDS:
            with self.subTest(backend=backend_name):
                self._check(backend_name, causal=causal)

    def test_bidirectional(self):
        self._check_backends(causal=False)

    def test_causal(self):
        self._check_backends(causal=True)


if __name__ == "__main__":
    unittest.main()
