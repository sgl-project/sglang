"""DFlash KDA draft layers on the target's Triton KDA kernels against
independent FP32 SpecForge math: prefill, chunked-prefill and verify commits,
proposals, and CUDA graph replay over a changed state."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

HIDDEN, HEADS, HEAD_DIM, CONV, BLOCK, EPS = 256, 4, 64, 4, 8, 1e-6
PROJ = HEADS * HEAD_DIM
NUM_SLOTS = 5
DEVICE = "cuda"


def _export(context_state, lower_bound):
    return dict(
        architectures=["DFlashDraftModel"],
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=64,
        rms_norm_eps=EPS,
        num_target_layers=2,
        dflash_config={
            "attention_modes": ["kda", "gqa"],
            "block_size": BLOCK,
            "target_layer_ids": [0],
        },
        linear_attn_config={
            "context_state": context_state,
            "gate_lower_bound": lower_bound,
            "head_dim": HEAD_DIM,
            "num_heads": HEADS,
            "short_conv_kernel_size": CONV,
        },
    )


def _bf16_exact(shape, std):
    return (torch.randn(shape) * std).bfloat16().float()


def _specforge_weights():
    """SpecForge Qwen3DFlashKDAAttention tensors, exactly representable in BF16."""
    w = {
        f"{name}.weight": _bf16_exact((PROJ, HIDDEN), HIDDEN**-0.5)
        for name in ("q_proj", "k_proj", "v_proj")
    }
    w.update({f"{qkv}_conv1d.weight": _bf16_exact((PROJ, CONV), 0.4) for qkv in "qkv"})
    w["A_log"] = torch.empty(HEADS).uniform_(1, 16).log()
    w["dt_bias"] = _bf16_exact((PROJ,), 0.5)
    w["b_proj.weight"] = _bf16_exact((HEADS, HIDDEN), HIDDEN**-0.5)
    for gate in ("f", "g"):
        w[f"{gate}_a_proj.weight"] = _bf16_exact((HEAD_DIM, HIDDEN), HIDDEN**-0.5)
        w[f"{gate}_b_proj.weight"] = _bf16_exact((PROJ, HEAD_DIM), HEAD_DIM**-0.5)
    w["o_norm.weight"] = 1 + _bf16_exact((HEAD_DIM,), 0.1)
    w["o_proj.weight"] = _bf16_exact((HIDDEN, PROJ), PROJ**-0.5)
    return {name: tensor.to(DEVICE) for name, tensor in w.items()}


# ---- SpecForge reference math (specforge/modeling/draft/kda.py) --------------
def _short_conv(x, weight):
    padded = F.pad(x.t()[None], (CONV - 1, 0))
    return F.silu(F.conv1d(padded, weight[:, None], groups=weight.shape[0]))[0].t()


def _recurrence(q, k, v, raw_gate, raw_beta, w, lower_bound):
    """reference_kda over one sequence; returns outputs and the final [H, K, V] state."""
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    beta = raw_beta.sigmoid()
    gate_input = raw_gate + w["dt_bias"].view(HEADS, HEAD_DIM)
    decay = w["A_log"].exp().view(HEADS, 1)
    if lower_bound is None:
        log_decay = -decay * F.softplus(gate_input)
    else:
        log_decay = lower_bound * torch.sigmoid(decay * gate_input)
    state = q.new_zeros(HEADS, HEAD_DIM, HEAD_DIM)
    outputs = []
    for t in range(q.shape[0]):
        state = state * log_decay[t].exp()[..., None]
        delta = (v[t] - torch.einsum("hk,hkv->hv", k[t], state)) * beta[t][:, None]
        state = state + torch.einsum("hk,hv->hkv", k[t], delta)
        outputs.append(torch.einsum("hk,hkv->hv", q[t], state) * HEAD_DIM**-0.5)
    return torch.stack(outputs) if outputs else q.new_zeros(q.shape), state


def _reference(w, context, block, lower_bound):
    """The layer output at the block rows and the state after the context rows."""
    rows = torch.cat((context, block)).float()

    def proj(name, x=rows):
        return x @ w[f"{name}.weight"].t()

    q, k, v = (
        _short_conv(proj(f"{qkv}_proj"), w[f"{qkv}_conv1d.weight"]).view(
            -1, HEADS, HEAD_DIM
        )
        for qkv in "qkv"
    )
    raw_gate = proj("f_b_proj", proj("f_a_proj")).view(-1, HEADS, HEAD_DIM)
    raw_beta = proj("b_proj")
    n = context.shape[0]
    _, state = _recurrence(
        q[:n], k[:n], v[:n], raw_gate[:n], raw_beta[:n], w, lower_bound
    )
    out, _ = _recurrence(q, k, v, raw_gate, raw_beta, w, lower_bound)
    out = out[n:]
    gate = proj("g_b_proj", proj("g_a_proj"))[n:].view(-1, HEADS, HEAD_DIM)
    out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + EPS)
    out = out * w["o_norm.weight"] * gate.sigmoid()
    return proj("o_proj", out.flatten(1)), state


def _assert_close(test, actual, expected, msg):
    rel = (actual.float() - expected).norm() / expected.norm()
    test.assertLess(rel.item(), 0.02, msg)


class TestDFlashKDAKernels(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        torch.manual_seed(0)

    def _layer(self, context_state, lower_bound, w):
        """The draft layer as the model loader builds it: BF16, on the GPU."""
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(DEVICE):
                model = DFlashDraftModel(
                    Qwen3Config(**_export(context_state, lower_bound))
                )
        finally:
            torch.set_default_dtype(dtype)
        model.load_weights(
            (f"layers.0.self_attn.{name}", tensor.cpu()) for name, tensor in w.items()
        )
        return model.layers[0].self_attn

    @staticmethod
    def _propose(attn, slots, block):
        batch = SimpleNamespace(
            logical_forward_mode=ForwardMode.TARGET_VERIFY, req_pool_indices=slots
        )
        positions = torch.zeros(block.shape[0], dtype=torch.int64, device=DEVICE)
        with torch.no_grad():
            return attn(positions, block, batch)

    def _check_scan(self, lower_bound):
        w = _specforge_weights()
        attn = self._layer("scan", lower_bound, w)
        attn.allocate_context_state(num_slots=NUM_SLOTS, device=torch.device(DEVICE))
        # Stale state of earlier occupants; untouched slots must keep it.
        attn.attn.conv_state.fill_(1.0)
        attn.attn.ssm_state.fill_(1.0)
        slots = torch.tensor([3, 1], device=DEVICE)
        context = [_bf16_exact((5, HIDDEN), 1.0), _bf16_exact((11, HIDDEN), 1.0)]
        context = [c.to(DEVICE, torch.bfloat16) for c in context]

        def commit_extend(rows, req_slots, prefix_lens, extend_lens):
            with torch.no_grad():
                attn.write_context_state(
                    hidden_states=torch.cat(rows),
                    req_pool_indices=torch.tensor(req_slots, device=DEVICE),
                    prefix_lens=torch.tensor(prefix_lens, device=DEVICE),
                    extend_lens=torch.tensor(
                        extend_lens, dtype=torch.int32, device=DEVICE
                    ),
                    extend_lens_cpu=extend_lens,
                )

        # Two new requests, then the second request's next prefill chunk.
        commit_extend([context[0], context[1][:7]], [3, 1], [0, 0], [5, 7])
        commit_extend([context[1][7:]], [1], [7], [4])

        def check(block, msg):
            out = self._propose(attn, slots, block)
            for i, slot in enumerate(slots.tolist()):
                ref_out, ref_state = _reference(
                    w, context[i], block[i * BLOCK : (i + 1) * BLOCK], lower_bound
                )
                _assert_close(
                    self, out[i * BLOCK : (i + 1) * BLOCK], ref_out, f"{msg} out {i}"
                )
                _assert_close(
                    self,
                    attn.attn.ssm_state[slot].transpose(-1, -2),
                    ref_state,
                    f"{msg} state {i}",
                )
            return out

        block = _bf16_exact((2 * BLOCK, HIDDEN), 1.0).to(DEVICE, torch.bfloat16)
        state = (attn.attn.conv_state.clone(), attn.attn.ssm_state.clone())
        check(block, "after prefill")
        # A proposal reads the committed state and never writes it.
        torch.testing.assert_close(attn.attn.conv_state, state[0], rtol=0, atol=0)
        torch.testing.assert_close(attn.attn.ssm_state, state[1], rtol=0, atol=0)

        static_block = block.clone()
        graph = torch.cuda.CUDAGraph()
        self._propose(attn, slots, static_block)  # Triton autotune outside capture.
        with torch.cuda.graph(graph):
            graph_out = self._propose(attn, slots, static_block)

        # Verify rows: request 0 commits 3 of them, request 1 the whole block.
        verified = _bf16_exact((2 * BLOCK, HIDDEN), 1.0).to(DEVICE, torch.bfloat16)
        commit_lens = torch.tensor([3, BLOCK], dtype=torch.int32, device=DEVICE)
        with torch.no_grad():
            attn.write_context_state(
                hidden_states=verified, req_pool_indices=slots, commit_lens=commit_lens
            )
        for i, n in enumerate(commit_lens.tolist()):
            context[i] = torch.cat((context[i], verified[i * BLOCK : i * BLOCK + n]))
        eager_out = check(block, "after verify")

        graph.replay()
        torch.testing.assert_close(graph_out, eager_out, rtol=1e-3, atol=1e-3)
        for slot in (0, 2, 4):
            self.assertTrue(bool((attn.attn.ssm_state[slot] == 1.0).all()), slot)
            self.assertTrue(bool((attn.attn.conv_state[slot] == 1.0).all()), slot)

    def test_scan_softplus_gate(self):
        self._check_scan(lower_bound=None)

    def test_scan_safe_gate(self):
        self._check_scan(lower_bound=-5.0)

    def test_reset_blocks_start_from_zero(self):
        w = _specforge_weights()
        attn = self._layer("reset", -5.0, w)
        slots = torch.tensor([1, 2], device=DEVICE)
        block = _bf16_exact((2 * BLOCK, HIDDEN), 1.0).to(DEVICE, torch.bfloat16)
        out = self._propose(attn, slots, block)
        empty = block.new_zeros((0, HIDDEN))
        for i in range(2):
            ref_out, _ = _reference(w, empty, block[i * BLOCK : (i + 1) * BLOCK], -5.0)
            _assert_close(self, out[i * BLOCK : (i + 1) * BLOCK], ref_out, f"out {i}")


if __name__ == "__main__":
    unittest.main()
