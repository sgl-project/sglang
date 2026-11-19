import unittest
from unittest.mock import patch

import test_dsa_indexer as base
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import flashindexer
from sglang.srt.layers.attention.dsa.dsa_fp4_indexer import Fp4Indexer
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="nightly", runner_config="4-gpu-b200")


def _flashindexer_select_heads_ref(q, weights, gate_weight, bias, hidden, topk_heads):
    # Torch reference: rank by gate + bias, keep q rows, reweight by the raw gate.
    gates = (hidden @ gate_weight.t()).float()
    topk_idx = (gates + bias).topk(topk_heads, dim=-1).indices
    q_sel = q.gather(1, topk_idx[..., None].expand(-1, -1, q.shape[-1]))
    w_sel = weights.gather(1, topk_idx) * gates.gather(1, topk_idx)
    return q_sel, w_sel.to(weights.dtype)


def _flashindexer_inputs(tokens, heads, hidden_size, q_dtype):
    q = torch.randn(tokens, heads, 128, device="cuda").to(q_dtype)
    weights = torch.rand(tokens, heads, device="cuda")
    # Positive hidden and per-head gate rows keep the gate ranking tie-free.
    hidden = torch.rand(tokens, hidden_size, device="cuda").add_(0.1).bfloat16()
    rank = torch.randperm(heads, device="cuda").float().add_(1)
    gate_weight = (rank[:, None] / hidden_size).expand(heads, hidden_size)
    # A spread-out bias dominates the ranking, so it must change which heads win.
    bias = torch.randperm(heads, device="cuda").float().mul_(100)
    return q, weights, gate_weight.bfloat16().contiguous(), bias, hidden


class TestDSAIndexerFp4(CustomTestCase):
    setUpClass = classmethod(base.TestDSAIndexer.setUpClass.__func__)
    setUp = base.TestDSAIndexer.setUp
    _init_model_runner = base.TestDSAIndexer._init_model_runner
    _create_indexer = base.TestDSAIndexer._create_indexer
    _create_forward_batch = base.TestDSAIndexer._create_forward_batch
    _verify_topk_output = base.TestDSAIndexer._verify_topk_output

    def test_forward_extend_then_decode_fp4(self):
        self._run_extend_then_decode()

    def test_forward_extend_then_decode_fp4_with_flashindexer(self):
        self._check_flashindexer_on_extend_only(fp4=True)

    def test_forward_extend_then_decode_fp8_with_flashindexer(self):
        self._check_flashindexer_on_extend_only(fp4=False)

    def _check_flashindexer_on_extend_only(self, fp4):
        # FlashIndexer selects heads on the ragged extend only; paged decode keeps all.
        with patch.object(
            flashindexer,
            "flashindexer_select_heads_fused",
            wraps=flashindexer.flashindexer_select_heads_fused,
        ) as select:
            self._run_extend_then_decode(fp4=fp4, gate=True)
        self.assertEqual(select.call_count, 1)
        q = select.call_args.args[0]
        self.assertEqual(q.dtype, torch.bfloat16 if fp4 else torch.float8_e4m3fn)

    def test_flashindexer_fused_select_matches_torch(self):
        # FP8 q gathers 128-byte rows; Fp4Indexer's bf16 q gathers 256-byte rows.
        heads, hidden_size, topk = 64, 256, flashindexer.FLASHINDEXER_TOPK_HEADS
        for q_dtype in (torch.float8_e4m3fn, torch.bfloat16):
            # Odd counts leave a partial BLOCK_M tile for the row masks.
            for tokens in (1, 5, 33, 100):
                with self.subTest(q_dtype=q_dtype, tokens=tokens):
                    inputs = _flashindexer_inputs(tokens, heads, hidden_size, q_dtype)
                    fused_q, fused_w = flashindexer.flashindexer_select_heads_fused(
                        *inputs
                    )
                    ref_q, ref_w = _flashindexer_select_heads_ref(*inputs, topk)
                    self.assertEqual(fused_q.shape, (tokens, topk, 128))
                    self.assertEqual(fused_q.dtype, q_dtype)
                    torch.testing.assert_close(
                        fused_q.view(torch.uint8),
                        ref_q.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(fused_w, ref_w, rtol=1e-2, atol=1e-2)

    def _run_extend_then_decode(self, fp4=True, gate=False):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("DeepGEMM FP4 MQA requires SM100")

        self.enterContext(get_context().override_server_args(dsa_topk_backend="torch"))
        if fp4:
            self.enterContext(patch.object(base, "Indexer", Fp4Indexer))
        self.enterContext(envs.SGLANG_DSA_FUSE_TOPK.override(False))
        self.batch_size = 1
        self._init_model_runner({"use_fp4_indexer": True} if fp4 else None)
        indexer = self._create_indexer()
        if gate:
            with torch.device(self.device):
                flashindexer.attach_flashindexer_gate(indexer)
        for parameter in indexer.parameters():
            torch.nn.init.normal_(parameter, std=0.02)

        for forward_mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            forward_batch = self._create_forward_batch(forward_mode)
            forward_batch.out_cache_loc += self.config["page_size"]
            self.backend.init_forward_metadata(forward_batch)
            q_len = self.seq_len if forward_mode == ForwardMode.EXTEND else 1
            hidden_states = torch.randn(
                q_len,
                self.config["hidden_size"],
                dtype=self.dtype,
                device=self.device,
            )
            q_lora = torch.randn(
                q_len,
                self.config["q_lora_rank"],
                dtype=self.dtype,
                device=self.device,
            )
            seq_len = forward_batch.seq_lens_cpu[0].item()
            positions = torch.arange(seq_len - q_len, seq_len, device=self.device)

            topk_indices = indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.config["layer_id"],
            )

            self._verify_topk_output(
                topk_indices, self.batch_size, q_len, self.config["index_topk"]
            )
            self.assertEqual(topk_indices.dtype, torch.int32)
            torch.testing.assert_close(
                (topk_indices >= 0).sum(dim=-1),
                (positions + 1).clamp(max=self.config["index_topk"]),
            )


if __name__ == "__main__":
    unittest.main()
