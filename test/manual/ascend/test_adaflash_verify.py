import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import torch
from sglang.srt.hardware_backend.npu.utils import init_npu_backend
from sglang.srt.speculative import adaflash_verify
from sglang.srt.speculative.adaflash_verify import AdaFlashVerify
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from torch import nn

init_npu_backend()


class FixedRatios(nn.Module):
    def forward(self, hidden):
        return hidden[:, :1, 0]


def verifier():
    result = AdaFlashVerify.__new__(AdaFlashVerify)
    result.model = SimpleNamespace(thresh_head_two_model=FixedRatios())
    result.block_size = 16
    result.num_steps = 0
    result.num_verify_tokens = 0
    result.num_full_width_tokens = 0
    result.last_verify_lens = None
    return result


class TestAdaFlashVerify(unittest.TestCase):
    def test_thinking_export_target_depth_alias(self):
        config = SimpleNamespace(
            num_hidden_layers=5,
            num_target_layers=5,
            target_num_hidden_layers=36,
            dflash_config={"target_layer_ids": [1, 9, 17, 25, 33]},
        )
        parsed = parse_dflash_draft_config(draft_hf_config=config)
        self.assertEqual(parsed.num_target_layers, 36)
        self.assertEqual(
            parsed.resolve_target_layer_ids(target_num_layers=parsed.num_target_layers),
            [1, 9, 17, 25, 33],
        )

    def test_eager_config_uses_backend_value_not_string_identity(self):
        backend = bytearray(b"disabled").decode()
        self.assertEqual(backend, adaflash_verify.Backend.DISABLED)
        worker = SimpleNamespace(
            draft_model=object.__new__(adaflash_verify.DFlashAdaFlashDraftModel),
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"])
                )
            ),
            selector=None,
            lilicorr=None,
        )
        runtime = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend=backend)
                )
            )
        )
        with (
            patch.object(adaflash_verify, "get_exec", return_value=runtime),
            patch.object(
                adaflash_verify,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=1, attn_dp_enabled=False),
            ),
            patch.object(
                adaflash_verify,
                "read_ragged_verify_mode",
                return_value=adaflash_verify.RaggedVerifyMode.COMPACT,
            ),
            patch.object(
                adaflash_verify.envs.SGLANG_SIMULATE_ACC_LEN, "get", return_value=0
            ),
            patch.object(
                adaflash_verify, "AdaFlashVerify", return_value=sentinel.adapter
            ),
        ):
            self.assertIs(
                adaflash_verify.make_adaflash_verify(worker), sentinel.adapter
            )

    def test_direct_lengths_keep_anchor_and_bound_each_request(self):
        for device in ["cpu", "npu"]:
            with self.subTest(device=device):
                adapter = verifier()
                hidden = torch.tensor([-1.0, 0.4375, 2.0], device=device).reshape(
                    3, 1, 1
                )
                layout = adapter.layout(hidden)
                self.assertEqual(layout.verify_lens.cpu().tolist(), [1, 7, 16])
                self.assertEqual(layout.qo_indptr_device.cpu().tolist(), [0, 1, 8, 24])
                self.assertEqual(layout.total_verify_tokens, 24)
                self.assertEqual(layout.graph_num_tokens, 24)
                self.assertEqual(adapter.num_verify_tokens, 24)
                self.assertEqual(adapter.num_full_width_tokens, 48)

    def test_matching_padding_cannot_extend_acceptance_past_verified_prefix(self):
        for device in ["cpu", "npu"]:
            with self.subTest(device=device):
                adapter = verifier()
                hidden = torch.tensor([0.0, 0.4375, 1.0], device=device).reshape(
                    3, 1, 1
                )
                layout = adapter.layout(hidden)
                # Token zero also wins every unmaterialized padding row. A plain
                # rectangular acceptance scan would incorrectly accept all 15.
                candidates = torch.zeros((3, 16), dtype=torch.int64, device=device)
                logits = torch.zeros((48, 4), device=device)
                logits[:, 0] = 1
                correct, bonus = adapter.accept(
                    candidates=candidates, logits=logits, layout=layout
                )
                self.assertEqual(correct.cpu().tolist(), [0, 6, 15])
                self.assertEqual(bonus.cpu().tolist(), [0, 0, 0])

    def test_bonus_comes_from_last_materialized_target_row(self):
        adapter = verifier()
        layout = adapter.layout(torch.tensor([0.4375], device="npu").reshape(1, 1, 1))
        candidates = torch.zeros((1, 16), dtype=torch.int64, device="npu")
        logits = torch.zeros((16, 4), device="npu")
        logits[:, 0] = 1
        logits[6, 2] = 3
        correct, bonus = adapter.accept(
            candidates=candidates, logits=logits, layout=layout
        )
        self.assertEqual(correct.cpu().tolist(), [6])
        self.assertEqual(bonus.cpu().tolist(), [2])


if __name__ == "__main__":
    unittest.main()
