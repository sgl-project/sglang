"""CPU contracts for the GLM packed handoff and greedy candidate ownership."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.hardware_backend.npu.megakernel import draft, plugin
from sglang.srt.hardware_backend.npu.megakernel import glm52_packed_handoff as packed
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, stage="base-a", runner_config="cpu")


class TestGlm52Handoff(unittest.TestCase):
    def test_cross_page_and_reuse(self):
        table = torch.arange(256).view(1, -1)
        table[:, :128] += 384
        table[:, 128:] += 512
        req_pool = SimpleNamespace(req_to_token=table)
        key = torch.empty(6, 128, 1, 656, dtype=torch.float8_e4m3fn)
        index = torch.empty(6, 128, 1, 128, dtype=key.dtype)
        scale = torch.empty(6, 128, 1)
        pool = SimpleNamespace(
            page_size=128,
            start_layer=0,
            layer_num=2,
            index_head_dim=128,
            kv_cache_dim=656,
            indexer_layer_ids=(0,),
            indexer_layer_id_to_slot={0: 0},
            get_key_buffer=lambda i: key,
            get_index_k_buffer=lambda i: index,
            get_index_k_scale_buffer=lambda i: scale,
        )
        rope = SimpleNamespace(cos_sin_cache=torch.ones(512, 64, dtype=torch.bfloat16))
        attn = SimpleNamespace(
            rotary_emb=rope, indexer=SimpleNamespace(rotary_emb=rope)
        )
        model = SimpleNamespace(
            model=SimpleNamespace(decoder=SimpleNamespace(self_attn=attn))
        )
        bridge = packed.Glm52PackedDecodeHandoff(model)
        with (
            patch.object(packed, "get_token_to_kv_pool", return_value=pool),
            patch.object(packed, "get_req_to_token_pool", return_value=req_pool),
            patch.dict(
                sys.modules,
                {
                    "blockrt.models.glm_5_2.model_args": SimpleNamespace(
                        ForwardBatchInfo=SimpleNamespace
                    )
                },
            ),
        ):
            addresses = {}
            for start in (126, 1, 128):
                for rows in (1, 8):
                    positions = torch.arange(start, start + rows)
                    batch = SimpleNamespace(
                        batch_size=1,
                        req_pool_indices=torch.tensor([0]),
                        spec_info=None,
                        out_cache_loc=table[0, positions],
                    )
                    build = bridge.build if rows == 1 else bridge.build_target_verify
                    info, ids, pos = build(batch, positions, positions)
                    self.assertEqual(info.kv_seq_len.tolist(), [start + rows])
                    self.assertEqual(info.q_seq_len.tolist(), [rows])
                    self.assertEqual(
                        info.slot_mapping.flatten().tolist(),
                        table[0, positions].tolist(),
                    )
                    pages = [3] if start + rows <= 128 else [3, 5]
                    self.assertEqual(
                        info.block_tables.tolist(), [pages + [-1] * (128 - len(pages))]
                    )
                    self.assertEqual(
                        info.index_k_buffer[1].data_ptr(), index.data_ptr()
                    )
                    self.assertEqual(
                        info.index_k_scale_buffer[1].data_ptr(), scale.data_ptr()
                    )
                    self.assertIs(info.k_cache[0], key)
                    address = (
                        ids.data_ptr(),
                        pos.data_ptr(),
                        info.block_tables.data_ptr(),
                    )
                    self.assertEqual(address, addresses.setdefault(rows, address))

    def test_replay_output_has_one_consumer(self):
        for accepted in (1, 3, 8):
            req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3))
            batch = SimpleNamespace(reqs=[req])
            positions = torch.arange(124, 132)
            tokens = torch.arange(7).reshape(1, 7)
            hidden = torch.ones(1, 4)
            binding = SimpleNamespace(
                mode="fused_extend",
                steps=7,
                hit=lambda route: None,
                bridge=SimpleNamespace(
                    model=SimpleNamespace(config=SimpleNamespace(hidden_size=4))
                ),
                fused=lambda *_: SimpleNamespace(
                    token_ids=tokens, first_hidden_states=hidden
                ),
            )
            seed = SimpleNamespace()
            draft.publish(
                binding,
                batch,
                SimpleNamespace(positions=positions),
                torch.tensor([accepted - 1]),
                seed,
            )
            tokens.zero_()
            hidden.zero_()
            self.assertEqual(seed.hidden_states.tolist(), [[1] * 4])
            next_batch = SimpleNamespace(positions=torch.tensor([124 + accepted]))
            with patch.object(
                draft,
                "candidate_present",
                side_effect=lambda present, valid: present and valid,
            ):
                chain = draft.draft_chain(None, binding, batch, next_batch)
                self.assertEqual(chain.tolist(), [list(range(7))])
                self.assertIsNone(draft.draft_chain(None, binding, batch, next_batch))

    def test_sampling_admission(self):
        sampling = SimpleNamespace(is_all_greedy=True)
        batch = SimpleNamespace(sampling_info=sampling, return_logprob=False)
        self.assertTrue(plugin.greedy(batch))
        for field, value in (
            ("grammars", [object()]),
            ("logit_bias", torch.zeros(1)),
            ("return_sampling_masks", [True]),
        ):
            with patch.object(sampling, field, value, create=True):
                self.assertFalse(plugin.greedy(batch))
        batch.return_logprob = True
        self.assertFalse(plugin.greedy(batch))


if __name__ == "__main__":
    unittest.main()
