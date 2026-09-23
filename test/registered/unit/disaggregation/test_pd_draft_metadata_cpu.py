"""Source-isolated CPU tensor tests; not a replacement for PD transport tests."""

import ast
import logging
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]


def source_symbols(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    # Compile only checked-in declarations, without importing the CUDA runtime.
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace


class TestDraftMetadataCPU(unittest.TestCase):
    def setUp(self):
        self.spec = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=2,
            enable_multi_layer_eagle=False,
            speculative_use_rejection_sampling=True,
        )
        envs = SimpleNamespace(
            SGLANG_ENABLE_DISAGG_SAMPLING_MASK=SimpleNamespace(get=lambda: False),
            SGLANG_MOONCAKE_CUSTOM_MEM_POOL=SimpleNamespace(get=lambda: None),
        )
        self.ns = dict(
            torch=torch,
            nullcontext=nullcontext,
            envs=envs,
            is_npu=lambda: False,
            logger=logging.getLogger("metadata_cpu"),
            get_spec=lambda: self.spec,
            should_remap_pd_dsa_seed_to_local_slots=lambda: False,
            EagleDraftInput=lambda **kw: SimpleNamespace(**kw),
            CaptureHiddenMode=SimpleNamespace(LAST="last"),
        )
        source_symbols(
            "python/sglang/srt/disaggregation/utils.py", {"MetadataBuffers"}, self.ns
        )
        source_symbols(
            "python/sglang/srt/speculative/eagle_disaggregation.py",
            {"build_eagle_disagg_draft_input"},
            self.ns,
        )

    def buffer(self, width=4, **kwargs):
        return self.ns["MetadataBuffers"](
            2,
            2,
            torch.float32,
            max_sampling_mask_tokens=16,
            output_draft_probs_dim=width,
            **kwargs,
        )

    def request(self, probs):
        return SimpleNamespace(
            metadata_buffer_index=0,
            output_ids=[2],
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            multimodal_inputs=None,
            return_logprob=False,
            return_sampling_mask=False,
            hidden_states_tensor=torch.tensor([1.0, 2.0]),
            output_topk_p=torch.tensor([0.4]),
            output_topk_index=torch.tensor([3]),
            output_draft_probs=probs,
            output_dsa_topk_indices=None,
            bootstrap_room=9,
        )

    def test_tensor_roundtrip_preserves_probability_and_slot(self):
        b = self.buffer()
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])
        b.set_buf(self.request(q))
        result = b.get_buf(0)[12]
        self.assertTrue(torch.equal(result, q))
        result.zero_()
        self.assertTrue(torch.equal(b.output_draft_probs[0], q))

    def test_disabled_buffer_does_not_change_wire(self):
        off, on = self.buffer(0), self.buffer()
        self.assertIsNone(off.output_draft_probs)
        self.assertIsNone(off.get_buf(0)[12])
        self.assertEqual(len(on.get_buf_infos()[0]), len(off.get_buf_infos()[0]) + 1)

    def test_checksum_and_dsa_remain_after_draft_probs(self):
        for checksum in (False, True):
            for dsa in (0, 3):
                with self.subTest(checksum=checksum, dsa=dsa):
                    b = self.buffer(
                        kv_checksum_enabled=checksum, output_dsa_topk_indices_dim=dsa
                    )
                    pointers, _, sizes = b.get_buf_infos()
                    i = pointers.index(b.output_draft_probs.data_ptr())
                    self.assertEqual(sizes[i], 16)
                    self.assertEqual(
                        pointers[i + 1],
                        (
                            b.output_dsa_topk_indices if dsa else b.bootstrap_room
                        ).data_ptr(),
                    )
                    self.assertEqual(
                        pointers[-1],
                        (b.kv_checksum if checksum else b.bootstrap_room).data_ptr(),
                    )

    def test_missing_and_wrong_shape_rejected(self):
        for q in (None, torch.ones(3), torch.ones(1, 4)):
            with self.subTest(q=q):
                with self.assertRaises(RuntimeError):
                    self.buffer().set_buf(self.request(q))

    def test_decode_stacks_full_proposals(self):
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])
        batch = SimpleNamespace(
            reqs=[self.request(q), self.request(q.flip(0))],
            device="cpu",
            enable_overlap=False,
        )
        seed = self.ns["build_eagle_disagg_draft_input"](
            batch, torch.tensor([2, 2]), None
        )
        self.assertTrue(torch.equal(seed.draft_probs, torch.stack([q, q.flip(0)])))

    def test_decode_rejects_missing_proposal(self):
        batch = SimpleNamespace(
            reqs=[self.request(None)], device="cpu", enable_overlap=False
        )
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.ns["build_eagle_disagg_draft_input"](batch, torch.tensor([2]), None)

    def test_disabled_sampling_does_not_require_proposal(self):
        self.spec.speculative_use_rejection_sampling = False
        batch = SimpleNamespace(
            reqs=[self.request(None)], device="cpu", enable_overlap=False
        )
        seed = self.ns["build_eagle_disagg_draft_input"](batch, torch.tensor([2]), None)
        self.assertIsNone(seed.draft_probs)


if __name__ == "__main__":
    unittest.main()
