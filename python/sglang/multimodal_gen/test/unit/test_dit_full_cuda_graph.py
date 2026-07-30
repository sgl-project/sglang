# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DiT full-forward CUDA graph runner (--dit-cuda-graph full)
and the tensor-tree helpers it shares with the breakable-graph runner."""

import unittest

import torch

from sglang.multimodal_gen.runtime.utils.graph_tensor_tree import (
    clone_output,
    flatten_kwargs,
    map_tensors,
    signature_kwargs,
    static_buffer_like,
)


class TestGraphTensorTree(unittest.TestCase):
    def test_signature_covers_nested_tensors_and_controls(self):
        """A change hidden in a nested container or a non-tensor control must
        change the key — DiT kwargs carry both (qwen freqs_cis, txt_seq_lens)."""
        base = {
            "hidden_states": torch.zeros(2, 4),
            "freqs_cis": (torch.zeros(4, 8), torch.zeros(4, 8)),
            "txt_seq_lens": [3, 5],
            "guidance": 3.5,
        }
        self.assertEqual(signature_kwargs(base), signature_kwargs(dict(base)))

        nested_shape = dict(base, freqs_cis=(torch.zeros(4, 8), torch.zeros(6, 8)))
        self.assertNotEqual(signature_kwargs(base), signature_kwargs(nested_shape))

        control = dict(base, txt_seq_lens=[3, 7])
        self.assertNotEqual(signature_kwargs(base), signature_kwargs(control))

        scalar = dict(base, guidance=4.0)
        self.assertNotEqual(signature_kwargs(base), signature_kwargs(scalar))

    def test_flatten_collects_nested_leaves_deterministically(self):
        kwargs = {
            "b": [torch.zeros(1), {"y": torch.zeros(2), "x": torch.zeros(3)}],
            "a": torch.zeros(4),
        }
        leaves = flatten_kwargs(kwargs)
        self.assertEqual([tuple(t.shape) for t in leaves], [(4,), (1,), (3,), (2,)])
        self.assertEqual(
            [tuple(t.shape) for t in flatten_kwargs(kwargs)],
            [tuple(t.shape) for t in leaves],
        )

    def test_map_tensors_rebuilds_structure(self):
        kwargs = {"t": (torch.zeros(2), [torch.zeros(3)]), "flag": True}
        mapped = map_tensors(kwargs, lambda t: torch.ones_like(t))
        self.assertIs(mapped["flag"], True)
        self.assertTrue(torch.equal(mapped["t"][0], torch.ones(2)))
        self.assertTrue(torch.equal(mapped["t"][1][0], torch.ones(3)))

    def test_clone_output_detaches_from_static_buffer(self):
        static = {"sample": torch.zeros(2), "extra": [torch.zeros(2)]}
        cloned = clone_output(static)
        static["sample"].fill_(9)
        static["extra"][0].fill_(9)
        self.assertTrue(torch.equal(cloned["sample"], torch.zeros(2)))
        self.assertTrue(torch.equal(cloned["extra"][0], torch.zeros(2)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graph capture needs a GPU")
class TestFullGraphRunner(unittest.TestCase):
    """Capture/replay behaviour of _FullGraphRunner on a toy module."""

    def setUp(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
            _FullGraphRunner,
        )

        self.runner_cls = _FullGraphRunner
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _make_runner(self, model):
        return self.runner_cls(model, self.device)

    def test_serial_cfg_branches_do_not_alias(self):
        """Serial CFG calls the runner twice per step and holds both results
        while combining them; the second replay must not overwrite the first."""

        def model(hidden_states):
            return hidden_states * 2

        runner = self._make_runner(model)
        pos_in = torch.ones(4, device=self.device)
        neg_in = torch.full((4,), 3.0, device=self.device)

        for _ in range(3):  # eager step, capture step, then replay steps
            pos = runner.run({"hidden_states": pos_in})
            neg = runner.run({"hidden_states": neg_in})
            self.assertTrue(torch.equal(pos, torch.full((4,), 2.0, device=self.device)))
            self.assertTrue(torch.equal(neg, torch.full((4,), 6.0, device=self.device)))
        self.assertIsNotNone(runner.graph)

    def test_consecutive_requests_replay_fresh_inputs(self):
        """A second request reuses the graph; its own inputs must reach it."""

        def model(hidden_states, freqs_cis):
            return hidden_states + freqs_cis[0]

        runner = self._make_runner(model)
        freqs = (torch.ones(4, device=self.device),)
        for value in (1.0, 2.0, 5.0, 7.0):
            out = runner.run(
                {
                    "hidden_states": torch.full((4,), value, device=self.device),
                    "freqs_cis": freqs,
                }
            )
            self.assertTrue(
                torch.equal(out, torch.full((4,), value + 1.0, device=self.device))
            )
        self.assertIsNotNone(runner.graph)

    def test_nested_tensor_inputs_are_copied_on_replay(self):
        """Tensors inside containers get static buffers too (the shallow copy
        this replaces left them pointing at the captured request's data)."""

        def model(hidden_states, freqs_cis):
            return hidden_states + freqs_cis[0] + freqs_cis[1]["bias"]

        runner = self._make_runner(model)
        for value in (1.0, 2.0, 4.0, 6.0):
            out = runner.run(
                {
                    "hidden_states": torch.zeros(4, device=self.device),
                    "freqs_cis": (
                        torch.full((4,), value, device=self.device),
                        {"bias": torch.full((4,), value, device=self.device)},
                    ),
                }
            )
            self.assertTrue(
                torch.equal(out, torch.full((4,), 2 * value, device=self.device))
            )

    def test_signature_change_falls_back_to_eager(self):
        def model(hidden_states):
            return hidden_states * 2

        runner = self._make_runner(model)
        for _ in range(3):
            runner.run({"hidden_states": torch.ones(4, device=self.device)})
        out = runner.run({"hidden_states": torch.ones(8, device=self.device)})
        self.assertTrue(runner.disabled)
        self.assertTrue(torch.equal(out, torch.full((8,), 2.0, device=self.device)))

    def test_cpu_leaf_gets_device_static_buffer(self):
        """A host-built timestep must not leave an unpinned H2D copy inside the
        captured region."""
        buf = static_buffer_like(torch.zeros(2), self.device)
        self.assertEqual(buf.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
