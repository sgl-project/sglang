"""Unit test: windowed draft-decode KV index builder (StreamingLLM sink + window).

Covers the ``window_size`` / ``sink_size`` additions to
``generate_draft_decode_kv_indices`` (the shared kv-index kernel used by the
built-in MTP/NEXTN + EAGLE draft-decode path on BOTH the Triton and FlashInfer
draft attention backends).

Properties checked:

  1. IDENTITY (off-by-default is a no-op) -- ``window_size == 0`` reproduces the
     full-KV read plan. Both the CSR offsets (``kv_indptr``) and the gathered
     slots (``kv_indices``) are checked against an independent closed-form oracle
     derived only from the input ``seq_lens`` (NOT from the kernel's packing
     logic), so an offset bug is not mirrored by the oracle. This is the
     losslessness/regression guarantee.
  2. KV-INDPTR lengths -- with a window each draft-decode step keeps
     ``min(seq_len, sink + window)`` base tokens + ``it + 1`` never-windowed tree
     tokens; compared to a clamp+cumsum oracle. Parametrized past the
     256K/512K/1M thresholds (incl. the perf-smoke W4032/S64 params).
  3. CONTENT -- the actual gathered slots are, in order,
     ``[first sink] + [most-recent window] + [draft tree]`` (the StreamingLLM
     layout), read back at buffer offset 0 (num_seqs=1, topk=1).
  3b. TREE (``topk > 1``) -- with a tree draft every ``(request, topk)`` slot gets
     its own copy of the windowed base list plus its own branch of tree tokens.
     The windowed per-slot write offset is ``topk_id * (kept + it + 1)``, so a
     windowing bug there would silently overlap neighbouring branches while the
     lengths still look right. Slices are located by the *oracle* ``kv_indptr``
     (independent closed form) and their content checked, which ties the offset
     and the CSR bounds together. Covered for ``page_size == 1`` and for the
     ``page_size > 1 and topk > 1`` paged-tree branch, where the tree tokens are
     read from the UNCAPPED ``seq_len`` but written at the capped offset.

  4. RESOLUTION -- ``resolve_draft_decode_window`` maps server args to the
     kernel's (window, sink), and returns (0, 0) for a draft model that already
     has a sliding window of its own (that per-layer window wins over the flag,
     since this index builder emits one KV list for all draft layers). It warns
     only when it substitutes a different window than the one requested.
  5. BACKEND WIRING -- both ``TritonMultiStepDraftBackend`` and
     ``FlashInferMultiStepDraftBackend`` resolve the pair in ``__init__`` and
     forward it to the kernel, so windowing is honored identically on both
     backends (CPU-only guard).

The kernel is imported normally from ``sglang`` (no by-file-path staging), so it
tracks the installed tree.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.spec_utils import (
    generate_draft_decode_kv_indices,
    resolve_draft_decode_window,
)
from sglang.srt.utils import next_power_of_2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()


# --- Harness: mirror {Triton,FlashInfer}MultiStepDraftBackend.common_template ---
def _make_inputs(seq_lens, num_steps, topk, page_size=1, pool_len=None):
    dev = "cuda"
    num_seqs = len(seq_lens)
    bs = num_seqs * topk
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64, device=dev)
    max_seq = int(seq_lens_t.max().item())
    if pool_len is None:
        # Room for every branch's tree tokens: topk*num_steps slots (page_size==1)
        # or up to topk*num_new_pages*page_size past the prefix's last page.
        pool_len = max_seq + topk * (num_steps + page_size) + 16
    # Unique physical slot ids per (req, position) so gather checks are unambiguous.
    req_to_token = (
        torch.arange(num_seqs * pool_len, dtype=torch.int64, device=dev).reshape(
            num_seqs, pool_len
        )
        + 1000
    )
    req_pool_indices = torch.arange(num_seqs, dtype=torch.int64, device=dev)
    positions = torch.empty(bs, dtype=torch.int64, device=dev)
    for bid in range(num_seqs):
        for tk in range(topk):
            positions[bid * topk + tk] = seq_lens[bid]

    width = bs * (max_seq + num_steps) + 16
    kv_indices = torch.zeros((num_steps, width), dtype=torch.int64, device=dev)
    kv_indptr = torch.zeros((num_steps, bs + 1), dtype=torch.int64, device=dev)
    return dict(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        seq_lens=seq_lens_t,
        positions=positions,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        pool_len=pool_len,
        page_size=page_size,
        num_seqs=num_seqs,
        topk=topk,
        num_steps=num_steps,
        bs=bs,
    )


def _run(io, window_size=0, sink_size=0):
    kv_indices = io["kv_indices"].clone()
    kv_indptr = io["kv_indptr"].clone()
    generate_draft_decode_kv_indices[(io["num_steps"], io["num_seqs"], io["topk"])](
        io["req_pool_indices"],
        io["req_to_token"],
        io["seq_lens"],
        kv_indices,
        kv_indptr,
        io["positions"],
        io["pool_len"],
        kv_indices.shape[1],
        kv_indptr.shape[1],
        next_power_of_2(io["num_seqs"]),
        next_power_of_2(io["num_steps"]),
        next_power_of_2(io["bs"]),
        io["page_size"],
        window_size,
        sink_size,
    )
    torch.cuda.synchronize()
    return kv_indices, kv_indptr


def _expected_kv_indptr(io, cap=None):
    """Closed form: kv_indptr[it][zid] = sum(clamp(pos[:zid], cap)) + zid*(it+1)."""
    pos = io["positions"].to("cpu")
    if cap is not None:
        pos = torch.clamp(pos, max=cap)
    csum = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(pos, 0)])
    out = torch.zeros((io["num_steps"], io["bs"] + 1), dtype=torch.int64)
    for it in range(io["num_steps"]):
        for zid in range(io["bs"] + 1):
            out[it, zid] = csum[zid] + zid * (it + 1)
    return out


def _expected_base(r2t, seq_len, window, sink):
    """StreamingLLM base gather for one request (window==0 => full context)."""
    if window == 0:
        return r2t[0:seq_len]
    cap = window + sink
    kept = min(seq_len, cap)
    s_eff = min(sink, seq_len)
    recent_start = seq_len - (kept - s_eff)
    return torch.cat([r2t[0:s_eff], r2t[recent_start : recent_start + (kept - s_eff)]])


def _expected_tree(r2t, seq_len, topk_id, topk, num_steps, n_tree, page_size):
    """This branch's tree tokens, always read from the UNCAPPED prefix end.

    Mirrors the kernel's two extend layouts (linear, and the paged per-topk page
    stride), both of which windowing must leave untouched.
    """
    if page_size == 1 or topk == 1:
        start = seq_len + topk_id * num_steps
    else:
        last_page_len = seq_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        start = (
            (seq_len // page_size) * page_size
            + topk_id * num_new_pages_per_topk * page_size
            + last_page_len
        )
    return r2t[start : start + n_tree]


@unittest.skipUnless(_HAS_CUDA, "draft-decode kv-index kernel requires a CUDA GPU")
class TestDraftDecodeWindowKernel(unittest.TestCase):
    def test_window_zero_is_full_kv(self):
        """window_size==0 => original full-KV read plan (indptr + content)."""
        for seq_lens, num_steps, topk in [
            ([128], 4, 1),
            ([300, 130, 517], 5, 1),
            ([1024, 777], 4, 2),
        ]:
            with self.subTest(seq_lens=seq_lens, num_steps=num_steps, topk=topk):
                io = _make_inputs(seq_lens, num_steps, topk)
                self._assert_slot_layout(io, 0, 0)

    def test_windowed_kv_indptr_lengths(self):
        for seq_lens, window, sink in [
            ([5000, 2048, 900], 1024, 64),
            ([5000, 2048, 900], 1024, 0),  # pure recent window
            ([100, 200], 4096, 64),  # cap exceeds seq_len -> keep all
            # long-context regression (perf smoke used W4032/S64 past 256K/512K/1M)
            ([1029306], 4032, 64),
            ([1048576, 300000, 70000], 4032, 64),
            ([262145, 131072], 4032, 64),
        ]:
            with self.subTest(seq_lens=seq_lens, window=window, sink=sink):
                io = _make_inputs(seq_lens, num_steps=4, topk=1)
                _, indptr = _run(io, window_size=window, sink_size=sink)
                expected = _expected_kv_indptr(io, cap=window + sink)
                self.assertTrue(torch.equal(indptr.cpu(), expected))

    def test_windowed_content_sink_plus_recent(self):
        for seq_len, window, sink in [
            (5000, 1024, 64),
            (65536, 4032, 64),
            (262145, 4032, 64),  # just past native 262144
            (524288, 4032, 64),
            (1029306, 4032, 64),  # exact perf-smoke seq_len (W4032/S64)
        ]:
            with self.subTest(seq_len=seq_len, window=window, sink=sink):
                io = _make_inputs([seq_len], num_steps=4, topk=1)
                kv_indices, _ = _run(io, window_size=window, sink_size=sink)
                r2t = io["req_to_token"][0].cpu()
                kept = min(seq_len, window + sink)
                expected_base = _expected_base(r2t, seq_len, window, sink)
                for step in range(io["num_steps"]):
                    row = kv_indices[step].cpu()
                    self.assertTrue(
                        torch.equal(row[0:kept], expected_base),
                        f"base gather wrong at step {step} (seq_len={seq_len})",
                    )
                    n_tree = step + 1
                    self.assertTrue(
                        torch.equal(
                            row[kept : kept + n_tree], r2t[seq_len : seq_len + n_tree]
                        ),
                        f"tree tokens wrong at step {step} (seq_len={seq_len})",
                    )

    def _assert_slot_layout(self, io, window, sink):
        """Every (request, topk) slot holds [sink + recent] + its own tree branch.

        Slots are located by the closed-form kv_indptr oracle, so this also pins
        the per-slot write offset (topk_id * (kept + iters)). ``topk == 1`` is a
        one-branch tree and ``window == 0`` asks for the whole prefix, so this
        covers the identity case too.
        """
        cap = None if window == 0 else window + sink
        kv_indices, indptr = _run(io, window_size=window, sink_size=sink)
        expected_indptr = _expected_kv_indptr(io, cap=cap)
        self.assertTrue(torch.equal(indptr.cpu(), expected_indptr))

        topk, num_steps = io["topk"], io["num_steps"]
        seq_lens = io["seq_lens"].tolist()
        for step in range(num_steps):
            row = kv_indices[step].cpu()
            n_tree = step + 1
            for bid, seq_len in enumerate(seq_lens):
                r2t = io["req_to_token"][bid].cpu()
                kept = seq_len if cap is None else min(seq_len, cap)
                for topk_id in range(topk):
                    zid = bid * topk + topk_id
                    start = int(expected_indptr[step, zid])
                    where = f"step={step} bid={bid} topk_id={topk_id}"
                    self.assertTrue(
                        torch.equal(
                            row[start : start + kept],
                            _expected_base(r2t, seq_len, window, sink),
                        ),
                        f"base gather wrong at {where}",
                    )
                    self.assertTrue(
                        torch.equal(
                            row[start + kept : start + kept + n_tree],
                            _expected_tree(
                                r2t,
                                seq_len,
                                topk_id,
                                topk,
                                num_steps,
                                n_tree,
                                io["page_size"],
                            ),
                        ),
                        f"tree tokens wrong at {where}",
                    )

    def test_tree_topk_layout(self):
        """topk > 1 (tree draft): per-branch slices, windowed and unwindowed."""
        for seq_lens, num_steps, topk, window, sink in [
            ([600], 4, 2, 128, 16),
            ([600, 250], 3, 4, 128, 16),
            ([5000, 900], 4, 2, 1024, 64),
            ([5000, 900], 4, 2, 1024, 0),  # pure recent window
            ([300, 130], 3, 2, 4096, 64),  # cap exceeds seq_len -> keep all
            ([65536, 1029306], 3, 2, 4032, 64),  # long-context tree
            ([600, 250], 3, 4, 0, 0),  # off-by-default control
        ]:
            with self.subTest(seq_lens=seq_lens, topk=topk, window=window, sink=sink):
                io = _make_inputs(seq_lens, num_steps, topk)
                self._assert_slot_layout(io, window, sink)

    def test_tree_topk_paged(self):
        """page_size > 1 AND topk > 1: the paged-tree extend branch.

        The tree tokens are read from the uncapped prefix end with a per-topk page
        stride and stored at the CAPPED offset, so this is the one place where the
        windowed and unwindowed lengths must both be respected in one statement.
        """
        for seq_lens, num_steps, topk, page_size, window, sink in [
            ([600], 4, 2, 4, 128, 16),
            ([601, 255], 3, 2, 8, 128, 16),  # prefix not page-aligned
            ([604, 256], 3, 4, 4, 128, 0),  # page-aligned prefix, no sink
            ([5000, 900], 4, 2, 16, 1024, 64),
            ([601, 255], 3, 2, 8, 0, 0),  # off-by-default control
        ]:
            with self.subTest(
                seq_lens=seq_lens, topk=topk, page_size=page_size, window=window
            ):
                io = _make_inputs(seq_lens, num_steps, topk, page_size=page_size)
                self._assert_slot_layout(io, window, sink)


class TestResolveDraftDecodeWindow(unittest.TestCase):
    """CPU-only: server args -> (window, sink), incl. the opt-out for an SWA draft."""

    @staticmethod
    def _runner(window=None, sink=None, native_window=None, declared=None):
        server_args = SimpleNamespace(
            speculative_draft_window_size=window,
            speculative_draft_sink_size=sink,
        )
        if declared is not None:
            server_args._resolved_overrides = (
                ("handle_speculative_decoding", declared),
            )
        return SimpleNamespace(
            server_args=server_args,
            sliding_window_size=native_window,
        )

    def test_unset_is_full_attention(self):
        self.assertEqual(resolve_draft_decode_window(self._runner()), (0, 0))

    def test_window_only(self):
        self.assertEqual(
            resolve_draft_decode_window(self._runner(window=4032)), (4032, 0)
        )

    def test_window_and_sink(self):
        self.assertEqual(
            resolve_draft_decode_window(self._runner(window=4032, sink=64)), (4032, 64)
        )

    def test_declared_values_win_over_the_raw_fields(self):
        """handle_speculative_decoding declares these fields instead of assigning
        them, so a raw field read would answer with the unvalidated input."""
        runner = self._runner(
            window=None,
            sink=None,
            declared={
                "speculative_draft_window_size": 4032,
                "speculative_draft_sink_size": 64,
            },
        )
        self.assertEqual(resolve_draft_decode_window(runner), (4032, 64))

    def test_native_sliding_window_draft_opts_out(self):
        """A draft with its own window keeps it: the flag must not override it."""
        with mock.patch.object(spec_utils.logger, "warning") as warn:
            self.assertEqual(
                resolve_draft_decode_window(
                    self._runner(window=4032, sink=64, native_window=1024)
                ),
                (0, 0),
            )
        warn.assert_called_once()

    def test_native_window_equal_to_flag_is_quiet(self):
        """LlamaForCausalLMEagle3 routes this flag into its own window, so an equal
        native window is the requested one applied per layer, not a conflict."""
        with mock.patch.object(spec_utils.logger, "warning") as warn:
            self.assertEqual(
                resolve_draft_decode_window(
                    self._runner(window=4032, sink=64, native_window=4032)
                ),
                (0, 0),
            )
        warn.assert_not_called()

    def test_no_native_window_still_windows(self):
        """Guard must not fire for full-attention drafts (0 / None / absent)."""
        for native in (None, 0):
            with self.subTest(native_window=native):
                self.assertEqual(
                    resolve_draft_decode_window(
                        self._runner(window=4032, sink=64, native_window=native)
                    ),
                    (4032, 64),
                )
        runner = self._runner(window=4032, sink=64)
        del runner.sliding_window_size
        self.assertEqual(resolve_draft_decode_window(runner), (4032, 64))


class TestDraftDecodeWindowBackendWiring(unittest.TestCase):
    """CPU-only guard: both draft backends must forward window/sink to the kernel.

    The kernel tests above prove correctness once the args reach the kernel; this
    guards the per-backend plumbing (init resolution + kernel passthrough) so it
    cannot silently regress on either backend without a GPU-loaded model.
    """

    def _assert_backend_wires(self, cls):
        init_src = inspect.getsource(cls.__init__)
        self.assertIn("resolve_draft_decode_window", init_src)
        self.assertIn("self.draft_window_size", init_src)
        self.assertIn("self.draft_sink_size", init_src)
        tmpl_src = inspect.getsource(cls.common_template)
        self.assertIn("self.draft_window_size", tmpl_src)
        self.assertIn("self.draft_sink_size", tmpl_src)

    def test_triton_backend_wires_window_sink(self):
        try:
            from sglang.srt.layers.attention.triton_backend import (
                TritonMultiStepDraftBackend,
            )
        except ImportError as e:  # pragma: no cover
            self.skipTest(f"triton backend import unavailable: {e}")
        self._assert_backend_wires(TritonMultiStepDraftBackend)

    def test_flashinfer_backend_wires_window_sink(self):
        try:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )
        except ImportError as e:  # pragma: no cover
            self.skipTest(f"flashinfer backend import unavailable: {e}")
        self._assert_backend_wires(FlashInferMultiStepDraftBackend)


if __name__ == "__main__":
    unittest.main()
