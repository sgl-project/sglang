"""GPU oracle: rewind SM70 CSA2 by restoring only the ring and pending.

A stale ratio-2 pending pair or SWA ring still produces fluent attention.
This checks the stop-point resume path, not a full-state clone: prefill to a stop, copy
the ring and pending through ``Csa2BoundaryStore``, overwrite them (and the
compressed rows past the stop), restore, extend the real suffix, and match a
one-shot prefill. Tolerance is the T=700 decode oracle (hidden atol/rtol
2e-2, top-k Jaccard 0.95).

Skips unless the device is SM70 with a couple of GiB free. Do not run this
against a tight live TP8 job if the allocator is already near the edge.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsv4 import sm70_csa2_reference as R
from sglang.srt.layers.attention.dsv4.sm70_csa2 import (
    get_state,
    sm70_commit_target_verify,
    sm70_forward_low_ratio_sources,
    sm70_forward_sparse,
)
from sglang.srt.layers.attention.dsv4.sm70_csa2_boundary import (
    csa2_finish_forward,
    csa2_prepare_decode,
    csa2_prepare_extend,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=180,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

FP16 = torch.float16
ATOL = 2e-2
RTOL = 2e-2
LAYER_ORDER = (2, 20, 24)
TOPK_LAYERS = (20, 24)
# Longer than the 128-slot ring, so a restore that forgot the ring cannot pass.
POLLUTE = 160
GAMMA_BLOCK = 8
CHUNK = 2048


def _require_sm70():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")
    if torch.cuda.get_device_capability()[0] != 7:
        raise unittest.SkipTest("SM70 required")
    free, _ = torch.cuda.mem_get_info()
    if free < 2 * (1 << 30):
        raise unittest.SkipTest(
            f"need >=2 GiB free GPU memory, have {free / (1 << 30):.1f} GiB"
        )


def _lin(v, weight):
    return (v.float() @ weight.float().t()).to(v.dtype)


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = set(int(i) for i in a.reshape(-1).tolist() if i >= 0)
    sb = set(int(i) for i in b.reshape(-1).tolist() if i >= 0)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


class TestSm70Csa2BoundaryOracle(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _require_sm70()
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        torch.set_num_threads(4)
        cls.dev = torch.device("cuda")
        g = torch.Generator(device="cpu").manual_seed(7)
        hidden, heads, q_lora_r, n_ih = 64, 4, 32, 8
        cls.hidden = hidden
        cls.heads = heads
        cls.q_lora_r = q_lora_r
        # 701-token stop + a 2048-token suffix chunk + a short tail.
        n_pos = 701 + CHUNK + 64
        cls.freqs = R.precompute_freqs_cis(64, n_pos + 8, 0, 10000.0, 1.0, 32.0, 1.0)[
            :n_pos
        ].to(cls.dev)
        cls.wkv = (torch.randn(512, hidden, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.kv_norm = torch.ones(512, dtype=FP16, device=cls.dev)
        cls.wq_b = (torch.randn(n_ih * 128, q_lora_r, generator=g).to(FP16) * 0.02).to(
            cls.dev
        )
        cls.w_proj = (torch.randn(n_ih, hidden, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.wk = (torch.randn(128, 512, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.k_norm = torch.ones(128, dtype=FP16, device=cls.dev)
        cls.wkv_c = (torch.randn(512, hidden, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.wgate = (torch.randn(512, hidden, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.norm_c = torch.ones(512, dtype=FP16, device=cls.dev)
        cls.sink = torch.zeros(heads, dtype=torch.float32, device=cls.dev)
        cls.x = (torch.randn(n_pos, hidden, generator=g).to(FP16) * 0.05).to(cls.dev)
        cls.q_lora = (torch.randn(n_pos, q_lora_r, generator=g).to(FP16) * 0.05).to(
            cls.dev
        )
        cls.q = (torch.randn(n_pos, heads, 512, generator=g).to(FP16) * 0.1).to(cls.dev)
        gj = torch.Generator(device="cpu").manual_seed(99)
        cls.junk_x = (torch.randn(POLLUTE, hidden, generator=gj).to(FP16) * 0.2).to(
            cls.dev
        )
        cls.junk_ql = (torch.randn(POLLUTE, q_lora_r, generator=gj).to(FP16) * 0.2).to(
            cls.dev
        )
        cls.junk_q = (torch.randn(POLLUTE, heads, 512, generator=gj).to(FP16) * 0.2).to(
            cls.dev
        )
        ga = torch.Generator(device="cpu").manual_seed(123)
        cls.alt_x = (torch.randn(POLLUTE, hidden, generator=ga).to(FP16) * 0.05).to(
            cls.dev
        )
        cls.alt_ql = (torch.randn(POLLUTE, q_lora_r, generator=ga).to(FP16) * 0.05).to(
            cls.dev
        )
        cls.alt_q = (torch.randn(POLLUTE, heads, 512, generator=ga).to(FP16) * 0.1).to(
            cls.dev
        )

    def _indexer(self, owns_k, is_src, uses_cand):
        return SimpleNamespace(
            owns_k=owns_k,
            is_candidate_source=is_src,
            uses_candidates=uses_cand,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            index_topk=512,
            wk=SimpleNamespace(weight=self.wk),
            k_norm=SimpleNamespace(weight=self.k_norm),
            queries=lambda ql, fr: R.fake_quant_fp4_ue8m0(
                R.rope_tail(_lin(ql, self.wq_b).view(ql.shape[0], 8, 128), fr, 64)
            ),
            head_weights=lambda xx: _lin(xx, self.w_proj) * (128**-0.5 * 8**-0.5),
        )

    def _layer(self, lid, ratio, compressor, indexer):
        return SimpleNamespace(
            layer_id=lid,
            compress_ratio=ratio,
            head_dim=512,
            qk_rope_head_dim=64,
            eps=1e-20,
            sliding_window=128,
            softmax_scale=512**-0.5,
            freqs_cis=self.freqs,
            wkv=lambda v: (_lin(v, self.wkv), None),
            kv_norm=SimpleNamespace(weight=self.kv_norm),
            compressor=compressor,
            indexer=indexer,
            attn_sink=self.sink,
        )

    def _layers(self):
        c1 = SimpleNamespace(
            wkv=SimpleNamespace(weight=self.wkv_c),
            norm=SimpleNamespace(weight=self.norm_c),
        )
        c2 = SimpleNamespace(
            wkv=SimpleNamespace(weight=self.wkv_c),
            wgate=SimpleNamespace(weight=self.wgate),
            norm=SimpleNamespace(weight=self.norm_c),
        )
        return {
            2: self._layer(2, 2, c2, self._indexer(True, False, False)),
            20: self._layer(20, 1, c1, self._indexer(True, True, False)),
            24: self._layer(24, 1, None, self._indexer(False, False, True)),
        }

    def _backend(self):
        return SimpleNamespace(
            model_runner=SimpleNamespace(decode_num_tokens_per_req=lambda: GAMMA_BLOCK),
            max_context_len=4096,
        )

    def _fb(self, decode: bool, target_verify: bool = False):
        return SimpleNamespace(
            batch_size=1,
            forward_mode=SimpleNamespace(
                is_decode=lambda: decode and not target_verify,
                is_extend=lambda: (not decode) and (not target_verify),
                is_target_verify=lambda: target_verify,
            ),
        )

    def _run(self, backend, layers, x, ql, q, positions, *, verify: bool = False):
        fb = self._fb(False, target_verify=verify)
        outs = {}
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend, layers[lid], x, ql, positions, fb
            )
            outs[lid] = sm70_forward_sparse(
                backend, q, layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        return outs

    def _extend(self, backend, layers, start: int, end: int, *, src: str = "real"):
        n = end - start
        if src == "real":
            x, ql, q = self.x[start:end], self.q_lora[start:end], self.q[start:end]
        elif src == "junk":
            x, ql, q = self.junk_x[:n], self.junk_ql[:n], self.junk_q[:n]
        elif src == "alt":
            x, ql, q = self.alt_x[:n], self.alt_ql[:n], self.alt_q[:n]
        else:
            raise AssertionError(src)
        positions = torch.arange(start, end, dtype=torch.int64, device=self.dev)
        return self._run(backend, layers, x, ql, q, positions)

    def _maxdiff(self, a, b) -> float:
        return float((a.float() - b.float()).abs().max().item())

    def _check_rows(self, ref, got, tag: str):
        for lid in LAYER_ORDER:
            d = self._maxdiff(ref[lid], got[lid])
            print(f"{tag} L{lid} maxabs={d:.5f}")
            self.assertFalse(torch.isnan(got[lid]).any().item(), f"{tag} L{lid} nan")
            self.assertTrue(
                torch.allclose(ref[lid].float(), got[lid].float(), atol=ATOL, rtol=RTOL),
                f"{tag} L{lid} maxabs={d}",
            )

    def _check_topk(self, ref_st, got_st, tag: str):
        for lid in TOPK_LAYERS:
            jac = _jaccard(got_st.prefill_topk[lid][-1], ref_st.prefill_topk[lid][-1])
            print(f"{tag} L{lid} topk Jaccard={jac:.4f}")
            self.assertGreaterEqual(jac, 0.95, f"{tag} L{lid} topk Jaccard {jac}")
        jac = _jaccard(got_st.prefill_cand_ids[-1], ref_st.prefill_cand_ids[-1])
        print(f"{tag} cand Jaccard={jac:.4f}")
        self.assertGreaterEqual(jac, 0.95, f"{tag} cand Jaccard {jac}")

    def _rewind_suffix(self, stop: int, suffix: int, *, chunk: int | None):
        """Prefill ``stop``, pollute past the ring, restore, extend the real suffix."""
        end = stop + suffix
        ref_be = self._backend()
        ref_layers = self._layers()
        ref = self._extend(ref_be, ref_layers, 0, end)
        ref_st = get_state(ref_be)

        be = self._backend()
        layers = self._layers()
        self._extend(be, layers, 0, stop)
        csa2_finish_forward([be], stop, from_extend=True)
        # Freeze the stop before the next extend overwrites the live ring.
        csa2_prepare_extend([be], stop)
        self._extend(be, layers, stop, stop + POLLUTE, src="junk")
        csa2_finish_forward([be], stop + POLLUTE, from_extend=True)
        csa2_prepare_extend([be], stop)

        if chunk is None:
            got = self._extend(be, layers, stop, end)
            self._check_rows(
                {lid: ref[lid][stop:] for lid in LAYER_ORDER},
                got,
                f"stop={stop} suffix={suffix}",
            )
        else:
            pos = stop
            while pos < end:
                nxt = min(pos + chunk, end)
                got = self._extend(be, layers, pos, nxt)
                self._check_rows(
                    {lid: ref[lid][pos:nxt] for lid in LAYER_ORDER},
                    got,
                    f"stop={stop} chunk={pos}:{nxt}",
                )
                # The worker freezes this chunk, then the next extend sees
                # start == resident and must not rewind mid-suffix.
                csa2_finish_forward([be], nxt, from_extend=True)
                if nxt < end:
                    csa2_prepare_extend([be], nxt)
                pos = nxt
        self._check_topk(ref_st, get_state(be), f"stop={stop} suffix={suffix}")

    def test_even_and_odd_stop_match_oneshot(self):
        # 700 is even: the next token does not read pending. 701 is odd: it does.
        self._rewind_suffix(700, 32, chunk=None)
        self._rewind_suffix(701, 32, chunk=None)

    def test_chunked_suffix_2048_matches_oneshot(self):
        self._rewind_suffix(701, CHUNK + 32, chunk=CHUNK)

    def test_short_verify_accept_then_rewind(self):
        """Commit fewer verify tokens than the block, then rewind to that stop.

        The rejected tail must not survive in the pending pair. ``stop`` is
        odd so the suffix's first token pools with the restored pending.
        """
        stop0 = 700
        commit_n = 3
        block = GAMMA_BLOCK
        self.assertLess(commit_n, block)
        stop = stop0 + commit_n
        self.assertEqual(stop % 2, 1)
        suffix = 32
        end = stop + suffix

        ref_be = self._backend()
        ref_layers = self._layers()
        ref = self._extend(ref_be, ref_layers, 0, end)

        be = self._backend()
        layers = self._layers()
        self._extend(be, layers, 0, stop0)
        csa2_finish_forward([be], stop0, from_extend=True)
        csa2_prepare_decode([be])
        x = self.x[stop0 : stop0 + block].clone()
        ql = self.q_lora[stop0 : stop0 + block].clone()
        q = self.q[stop0 : stop0 + block].clone()
        g = torch.Generator(device="cpu").manual_seed(1000)
        n_junk = block - commit_n
        x[commit_n:] = (torch.randn(n_junk, self.hidden, generator=g).to(FP16) * 0.2).to(
            self.dev
        )
        ql[commit_n:] = (
            torch.randn(n_junk, self.q_lora_r, generator=g).to(FP16) * 0.2
        ).to(self.dev)
        q[commit_n:] = (
            torch.randn(n_junk, self.heads, 512, generator=g).to(FP16) * 0.2
        ).to(self.dev)
        positions = torch.arange(stop0, stop0 + block, dtype=torch.int64, device=self.dev)
        self._run(be, layers, x, ql, q, positions, verify=True)
        sm70_commit_target_verify(
            be,
            torch.tensor([commit_n], dtype=torch.int32, device=self.dev),
            num_positions=block,
        )
        csa2_finish_forward([be], stop, from_extend=False)
        csa2_prepare_extend([be], stop)
        self._extend(be, layers, stop, stop + POLLUTE, src="junk")
        csa2_finish_forward([be], stop + POLLUTE, from_extend=True)
        csa2_prepare_extend([be], stop)
        got = self._extend(be, layers, stop, end)
        self._check_rows(
            {lid: ref[lid][stop:] for lid in LAYER_ORDER},
            got,
            f"verify commit={commit_n}/{block}",
        )
        self._check_topk(get_state(ref_be), get_state(be), f"verify commit={commit_n}")

    def _verify_step(self, be, layers, start: int, real: int, commit: int, seed: int):
        """One verify block: ``real`` true inputs, then junk; publish ``commit``."""
        block = GAMMA_BLOCK
        x = self.x[start : start + block].clone()
        ql = self.q_lora[start : start + block].clone()
        q = self.q[start : start + block].clone()
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = block - real
        x[real:] = (torch.randn(n, self.hidden, generator=g).to(FP16) * 0.2).to(self.dev)
        ql[real:] = (torch.randn(n, self.q_lora_r, generator=g).to(FP16) * 0.2).to(
            self.dev
        )
        q[real:] = (torch.randn(n, self.heads, 512, generator=g).to(FP16) * 0.2).to(
            self.dev
        )
        positions = torch.arange(start, start + block, dtype=torch.int64, device=self.dev)
        self._run(be, layers, x, ql, q, positions, verify=True)
        sm70_commit_target_verify(
            be,
            torch.tensor([commit], dtype=torch.int32, device=self.dev),
            num_positions=block,
        )
        csa2_finish_forward([be], start + commit, from_extend=False, start=start)

    def _cut_to_pin(self, *, overlap: bool, check: bool = True) -> float:
        """Verify past the stop, resume at the pin, match a one-shot prefill.

        The request stops on an accepted draft at an odd pin, so the
        suffix's first token pools with pending. The tokens published past
        the pin are not the next turn's, as in a live session. Returns the
        largest row difference.
        """
        stop0, pin, suffix = 700, 703, 32
        end = pin + suffix
        ref_be = self._backend()
        ref = self._extend(ref_be, self._layers(), 0, end)

        be = self._backend()
        layers = self._layers()
        self._extend(be, layers, 0, stop0)
        csa2_finish_forward([be], stop0, from_extend=True)
        csa2_prepare_decode([be])
        self._verify_step(be, layers, stop0, pin - stop0, 5, seed=1001)
        if overlap:
            self._verify_step(be, layers, stop0 + 5, 0, 3, seed=1002)
        csa2_prepare_extend([be], pin)
        got = self._extend(be, layers, pin, end)
        tag = f"cut to pin overlap={overlap}"
        if check:
            self._check_rows({lid: ref[lid][pin:] for lid in LAYER_ORDER}, got, tag)
            self._check_topk(get_state(ref_be), get_state(be), tag)
        return max(self._maxdiff(ref[lid][pin:], got[lid]) for lid in LAYER_ORDER)

    def test_verify_past_the_pin_is_cut_back(self):
        self._cut_to_pin(overlap=False)

    def test_overlap_step_past_the_pin_is_cut_back(self):
        self._cut_to_pin(overlap=True)

    def test_relabel_past_the_pin_is_measurably_off(self):
        """Control: without the cut the same resume drifts from the oracle.

        These small weights keep the drift inside the oracle tolerance, so
        compare it with the cut, which matches the one-shot rows exactly.
        """
        from unittest import mock

        from sglang.srt.layers.attention.dsv4.sm70_csa2_boundary import (
            Csa2BoundaryStore,
        )

        cut = self._cut_to_pin(overlap=False, check=False)
        with mock.patch.object(Csa2BoundaryStore, "trim_tip", return_value=False):
            relabel = self._cut_to_pin(overlap=False, check=False)
        print(f"cut maxabs={cut:.6f} relabel maxabs={relabel:.6f}")
        self.assertLess(cut, 1e-3)
        self.assertGreater(relabel, 5e-3)

    def test_disk_session_then_suffix_matches_oneshot(self):
        """Spill the image, clobber it, load it, then extend the real suffix."""
        import tempfile

        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.sm70_csa2_session import (
            request_load,
            request_spill,
            reset_handoff,
            session_key,
        )

        stop, suffix = 701, 32
        end = stop + suffix
        ref_be = self._backend()
        ref = self._extend(ref_be, self._layers(), 0, end)
        be = self._backend()
        layers = self._layers()
        self._extend(be, layers, 0, stop)
        csa2_finish_forward([be], stop, from_extend=True)
        csa2_prepare_extend([be], stop)
        directory = tempfile.mkdtemp()
        key = session_key(list(range(stop)), None, None)
        envs.SGLANG_DSV41_CSA2_SESSION_DIR.set(directory)
        try:
            reset_handoff()
            request_spill(key)
            csa2_prepare_extend([be], 0)
            state = get_state(be)
            for rows in state.kv_rows.values():
                rows.fill_(0)
            for ring in state.swa_ring.values():
                ring.fill_(0)
            self._extend(be, layers, 0, POLLUTE, src="junk")
            request_load(key)
            csa2_prepare_extend([be], stop)
            got = self._extend(be, layers, stop, end)
            self._check_rows(
                {lid: ref[lid][stop:] for lid in LAYER_ORDER},
                got,
                "disk session suffix",
            )
            self._check_topk(get_state(ref_be), get_state(be), "disk session suffix")
        finally:
            envs.SGLANG_DSV41_CSA2_SESSION_DIR.clear()
            reset_handoff()

    def test_prefix0_drops_the_stop_and_reprefill_matches(self):
        be = self._backend()
        layers = self._layers()
        self._extend(be, layers, 0, POLLUTE)
        csa2_finish_forward([be], POLLUTE, from_extend=True)
        csa2_prepare_extend([be], 0)
        with self.assertRaises(RuntimeError):
            csa2_prepare_extend([be], POLLUTE)
        got = self._extend(be, layers, 0, POLLUTE, src="alt")

        ref_be = self._backend()
        ref = self._extend(ref_be, self._layers(), 0, POLLUTE, src="alt")
        self._check_rows(ref, got, "prefix0 reprefill")
        self._check_topk(get_state(ref_be), get_state(be), "prefix0 reprefill")
