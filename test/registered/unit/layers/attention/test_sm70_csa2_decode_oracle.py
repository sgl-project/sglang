"""GPU oracle: D3 CSA2 prefill vs decode, hierarchy, and CUDA-graph decode."""

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
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=90,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

FP16 = torch.float16
ATOL = 2e-2
RTOL = 2e-2
LAYER_ORDER = (2, 20, 24)


def _require_sm70():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")
    if torch.cuda.get_device_capability()[0] != 7:
        raise unittest.SkipTest("SM70 required")


def _lin(v, weight):
    return (v.float() @ weight.float().t()).to(v.dtype)


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = set(a[a >= 0].tolist())
    sb = set(b[b >= 0].tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


class TestSm70Csa2DecodeOracle(CustomTestCase):
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
        n_pos = 4096
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
        t = 712
        cls.x = (torch.randn(t, hidden, generator=g).to(FP16) * 0.05).to(cls.dev)
        cls.q_lora = (torch.randn(t, q_lora_r, generator=g).to(FP16) * 0.05).to(cls.dev)
        cls.q = (torch.randn(t, heads, 512, generator=g).to(FP16) * 0.1).to(cls.dev)
        cls.positions = torch.arange(t, dtype=torch.int64, device=cls.dev)

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
                R.rope_tail(
                    _lin(ql, self.wq_b).view(ql.shape[0], 8, 128), fr, 64
                )
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
        l2 = self._layer(2, 2, c2, self._indexer(True, False, False))
        l20 = self._layer(20, 1, c1, self._indexer(True, True, False))
        l24 = self._layer(24, 1, None, self._indexer(False, False, True))
        return {2: l2, 20: l20, 24: l24}

    def _backend(self):
        return SimpleNamespace(model_runner=None, max_context_len=4096)

    def _fb(self, decode: bool, target_verify: bool = False):
        return SimpleNamespace(
            batch_size=1,
            forward_mode=SimpleNamespace(
                is_decode=lambda: decode and not target_verify,
                is_extend=lambda: (not decode) and (not target_verify),
                is_target_verify=lambda: target_verify,
            ),
        )

    def _run_extend(self, backend, layers, sl: slice):
        fb = self._fb(False)
        outs = {}
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend,
                layers[lid],
                self.x[sl],
                self.q_lora[sl],
                self.positions[sl],
                fb,
            )
            outs[lid] = sm70_forward_sparse(
                backend, self.q[sl], layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        return outs

    def _run_decode(self, backend, layers, i: int):
        fb = self._fb(True)
        sl = slice(i, i + 1)
        outs = {}
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend,
                layers[lid],
                self.x[sl],
                self.q_lora[sl],
                self.positions[sl],
                fb,
            )
            outs[lid] = sm70_forward_sparse(
                backend, self.q[sl], layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        return outs

    def _maxdiff(self, a, b):
        return float((a.float() - b.float()).abs().max().item())

    def _check_last(self, prefill_outs, decode_outs, tag: str):
        for lid in LAYER_ORDER:
            a = prefill_outs[lid][-1:]
            b = decode_outs[lid]
            d = self._maxdiff(a, b)
            print(f"{tag} L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(a.float(), b.float(), atol=ATOL, rtol=RTOL),
                f"{tag} L{lid} maxabs={d}",
            )
            self.assertFalse(torch.isnan(b).any().item())

    def _scenario(self, t: int):
        layers_a = self._layers()
        be_a = self._backend()
        outs_a = self._run_extend(be_a, layers_a, slice(0, t))
        st_a = get_state(be_a)

        layers_b = self._layers()
        be_b = self._backend()
        self._run_extend(be_b, layers_b, slice(0, t - 1))
        outs_b = self._run_decode(be_b, layers_b, t - 1)
        st_b = get_state(be_b)
        self._check_last(outs_a, outs_b, f"T={t} a-vs-b")

        layers_c = self._layers()
        be_c = self._backend()
        if t - 1 > 301:
            self._run_extend(be_c, layers_c, slice(0, 301))
            self._run_extend(be_c, layers_c, slice(301, t - 1))
        else:
            self._run_extend(be_c, layers_c, slice(0, t - 1))
        outs_c = self._run_decode(be_c, layers_c, t - 1)
        self._check_last(
            {lid: outs_b[lid] for lid in LAYER_ORDER},
            outs_c,
            f"T={t} b-vs-c",
        )
        # b-vs-c compares decode to decode; reuse _check_last by faking prefill[-1:]
        for lid in LAYER_ORDER:
            d = self._maxdiff(outs_b[lid], outs_c[lid])
            self.assertTrue(
                torch.allclose(
                    outs_b[lid].float(), outs_c[lid].float(), atol=ATOL, rtol=RTOL
                ),
                f"T={t} b-vs-c L{lid} maxabs={d}",
            )

        extra = 3
        print(
            f"T={t} L20 topk jaccard={_jaccard(st_b.topk[20][0], st_a.prefill_topk[20][-1]):.4f} "
            f"cand jaccard={_jaccard(st_b.cand_ids[0], st_a.prefill_cand_ids[-1]):.4f} "
            f"L24 topk jaccard={_jaccard(st_b.topk[24][0], st_a.prefill_topk[24][-1]):.4f}"
        )
        self.assertGreaterEqual(
            _jaccard(st_b.topk[20][0], st_a.prefill_topk[20][-1]), 0.95
        )
        self.assertGreaterEqual(
            _jaccard(st_b.cand_ids[0], st_a.prefill_cand_ids[-1]), 0.95
        )
        self.assertGreaterEqual(
            _jaccard(st_b.topk[24][0], st_a.prefill_topk[24][-1]), 0.95
        )
        for i in range(t, t + extra):
            more = self._run_decode(be_b, layers_b, i)
            for lid in LAYER_ORDER:
                self.assertTrue(torch.isfinite(more[lid].float()).all().item())
        layers_d = self._layers()
        be_d = self._backend()
        outs_d = self._run_extend(be_d, layers_d, slice(0, t + extra))
        layers_e = self._layers()
        be_e = self._backend()
        self._run_extend(be_e, layers_e, slice(0, t + extra - 1))
        outs_e = self._run_decode(be_e, layers_e, t + extra - 1)
        self._check_last(outs_d, outs_e, f"T={t}+3 a-vs-decode")

    def test_prefill_vs_decode_t700_and_t9(self):
        self._scenario(700)
        self._scenario(9)

    def test_decode_cuda_graph_capture(self):
        t = 64
        fb = self._fb(True)
        static_x = self.x[t : t + 1].clone()
        static_ql = self.q_lora[t : t + 1].clone()
        static_q = self.q[t : t + 1].clone()
        static_pos = self.positions[t : t + 1].clone()

        def warmup_to_tm1(backend, layers):
            self._run_extend(backend, layers, slice(0, t - 1))
            static_x.copy_(self.x[t - 1 : t])
            static_ql.copy_(self.q_lora[t - 1 : t])
            static_q.copy_(self.q[t - 1 : t])
            static_pos.copy_(self.positions[t - 1 : t])
            for lid in LAYER_ORDER:
                sm70_forward_low_ratio_sources(
                    backend, layers[lid], static_x, static_ql, static_pos, fb
                )
                sm70_forward_sparse(
                    backend, static_q, layers[lid], fb, layers[lid].compress_ratio, self.sink
                )

        layers_ref = self._layers()
        backend_ref = self._backend()
        warmup_to_tm1(backend_ref, layers_ref)
        ref = self._run_decode(backend_ref, layers_ref, t)

        layers = self._layers()
        backend = self._backend()
        warmup_to_tm1(backend, layers)
        static_x.copy_(self.x[t : t + 1])
        static_ql.copy_(self.q_lora[t : t + 1])
        static_q.copy_(self.q[t : t + 1])
        static_pos.copy_(self.positions[t : t + 1])
        st = get_state(backend)
        snap = {
            "swa": {k: v.clone() for k, v in st.swa_ring.items()},
            "kv": {k: v.clone() for k, v in st.kv_rows.items()},
            "ix": {k: v.clone() for k, v in st.index_rows.items()},
            "pkv": {k: v.clone() for k, v in st.pending_kv.items()},
            "psc": {k: v.clone() for k, v in st.pending_score.items()},
            "topk": {k: v.clone() for k, v in st.topk.items()},
            "cand": None if st.cand_ids is None else st.cand_ids.clone(),
            "ck": None if st.cand_key_ids is None else st.cand_key_ids.clone(),
        }

        def restore():
            for k, v in snap["swa"].items():
                st.swa_ring[k].copy_(v)
            for k, v in snap["kv"].items():
                st.kv_rows[k].copy_(v)
            for k, v in snap["ix"].items():
                st.index_rows[k].copy_(v)
            for k, v in snap["pkv"].items():
                st.pending_kv[k].copy_(v)
            for k, v in snap["psc"].items():
                st.pending_score[k].copy_(v)
            for k, v in snap["topk"].items():
                st.topk[k].copy_(v)
            if snap["cand"] is not None:
                st.cand_ids.copy_(snap["cand"])
            if snap["ck"] is not None:
                st.cand_key_ids.copy_(snap["ck"])

        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        captured = {}
        with torch.cuda.graph(g):
            for lid in LAYER_ORDER:
                sm70_forward_low_ratio_sources(
                    backend, layers[lid], static_x, static_ql, static_pos, fb
                )
                captured[lid] = sm70_forward_sparse(
                    backend,
                    static_q,
                    layers[lid],
                    fb,
                    layers[lid].compress_ratio,
                    self.sink,
                )
        restore()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        for lid in LAYER_ORDER:
            d = self._maxdiff(captured[lid], ref[lid])
            print(f"graph replay L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(
                    captured[lid].float(), ref[lid].float(), atol=1e-3, rtol=1e-3
                ),
                f"graph L{lid} maxabs={d}",
            )

    def _run_verify(self, backend, layers, sl: slice):
        fb = self._fb(False, target_verify=True)
        outs = {}
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend,
                layers[lid],
                self.x[sl],
                self.q_lora[sl],
                self.positions[sl],
                fb,
            )
            outs[lid] = sm70_forward_sparse(
                backend, self.q[sl], layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        return outs

    def test_target_verify_t6_matches_prefill_window(self):
        t0, t = 64, 6
        backend = self._backend()
        backend.model_runner = SimpleNamespace(decode_num_tokens_per_req=lambda: 6)
        layers = self._layers()
        pref = self._run_extend(backend, layers, slice(0, t0 + t))
        be_v = self._backend()
        be_v.model_runner = SimpleNamespace(decode_num_tokens_per_req=lambda: 6)
        layers_v = self._layers()
        self._run_extend(be_v, layers_v, slice(0, t0))
        ver = self._run_verify(be_v, layers_v, slice(t0, t0 + t))
        for lid in LAYER_ORDER:
            a = pref[lid][-t:]
            b = ver[lid]
            d = self._maxdiff(a, b)
            print(f"verify T=6 L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(a.float(), b.float(), atol=ATOL, rtol=RTOL),
                f"verify T=6 L{lid} maxabs={d}",
            )

    def test_target_verify_cuda_graph_capture(self):
        t0, t = 64, 6
        fb = self._fb(False, target_verify=True)
        backend = self._backend()
        backend.model_runner = SimpleNamespace(decode_num_tokens_per_req=lambda: 6)
        layers = self._layers()
        self._run_extend(backend, layers, slice(0, t0))
        static_x = self.x[t0 : t0 + t].clone()
        static_ql = self.q_lora[t0 : t0 + t].clone()
        static_q = self.q[t0 : t0 + t].clone()
        static_pos = self.positions[t0 : t0 + t].clone()
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend, layers[lid], static_x, static_ql, static_pos, fb
            )
            sm70_forward_sparse(
                backend, static_q, layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        ref = self._run_verify(backend, layers, slice(t0, t0 + t))
        st = get_state(backend)
        snap = {
            "swa": {k: v.clone() for k, v in st.swa_ring.items()},
            "kv": {k: v.clone() for k, v in st.kv_rows.items()},
            "ix": {k: v.clone() for k, v in st.index_rows.items()},
            "pkv": {k: v.clone() for k, v in st.pending_kv.items()},
            "psc": {k: v.clone() for k, v in st.pending_score.items()},
            "topk": {k: v.clone() for k, v in st.topk.items()},
            "vtopk": {k: v.clone() for k, v in st.verify_topk.items()},
            "cand": None if st.cand_ids is None else st.cand_ids.clone(),
            "vcand": None if st.verify_cand is None else st.verify_cand.clone(),
            "ck": None if st.cand_key_ids is None else st.cand_key_ids.clone(),
        }

        def restore():
            for k, v in snap["swa"].items():
                st.swa_ring[k].copy_(v)
            for k, v in snap["kv"].items():
                st.kv_rows[k].copy_(v)
            for k, v in snap["ix"].items():
                st.index_rows[k].copy_(v)
            for k, v in snap["pkv"].items():
                st.pending_kv[k].copy_(v)
            for k, v in snap["psc"].items():
                st.pending_score[k].copy_(v)
            for k, v in snap["topk"].items():
                st.topk[k].copy_(v)
            for k, v in snap["vtopk"].items():
                st.verify_topk[k].copy_(v)
            if snap["cand"] is not None:
                st.cand_ids.copy_(snap["cand"])
            if snap["vcand"] is not None:
                st.verify_cand.copy_(snap["vcand"])
            if snap["ck"] is not None:
                st.cand_key_ids.copy_(snap["ck"])

        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        captured = {}
        with torch.cuda.graph(g):
            for lid in LAYER_ORDER:
                sm70_forward_low_ratio_sources(
                    backend, layers[lid], static_x, static_ql, static_pos, fb
                )
                captured[lid] = sm70_forward_sparse(
                    backend,
                    static_q,
                    layers[lid],
                    fb,
                    layers[lid].compress_ratio,
                    self.sink,
                )
        restore()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        for lid in LAYER_ORDER:
            d = self._maxdiff(captured[lid], ref[lid])
            print(f"verify graph replay L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(
                    captured[lid].float(), ref[lid].float(), atol=1e-3, rtol=1e-3
                ),
                f"verify graph L{lid} maxabs={d}",
            )

    def _snap_persistent(self, st):
        return {
            "swa": {k: v.clone() for k, v in st.swa_ring.items()},
            "kv": {k: v.clone() for k, v in st.kv_rows.items()},
            "ix": {k: v.clone() for k, v in st.index_rows.items()},
            "pkv": {k: v.clone() for k, v in st.pending_kv.items()},
            "psc": {k: v.clone() for k, v in st.pending_score.items()},
        }

    def _restore_persistent(self, st, snap):
        for k, v in snap["swa"].items():
            st.swa_ring[k].copy_(v)
        for k, v in snap["kv"].items():
            st.kv_rows[k].copy_(v)
        for k, v in snap["ix"].items():
            st.index_rows[k].copy_(v)
        for k, v in snap["pkv"].items():
            st.pending_kv[k].copy_(v)
        for k, v in snap["psc"].items():
            st.pending_score[k].copy_(v)

    def _assert_persistent(self, got, exp, tag: str):
        for name, exact in (("swa", True), ("kv", True), ("ix", True)):
            for k, ev in exp[name].items():
                gv = got[name][k]
                n = int((gv != ev).sum().item())
                self.assertEqual(n, 0, f"{tag} {name} L{k} byte diffs={n}")
        for name in ("pkv", "psc"):
            for k, ev in exp[name].items():
                d = self._maxdiff(got[name][k], ev)
                self.assertLess(d, 1e-4, f"{tag} {name} L{k} maxabs={d}")

    def _backend_verify(self):
        be = self._backend()
        be.model_runner = SimpleNamespace(decode_num_tokens_per_req=lambda: 6)
        return be

    def _run_verify_batch(self, backend, layers, x, q_lora, q, positions):
        fb = self._fb(False, target_verify=True)
        outs = {}
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend, layers[lid], x, q_lora, positions, fb
            )
            outs[lid] = sm70_forward_sparse(
                backend, q, layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        return outs

    def _mixed_block(self, t0: int, commit_n: int, block: int = 6):
        """Real tokens in the accepted prefix, unrelated hidden states after."""
        sl = slice(t0, t0 + block)
        x = self.x[sl].clone()
        ql = self.q_lora[sl].clone()
        q = self.q[sl].clone()
        pos = self.positions[sl].clone()
        n_junk = block - commit_n
        g = torch.Generator(device="cpu").manual_seed(1000 + t0 * 10 + commit_n)
        x[commit_n:] = (
            torch.randn(n_junk, self.hidden, generator=g).to(FP16) * 0.2
        ).to(self.dev)
        ql[commit_n:] = (
            torch.randn(n_junk, self.q_lora_r, generator=g).to(FP16) * 0.2
        ).to(self.dev)
        q[commit_n:] = (
            torch.randn(n_junk, self.heads, 512, generator=g).to(FP16) * 0.2
        ).to(self.dev)
        return x, ql, q, pos

    def _reference_accept(self, t0: int, commit_n: int):
        """Token-by-token decode of the committed inputs, then one more token."""
        layers = self._layers()
        be = self._backend_verify()
        self._run_extend(be, layers, slice(0, t0))
        outs = []
        for i in range(commit_n):
            outs.append(self._run_decode(be, layers, t0 + i))
        snap = self._snap_persistent(get_state(be))
        nxt = self._run_decode(be, layers, t0 + commit_n)
        return outs, snap, nxt

    def _check_accept_outputs(self, ver, ref_outs, nxt, ref_nxt, tag: str):
        for i, ref in enumerate(ref_outs):
            for lid in LAYER_ORDER:
                d = self._maxdiff(ver[lid][i : i + 1], ref[lid])
                print(f"{tag} verify[{i}] L{lid} maxabs={d:.5f}")
                self.assertTrue(
                    torch.allclose(ver[lid][i : i + 1].float(), ref[lid].float(), atol=ATOL, rtol=RTOL),
                    f"{tag} verify[{i}] L{lid} maxabs={d}",
                )
        for lid in LAYER_ORDER:
            d = self._maxdiff(nxt[lid], ref_nxt[lid])
            print(f"{tag} next L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(nxt[lid].float(), ref_nxt[lid].float(), atol=ATOL, rtol=RTOL),
                f"{tag} next L{lid} maxabs={d}",
            )

    def test_verify_rejection_matches_decode(self):
        """Rejected drafts used to stay in the live ring and the ratio-2 slot.

        Position 200's window still contains the slots that 201..205 alias
        (distance 128). The verify block puts unrelated hidden states in that
        tail. After commit of only the accepted prefix, the next real token
        must match token-by-token decode that never saw the tail.
        """
        block = 6
        for t0, commit_n in ((200, 1), (200, 2), (200, 5), (205, 1)):
            tag = f"t0={t0} commit={commit_n}"
            ref_outs, ref_snap, ref_nxt = self._reference_accept(t0, commit_n)
            layers = self._layers()
            be = self._backend_verify()
            self._run_extend(be, layers, slice(0, t0))
            st = get_state(be)
            before = self._snap_persistent(st)
            x, ql, q, pos = self._mixed_block(t0, commit_n, block)
            ver = self._run_verify_batch(be, layers, x, ql, q, pos)
            during = self._snap_persistent(st)
            self._assert_persistent(
                {"swa": during["swa"], "kv": {}, "ix": {},
                 "pkv": during["pkv"], "psc": during["psc"]},
                {"swa": before["swa"], "kv": {}, "ix": {},
                 "pkv": before["pkv"], "psc": before["psc"]},
                f"{tag} live-ring-and-pending",
            )
            # The forward did publish speculative compressed rows. Commit peels them.
            self.assertFalse(
                torch.equal(during["kv"][2], before["kv"][2]),
                f"{tag} ratio-2 rows were not written during verify",
            )
            sm70_commit_target_verify(
                be,
                torch.tensor([commit_n], dtype=torch.int32, device=self.dev),
                num_positions=block,
            )
            self._assert_persistent(self._snap_persistent(st), ref_snap, f"{tag} after-commit")
            nxt = self._run_decode(be, layers, t0 + commit_n)
            self._check_accept_outputs(ver, ref_outs, nxt, ref_nxt, tag)

    def test_verify_rejection_commit_after_cuda_graph(self):
        """Sampling accept runs after the verify graph. Commit must see the
        replayed scratch, not the capture-time tokens.
        """
        t0, block, commit_n = 200, 6, 1
        tag = "graph commit=1"
        ref_outs, ref_snap, ref_nxt = self._reference_accept(t0, commit_n)
        layers = self._layers()
        be = self._backend_verify()
        fb = self._fb(False, target_verify=True)
        self._run_extend(be, layers, slice(0, t0))
        st = get_state(be)
        base = self._snap_persistent(st)
        x, ql, q, pos = self._mixed_block(t0, commit_n, block)
        static_x, static_ql, static_q, static_pos = x.clone(), ql.clone(), q.clone(), pos.clone()
        # Allocate verify scratch, then put the live cache back.
        self._run_verify_batch(be, layers, static_x, static_ql, static_q, static_pos)
        self._restore_persistent(st, base)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        captured = {}
        with torch.cuda.graph(graph):
            for lid in LAYER_ORDER:
                sm70_forward_low_ratio_sources(
                    be, layers[lid], static_x, static_ql, static_pos, fb
                )
                captured[lid] = sm70_forward_sparse(
                    be, static_q, layers[lid], fb, layers[lid].compress_ratio, self.sink
                )
        self._restore_persistent(st, base)
        # Replay must not keep the capture-time tail. Fill a second junk pattern.
        x2, ql2, q2, pos2 = self._mixed_block(t0, commit_n, block)
        g = torch.Generator(device="cpu").manual_seed(4242)
        x2[commit_n:] = (torch.randn(block - commit_n, self.hidden, generator=g).to(FP16) * 0.3).to(self.dev)
        q2[commit_n:] = (
            torch.randn(block - commit_n, self.heads, 512, generator=g).to(FP16) * 0.3
        ).to(self.dev)
        static_x.copy_(x2)
        static_ql.copy_(ql2)
        static_q.copy_(q2)
        static_pos.copy_(pos2)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        during = self._snap_persistent(st)
        self._assert_persistent(
            {"swa": during["swa"], "kv": {}, "ix": {},
             "pkv": during["pkv"], "psc": during["psc"]},
            {"swa": base["swa"], "kv": {}, "ix": {},
             "pkv": base["pkv"], "psc": base["psc"]},
            f"{tag} live-ring-and-pending",
        )
        sm70_commit_target_verify(
            be,
            torch.tensor([commit_n], dtype=torch.int32, device=self.dev),
            num_positions=block,
        )
        self._assert_persistent(self._snap_persistent(st), ref_snap, f"{tag} after-commit")
        nxt = self._run_decode(be, layers, t0 + commit_n)
        self._check_accept_outputs(captured, ref_outs, nxt, ref_nxt, tag)


if __name__ == "__main__":
    unittest.main()
