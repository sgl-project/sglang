"""C1/C2 compression before RoPE, indexer quantization, or KV-cache writes.

The independent reference follows the GPU/private compressor arithmetic and
request-local pairing, not the NPU implementation's vectorized grouping. CPU
coverage needs only torch; the same cases also run on a real available NPU.
"""

import ast
import importlib.util
import sys
import unittest
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[5]
_PYTHON_ROOT = _REPO_ROOT / "python"


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, _PYTHON_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through sys.modules during execution.
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


_ci = _load_module("_dsv41_compress_ci", "sglang/test/ci/ci_register.py")
register_npu_ci = _ci.register_npu_ci
register_npu_ci(est_time=10, suite="base-a-test-1-npu-a2")

_SHARED_NAME = "sglang.srt.layers.attention.dsv4.dsv41_compressor"
_shared = _load_module(
    _SHARED_NAME, "sglang/srt/layers/attention/dsv4/dsv41_compressor.py"
)
with patch.dict(sys.modules, {_SHARED_NAME: _shared}):
    _npu = _load_module(
        "_dsv41_npu_compressor",
        "sglang/srt/hardware_backend/npu/dsv4/dsv41_compressor.py",
    )
DeepseekV41Compressor = _shared.DeepseekV41Compressor
compress_low_ratio = _npu.compress_low_ratio
compress_low_ratio_batch = _npu.compress_low_ratio_batch


def _class_node(relative_path, class_name):
    path = _PYTHON_ROOT / relative_path
    module = ast.parse(path.read_text(), filename=str(path))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ), path


def _load_forward_mode():
    node, path = _class_node(
        "sglang/srt/model_executor/forward_batch_info.py", "ForwardMode"
    )
    namespace = {"IntEnum": IntEnum, "auto": auto}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["ForwardMode"]


ForwardMode = _load_forward_mode()


def _backend_dispatch(pool, *, cp_size=1, dcp_size=1):
    """Test the real two dispatch methods without importing the heavy backend.

    This intentionally does not validate the complete backend's imports, MRO,
    runtime initialization, fused C4/C128 operators, or attention execution.
    """
    node, path = _class_node(
        "sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py",
        "CompressorAscendBackendMixin",
    )
    names = {"forward_core_compressor", "forward_low_ratio_compressor"}
    methods = [
        method
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name in names
    ]
    assert len(methods) == len(names)
    module = ast.parse(
        "from __future__ import annotations\nclass Dispatch:\n    pass\n"
    )
    module.body[1].body = methods
    namespace = {
        "torch": torch,
        "LowRatioCompressResult": _npu.LowRatioCompressResult,
        "compress_low_ratio_batch": compress_low_ratio_batch,
        "get_parallel": lambda: SimpleNamespace(
            attn_cp_size=cp_size, attn_dcp_size=dcp_size
        ),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    backend = namespace["Dispatch"]()
    backend.token_to_kv_pool = pool
    return backend


def _has_npu():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


_NPU_AVAILABLE = _has_npu()


def _reference_finish(pooled, weight, eps):
    """The BF16 round before FP32 RMSNorm is part of the model contract."""
    rounded = pooled.to(torch.bfloat16).float()
    normalized = rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(torch.bfloat16)


def _reference_extend(compressor, x, positions, req_indices, raw_out_loc, state):
    """CPU loop oracle: every even token parks state, every odd token emits."""
    ratio = compressor.compress_ratio
    x = x.cpu()
    weight = compressor.wkv.weight.detach().cpu()
    kv = F.linear(x if ratio == 1 else x.float(), weight)
    score = (
        F.linear(x.float(), compressor.wgate.weight.detach().cpu())
        if ratio == 2
        else None
    )
    values, group_pos, slots = [], [], []
    for i, (pos, req, raw) in enumerate(
        zip(positions.tolist(), req_indices.tolist(), raw_out_loc.tolist())
    ):
        if raw <= 0:
            continue
        if ratio == 1:
            pooled = kv[i]
        elif pos % 2 == 0:
            state[0][req] = kv[i]
            state[1][req] = score[i]
            continue
        else:
            pair_kv = torch.stack((state[0][req], kv[i]))
            pair_score = torch.stack((state[1][req], score[i]))
            pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0)
        values.append(pooled)
        group_pos.append(pos if ratio == 1 else pos - 1)
        slots.append(raw // ratio)
    if values:
        pooled = torch.stack(values)
    else:
        pooled = torch.empty((0, weight.shape[0]), dtype=weight.dtype)
    latent = _reference_finish(
        pooled, compressor.norm.weight.detach().cpu(), compressor.norm.eps
    )
    return latent, group_pos, slots


class _CompressorCases:
    device = "cpu"
    hidden_size = 7
    head_dim = 8
    req_slots = 8

    def _compressor(self, ratio):
        torch.manual_seed(918)
        module = DeepseekV41Compressor(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            compress_ratio=ratio,
            eps=1e-6,
        )
        with torch.no_grad():
            module.norm.weight.copy_(torch.linspace(0.5, 1.5, self.head_dim))
        return module.to(self.device)

    def _inputs(self, count, *, offset=0):
        values = torch.arange(
            offset, offset + count * self.hidden_size, dtype=torch.float32
        ).reshape(count, self.hidden_size)
        return (values.sin() * 2.5).to(device=self.device, dtype=torch.bfloat16)

    def _long(self, values):
        return torch.tensor(values, dtype=torch.int64, device=self.device)

    def _state(self):
        shape = (self.req_slots + 1, self.head_dim)
        return (
            torch.zeros(shape, dtype=torch.float32, device=self.device),
            torch.zeros(shape, dtype=torch.float32, device=self.device),
        )

    def _run(self, compressor, x, pos, req, raw, *, decode=False, state=None):
        kwargs = {}
        if state is not None:
            kwargs.update(
                state_kv=state[0], state_score=state[1], pad_row=self.req_slots
            )
        with torch.no_grad():
            return compress_low_ratio(
                compressor,
                x,
                positions=self._long(pos),
                req_indices=self._long(req),
                raw_out_loc=self._long(raw),
                is_decode=decode,
                **kwargs,
            )

    def _assert_latent(self, actual, expected):
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.device.type, self.device)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        # BF16 NPU GEMM/norm may differ by a rounding unit from CPU torch.
        rtol, atol = (0.0, 0.0) if self.device == "cpu" else (0.02, 0.016)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol)

    def _assert_extend_reference(self, result, expected):
        latent, pos, loc = expected
        self._assert_latent(result.latent, latent)
        self.assertEqual(result.positions.cpu().tolist(), pos)
        self.assertEqual(result.out_loc.cpu().tolist(), loc)
        self.assertEqual(result.valid_mask.dtype, torch.bool)
        self.assertEqual(result.valid_mask.cpu().tolist(), [True] * len(loc))

    def test_projection_dtype_and_bf16_before_norm_contract(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                compressor = self._compressor(ratio)
                expected_dtype = torch.bfloat16 if ratio == 1 else torch.float32
                self.assertEqual(compressor.wkv.weight.dtype, expected_dtype)
                kv, score = compressor.project(self._inputs(3))
                self.assertEqual(kv.dtype, expected_dtype)
                if ratio == 1:
                    self.assertIsNone(score)
                    self.assertFalse(hasattr(compressor, "wgate"))
                else:
                    self.assertEqual(score.dtype, torch.float32)
                    self.assertEqual(compressor.wgate.weight.dtype, torch.float32)
                pooled = torch.linspace(-3.127, 2.313, 24).reshape(3, 8)
                expected = _reference_finish(
                    pooled, compressor.norm.weight.detach().cpu(), compressor.norm.eps
                )
                self._assert_latent(compressor.finish(pooled.to(self.device)), expected)

    def test_c1_needs_no_state_and_returns_pre_rope_latents(self):
        compressor = self._compressor(1)
        x = self._inputs(4)
        # Identical projected input at different positions must remain identical:
        # this API must not apply positional rotation or FP4 quantization.
        x[1] = x[0]
        pos, req, raw = [3, 77, 9, 15], [0, 2, 4, 6], [131, 845, 393, 911]
        expected = _reference_extend(
            compressor, x, torch.tensor(pos), torch.tensor(req), torch.tensor(raw), None
        )
        for decode in (False, True):
            with self.subTest(decode=decode):
                result = self._run(compressor, x, pos, req, raw, decode=decode)
                self._assert_extend_reference(result, expected)
                self.assertTrue(torch.equal(result.latent[0], result.latent[1]))

    def test_c1_invalid_slots_are_compacted_only_for_extend(self):
        compressor = self._compressor(1)
        x = self._inputs(4)
        pos, req, raw = [2, 0, 8, 0], [2, 0, 4, 0], [258, 0, 520, -1]
        expected = _reference_extend(
            compressor, x, torch.tensor(pos), torch.tensor(req), torch.tensor(raw), None
        )
        result = self._run(compressor, x, pos, req, raw)
        self._assert_extend_reference(result, expected)
        result = self._run(compressor, x, pos, req, raw, decode=True)
        self.assertEqual(result.latent.shape, (4, self.head_dim))
        self.assertEqual(result.out_loc.cpu().tolist(), [258, -1, 520, -1])
        self.assertEqual(result.valid_mask.cpu().tolist(), [True, False, True, False])
        self._assert_latent(result.latent[result.valid_mask], expected[0])

    def test_c2_extreme_gate_is_softmax_over_tokens_per_channel(self):
        compressor = self._compressor(2)
        with torch.no_grad():
            compressor.wkv.weight.zero_()
            compressor.wgate.weight.zero_()
            compressor.wkv.weight[:, 0] = torch.arange(
                1, self.head_dim + 1, dtype=torch.float32, device=self.device
            )
            compressor.wgate.weight[:, 0] = torch.tensor(
                [10000, -10000] * 4, dtype=torch.float32, device=self.device
            )
        x = torch.zeros((2, self.hidden_size), dtype=torch.bfloat16, device=self.device)
        x[:, 0] = self._long([1, -1]).to(torch.bfloat16)
        state = self._state()
        result = self._run(compressor, x, [0, 1], [2, 2], [256, 257], state=state)
        pooled = torch.tensor([[1, -2, 3, -4, 5, -6, 7, -8]], dtype=torch.float32)
        expected = _reference_finish(
            pooled, compressor.norm.weight.detach().cpu(), compressor.norm.eps
        )
        self._assert_extend_reference(result, (expected, [0], [128]))

    def test_c2_odd_chunk_boundary_and_request_reordering(self):
        compressor = self._compressor(2)
        state = self._state()
        reference_state = tuple(value.cpu().clone() for value in state)
        # Request 4 ends its first chunk on an even token; its next chunk begins
        # with the odd partner. Request 2 changes from a short to a long chunk.
        chunks = [
            [(4, range(0, 5)), (0, range(0, 2)), (2, range(0, 1))],
            [(2, range(1, 7)), (4, range(5, 6)), (0, range(2, 6))],
            [(4, range(6, 8)), (0, range(6, 7)), (2, range(7, 8))],
        ]
        for step, chunk in enumerate(chunks):
            with self.subTest(step=step):
                rows = [(req, pos) for req, positions in chunk for pos in positions]
                req, pos = zip(*rows)
                raw = [(r + 1) * 128 + p for r, p in rows]
                x = torch.cat([self._inputs(1, offset=r * 80 + p * 7) for r, p in rows])
                expected = _reference_extend(
                    compressor,
                    x,
                    torch.tensor(pos),
                    torch.tensor(req),
                    torch.tensor(raw),
                    reference_state,
                )
                result = self._run(compressor, x, pos, req, raw, state=state)
                self._assert_extend_reference(result, expected)
                for actual, reference in zip(state, reference_state):
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(
                        actual[: self.req_slots].cpu(),
                        reference[: self.req_slots],
                        rtol=1e-4,
                        atol=1e-4,
                    )

    def test_c2_decode_parks_even_then_reads_odd_after_batch_reordering(self):
        compressor = self._compressor(2)
        state = self._state()
        x = self._inputs(3)
        parked = self._run(
            compressor,
            x,
            [0, 4, 8],
            [0, 2, 4],
            [128, 388, 648],
            decode=True,
            state=state,
        )
        self.assertEqual(parked.latent.shape, (3, self.head_dim))
        self.assertEqual(parked.out_loc.cpu().tolist(), [-1, -1, -1])
        self.assertEqual(parked.valid_mask.cpu().tolist(), [False] * 3)
        state_before = tuple(value.clone() for value in state)
        reference_state = tuple(value.cpu().clone() for value in state)
        pos, req, raw = [9, 1, 5], [4, 0, 2], [649, 129, 389]
        x = self._inputs(3, offset=70)
        expected = _reference_extend(
            compressor,
            x,
            torch.tensor(pos),
            torch.tensor(req),
            torch.tensor(raw),
            reference_state,
        )
        result = self._run(compressor, x, pos, req, raw, decode=True, state=state)
        self._assert_extend_reference(result, expected)
        for actual, before in zip(state, state_before):
            self.assertTrue(torch.equal(actual, before))

    def test_c2_decode_padding_cannot_clobber_live_request_zero(self):
        compressor = self._compressor(2)
        state = self._state()
        self._run(
            compressor,
            self._inputs(2),
            [6, 2],
            [0, 2],
            [134, 386],
            decode=True,
            state=state,
        )
        before = tuple(value.clone() for value in state)
        reference_state = tuple(value.cpu().clone() for value in state)
        pos, req, raw = [7, 0, 0, 3, 1], [0, 0, 0, 2, 0], [135, 0, -1, 387, 0]
        x = self._inputs(5, offset=133)
        expected = _reference_extend(
            compressor,
            x,
            torch.tensor(pos),
            torch.tensor(req),
            torch.tensor(raw),
            reference_state,
        )
        result = self._run(compressor, x, pos, req, raw, decode=True, state=state)
        self.assertEqual(result.latent.shape, (5, self.head_dim))
        self.assertEqual(
            result.valid_mask.cpu().tolist(), [True, False, False, True, False]
        )
        self.assertEqual(result.out_loc.cpu().tolist(), [67, -1, -1, 193, -1])
        self.assertEqual(result.positions[result.valid_mask].cpu().tolist(), [6, 2])
        self._assert_latent(result.latent[result.valid_mask], expected[0])
        for actual, original in zip(state, before):
            self.assertTrue(
                torch.equal(actual[: self.req_slots], original[: self.req_slots])
            )

    def test_c2_no_complete_extend_group_still_saves_pending_state(self):
        compressor = self._compressor(2)
        state = self._state()
        x = self._inputs(2)
        result = self._run(compressor, x, [0, 6], [0, 2], [128, 390], state=state)
        self.assertEqual(result.latent.shape, (0, self.head_dim))
        self.assertEqual(result.positions.numel(), 0)
        self.assertEqual(result.out_loc.numel(), 0)
        self.assertEqual(result.valid_mask.numel(), 0)
        kv, score = compressor.project(x)
        for actual, expected in ((state[0][[0, 2]], kv), (state[1][[0, 2]], score)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_batch_is_noop(self):
        for ratio in (1, 2):
            for decode in (False, True):
                with self.subTest(ratio=ratio, decode=decode):
                    compressor = self._compressor(ratio)
                    state = self._state() if ratio == 2 else None
                    if state is not None:
                        state[0].fill_(1.25)
                        state[1].fill_(-3.5)
                    result = self._run(
                        compressor,
                        self._inputs(0),
                        [],
                        [],
                        [],
                        decode=decode,
                        state=state,
                    )
                    self.assertEqual(result.latent.shape, (0, self.head_dim))
                    self.assertEqual(result.latent.dtype, torch.bfloat16)
                    self.assertEqual(result.positions.numel(), 0)
                    self.assertEqual(result.out_loc.numel(), 0)
                    self.assertEqual(result.valid_mask.numel(), 0)
                    if state is not None:
                        self.assertTrue(bool((state[0] == 1.25).all()))
                        self.assertTrue(bool((state[1] == -3.5).all()))

    def test_invalid_ratio_fails_without_mutating_pair_state(self):
        compressor = self._compressor(2)
        compressor.compress_ratio = 4
        state = self._state()
        state[0].fill_(1.25)
        state[1].fill_(-3.5)
        with self.assertRaises((ValueError, AssertionError)):
            self._run(compressor, self._inputs(1), [0], [0], [128], state=state)
        self.assertTrue(bool((state[0] == 1.25).all()))
        self.assertTrue(bool((state[1] == -3.5).all()))

    def test_c2_rejects_missing_or_non_fp32_state_before_mutation(self):
        compressor = self._compressor(2)
        x = self._inputs(1)
        with self.assertRaises(ValueError):
            self._run(compressor, x, [0], [0], [128])
        state = tuple(value.to(torch.bfloat16) for value in self._state())
        with self.assertRaises(ValueError):
            self._run(compressor, x, [0], [0], [128], state=state)
        for value in state:
            self.assertTrue(bool((value == 0).all()))
        state = self._state()
        with self.assertRaises(ValueError):
            compress_low_ratio(
                compressor,
                x,
                positions=self._long([0]),
                req_indices=self._long([0]),
                raw_out_loc=self._long([128]),
                is_decode=True,
                state_kv=state[0],
                state_score=state[1],
                pad_row=0,
            )
        for value in state:
            self.assertTrue(bool((value == 0).all()))

    def test_float_metadata_is_rejected_without_mutating_state(self):
        compressor = self._compressor(2)
        state = self._state()
        for name in ("positions", "req_indices", "raw_out_loc"):
            metadata = {
                "positions": self._long([0]),
                "req_indices": self._long([0]),
                "raw_out_loc": self._long([128]),
            }
            metadata[name] = metadata[name].float() + 0.5
            with self.subTest(name=name), self.assertRaises(ValueError):
                compress_low_ratio(
                    compressor,
                    self._inputs(1),
                    **metadata,
                    is_decode=True,
                    state_kv=state[0],
                    state_score=state[1],
                    pad_row=self.req_slots,
                )
            for value in state:
                self.assertTrue(bool((value == 0).all()))

    @staticmethod
    def _mode(name):
        return ForwardMode[name.upper()]

    def test_batch_c1_uses_full_bundle_not_swa_and_needs_no_pool(self):
        compressor = self._compressor(1)
        x = self._inputs(4)
        pos, req, raw = [8, 0, 1, 2], [5, 1, 1, 1], [2056, 512, 513, 514]
        batch = SimpleNamespace(
            forward_mode=self._mode("extend"),
            req_pool_indices=self._long([5, 1]),
            extend_seq_lens=self._long([1, 3]),
            out_cache_loc=self._long([7, 8, 9, 10]),
            out_cache_loc_dsv4=SimpleNamespace(
                out_full_loc=self._long(raw), out_swa_loc=self._long([7, 8, 9, 10])
            ),
        )
        expected = _reference_extend(
            compressor, x, torch.tensor(pos), torch.tensor(req), torch.tensor(raw), None
        )
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            with self.subTest(mode=mode.name), torch.no_grad():
                batch.forward_mode = mode
                result = compress_low_ratio_batch(
                    compressor=compressor,
                    x=x,
                    positions=self._long(pos),
                    forward_batch=batch,
                    layer_id=7,
                    token_to_kv_pool=None,
                )
                self._assert_extend_reference(result, expected)

    def test_batch_c2_repeats_requests_and_isolates_source_layer_state(self):
        compressor = self._compressor(2)
        state, other_source_state = self._state(), self._state()
        other_source_state[0].fill_(19.0)
        other_source_state[1].fill_(-21.0)
        reference_state = tuple(value.cpu().clone() for value in state)
        pool = SimpleNamespace(
            c2_pair_kv_state={7: state[0], 11: other_source_state[0]},
            c2_pair_score_state={7: state[1], 11: other_source_state[1]},
            c2_pair_pad_row=self.req_slots,
        )
        # Extend leaves request 2 at token 0 and request 0 at token 2. The
        # subsequent decode reorders the batch and completes both pending pairs.
        for step, (mode, pos, req, raw) in enumerate(
            (
                ("extend", [0, 0, 1, 2], [2, 0, 0, 0], [256, 512, 513, 514]),
                ("decode", [3, 1], [0, 2], [515, 257]),
            )
        ):
            with self.subTest(mode=mode):
                x = self._inputs(len(pos), offset=step * 63)
                batch = SimpleNamespace(
                    forward_mode=self._mode(mode),
                    req_pool_indices=self._long([2, 0] if mode == "extend" else req),
                    extend_seq_lens=self._long([1, 3]) if mode == "extend" else None,
                    out_cache_loc=self._long(raw),
                    out_cache_loc_dsv4=None,
                )
                expected = _reference_extend(
                    compressor,
                    x,
                    torch.tensor(pos),
                    torch.tensor(req),
                    torch.tensor(raw),
                    reference_state,
                )
                with torch.no_grad():
                    result = compress_low_ratio_batch(
                        compressor=compressor,
                        x=x,
                        positions=self._long(pos),
                        forward_batch=batch,
                        layer_id=7,
                        token_to_kv_pool=pool,
                    )
                self._assert_extend_reference(result, expected)
                for actual, reference in zip(state, reference_state):
                    torch.testing.assert_close(
                        actual.cpu(), reference, rtol=1e-4, atol=1e-4
                    )
                self.assertTrue(bool((other_source_state[0] == 19.0).all()))
                self.assertTrue(bool((other_source_state[1] == -21.0).all()))

    def test_batch_idle_and_unsupported_modes_do_not_access_state(self):
        compressor = self._compressor(2)
        # No req metadata or state pool is supplied: mode handling must happen
        # before either can be touched, including when there are zero tokens.
        for mode in ForwardMode:
            if mode.name in ("EXTEND", "MIXED", "DECODE"):
                continue
            with self.subTest(mode=mode.name):
                kwargs = dict(
                    compressor=compressor,
                    x=self._inputs(0),
                    positions=self._long([]),
                    forward_batch=SimpleNamespace(forward_mode=mode),
                    layer_id=7,
                    token_to_kv_pool=None,
                )
                if mode == ForwardMode.IDLE:
                    self.assertIsNone(compress_low_ratio_batch(**kwargs))
                else:
                    with self.assertRaises(NotImplementedError):
                        compress_low_ratio_batch(**kwargs)

    def test_backend_core_dispatch_c1_c2_returns_pre_rope_result(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                compressor = self._compressor(ratio)
                state = self._state()
                reference_state = tuple(value.cpu().clone() for value in state)
                pool = SimpleNamespace(
                    c2_pair_kv_state={7: state[0]},
                    c2_pair_score_state={7: state[1]},
                    c2_pair_pad_row=self.req_slots,
                )
                backend = _backend_dispatch(pool)
                x = self._inputs(2)
                pos, req, raw = [0, 1], [2, 2], [256, 257]
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    positions=self._long(pos),
                    req_pool_indices=self._long([2]),
                    extend_seq_lens=self._long([2]),
                    out_cache_loc=self._long(raw),
                    out_cache_loc_dsv4=None,
                )
                expected = _reference_extend(
                    compressor,
                    x,
                    torch.tensor(pos),
                    torch.tensor(req),
                    torch.tensor(raw),
                    reference_state,
                )
                with torch.no_grad():
                    result = backend.forward_core_compressor(x, batch, 7, compressor)
                self.assertIsInstance(result, _npu.LowRatioCompressResult)
                self._assert_extend_reference(result, expected)

    def test_backend_core_dispatch_c4_c128_preserves_legacy_call(self):
        for ratio in (4, 128):
            with self.subTest(ratio=ratio):
                backend = _backend_dispatch(None, cp_size=4, dcp_size=2)
                compressor = Mock(spec=["ratio"])
                compressor.ratio = ratio
                x = self._inputs(1)
                batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
                self.assertIsNone(
                    backend.forward_core_compressor(x, batch, 7, compressor)
                )
                compressor.assert_called_once_with(x, batch)

    def test_backend_idle_bypasses_compressor_state_and_cp_checks(self):
        backend = _backend_dispatch(None, cp_size=4, dcp_size=2)
        compressor = Mock()
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)
        x = self._inputs(0)
        self.assertIsNone(backend.forward_core_compressor(x, batch, 7, compressor))
        self.assertIsNone(
            backend.forward_low_ratio_compressor(
                compressor=compressor,
                x=x,
                positions=self._long([]),
                forward_batch=batch,
                layer_id=7,
            )
        )
        compressor.assert_not_called()

    def test_backend_rejects_cp_or_dcp_before_pair_state_is_updated(self):
        compressor = self._compressor(2)
        state = self._state()
        state[0].fill_(1.25)
        state[1].fill_(-3.5)
        pool = SimpleNamespace(
            c2_pair_kv_state={7: state[0]},
            c2_pair_score_state={7: state[1]},
            c2_pair_pad_row=self.req_slots,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            positions=self._long([0]),
            req_pool_indices=self._long([0]),
            out_cache_loc=self._long([128]),
            out_cache_loc_dsv4=None,
        )
        for cp_size, dcp_size in ((2, 1), (1, 2), (2, 2)):
            with self.subTest(cp_size=cp_size, dcp_size=dcp_size):
                backend = _backend_dispatch(pool, cp_size=cp_size, dcp_size=dcp_size)
                with self.assertRaises(NotImplementedError):
                    backend.forward_core_compressor(
                        self._inputs(1), batch, 7, compressor
                    )
                self.assertTrue(bool((state[0] == 1.25).all()))
                self.assertTrue(bool((state[1] == -3.5).all()))


class TestNpuDsv41CompressorCPU(_CompressorCases, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(_NPU_AVAILABLE, "requires an available torch_npu device")
class TestNpuDsv41CompressorNPU(_CompressorCases, unittest.TestCase):
    device = "npu"


if __name__ == "__main__":
    unittest.main()
