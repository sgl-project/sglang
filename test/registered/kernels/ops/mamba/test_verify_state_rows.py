"""Exercise actual Mamba2/LFM2 verify callers, kernels, and accepted-state commit."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
    fused_mamba_state_scatter_with_mask,
)
from sglang.kernels.ops.mamba.triton_ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import Mamba2AttnBackend
from sglang.srt.layers.attention.linear.short_conv_backend import ShortConvAttnBackend
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.lfm2 import shortconv_target_verify
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_STABLE_ROWS = "sglang.srt.layers.attention.linear.utils.pp_spec_stable_rows_enabled"


@pytest.fixture(autouse=True)
def triton_ssu():
    initialize_mamba_selective_state_update_backend(
        SimpleNamespace(
            mamba_backend="triton",
            enable_mamba_cache_stochastic_rounding=False,
            mamba_cache_philox_rounds=0,
        )
    )


class VerifyCase:
    """Small real state buffers; only projections/norm and the scheduler are absent."""

    def __init__(self, backend, case="stable", dedup=False, batch_size=None):
        self.backend = backend
        self.stable = case != "ordinary"
        self.bs = batch_size or (9 if case in ("padding", "last_row") else 2)
        self.steps, self.channels, self.history = 3, 8, 2
        self.rows = torch.tensor(
            ([0, 1] if case == "ordinary" else [4, 2]) + [0] * (self.bs - 2),
            device="cuda",
        )[: self.bs]
        self.slots = torch.tensor([3, 5] + [-1] * (self.bs - 2), device="cuda")[
            : self.bs
        ]
        self.accepted = torch.tensor([1, 2] + [-1] * (self.bs - 2), device="cuda")[
            : self.bs
        ]
        self.x = (
            torch.arange(
                1, self.bs * 3 * 8 + 1, device="cuda", dtype=torch.float32
            ).reshape(self.bs, 3, 8)
            / 32
        )
        if case == "reordered":
            self.rows = self.rows.flip(0)
            self.slots = self.slots.flip(0)
            self.accepted = self.accepted.flip(0)
            self.x = self.x.flip(0)
        if case == "last_row":
            self.rows[0] = (
                6  # Last live row also receives padding indices; kernels skip -1.
            )
        state = torch.zeros(12, 8, 2, device="cuda")
        ssm = torch.zeros(12, 1, 4, 2, device="cuda")
        if case in ("nonzero", "padding"):
            state.copy_(torch.arange(1, 13, device="cuda")[:, None, None] / 16)
            ssm.copy_(torch.arange(1, 13, device="cuda")[:, None, None, None] / 16)
        if dedup:
            backing = torch.full((7, 8, 4), -7.0, device="cuda")
            windows = backing.unfold(-1, 2, 1).permute(0, 2, 1, 3)
        else:
            windows = torch.full((7, 3, 8, 2), -7.0, device="cuda")
        self.cache = MambaPool.SpeculativeState(
            conv=[state],
            temporal=ssm,
            intermediate_conv_window=[windows],
            intermediate_ssm=torch.full((7, 3, 1, 4, 2), -9.0, device="cuda"),
        )
        self.initial_conv, self.initial_ssm = state.clone(), ssm.clone()
        self.initial_windows = windows.clone()
        self.initial_ssm_windows = self.cache.intermediate_ssm.clone()
        self.weight = torch.tensor([0.25, 0.5, 1.0], device="cuda").repeat(8, 1)
        self.conv = SimpleNamespace(
            conv_weight=self.weight,
            conv_bias=None,
            _intermediate_state_indices=torch.arange(
                2, device="cuda", dtype=torch.int32
            ),
        )
        self.mixer = SimpleNamespace(
            in_proj=lambda x: (x, None),
            out_proj=lambda x: (x, None),
            norm=lambda x, gate: x,
            intermediate_size=4,
            conv_dim=8,
            groups_ssm_state_size=2,
            tp_size=1,
            num_heads=1,
            n_groups=1,
            head_dim=4,
            ssm_state_size=2,
            activation=None,
            conv1d=SimpleNamespace(weight=self.weight[:, None], bias=None),
            A=torch.tensor([-0.125], device="cuda"),
            D=torch.tensor([0.25], device="cuda"),
            dt_bias=torch.tensor([0.125], device="cuda"),
        )
        self.md = ForwardMetadata(
            query_start_loc=torch.arange(self.bs + 1, device="cuda", dtype=torch.int32)
            * 3,
            mamba_cache_indices=self.slots,
        )
        self.batch = SimpleNamespace(
            req_pool_indices=self.rows,
            seq_lens=torch.ones(self.bs, device="cuda"),
            forward_mode=ForwardMode.TARGET_VERIFY,
            logical_forward_mode=ForwardMode.TARGET_VERIFY,
            extend_num_tokens=None,
            spec_info=SimpleNamespace(draft_token_num=3),
        )
        # This is the real sidecar's metadata construction, without a full runner.
        self.sidecar = object.__new__(ShortConvAttnBackend)
        self.sidecar.state_layer_cache = lambda layer_id: self.cache
        self.sidecar.forward_metadata = self.md
        self.sidecar._cache_indices = self.slots
        self.sidecar._has_initial_state = None
        self.sidecar._slot_ids_cpu = self.sidecar._has_prefix_cpu = None
        self.projected = torch.cat(
            [
                torch.zeros(self.bs, 3, 4, device="cuda"),
                self.x,
                torch.full((self.bs, 3, 1), -0.5, device="cuda"),
            ],
            -1,
        ).reshape(self.bs * 3, 13)

    def run(self):
        if self.backend == "lfm2":
            meta = self.sidecar.conv_state_metadata(0, self.batch)
            return shortconv_target_verify(
                self.conv, self.x.reshape(-1, 8), meta, 3, "LFM2"
            )
        metadata = Mamba2Metadata.prepare_mixed(self.md, 16, self.batch)
        return MambaMixer2.forward(
            self.mixer,
            hidden_states=self.projected,
            layer_cache=self.cache,
            metadata=metadata,
            use_triton_causal_conv=True,
        )[0]

    def expected(self):
        conv_windows, ssm_windows, outputs = [], [], []
        for req, slot in enumerate(self.slots.tolist()):
            if slot < 0:
                conv_windows.append(None)
                ssm_windows.append(None)
                outputs.append(None)
                continue
            joined = torch.cat([self.initial_conv[slot], self.x[req].T], -1)
            windows = torch.stack([joined[:, s + 1 : s + 3] for s in range(3)])
            conv = F.conv1d(joined[None], self.weight[:, None], groups=8)[0].T
            conv_windows.append(windows)
            if self.backend == "lfm2":
                ssm_windows.append(None)
                outputs.append(conv)
                continue
            state = self.initial_ssm[slot].clone()
            dt = F.softplus(torch.tensor(-0.375, device="cuda"))
            snapshots, out = [], []
            for step in range(3):
                x, b, c = conv[step].split([4, 2, 2])
                state = (
                    state * torch.exp(-0.125 * dt)
                    + dt * b[None, None, :] * x[None, :, None]
                )
                snapshots.append(state.clone())
                out.append((state * c).sum(-1).flatten() + 0.25 * x)
            ssm_windows.append(torch.stack(snapshots))
            outputs.append(torch.stack(out))
        return conv_windows, ssm_windows, outputs

    def check(self, output):
        conv_windows, ssm_windows, outputs = self.expected()
        expected_conv = self.initial_windows.clone()
        expected_ssm = self.initial_ssm_windows.clone()
        write_rows = self.rows.tolist() if self.stable else list(range(self.bs))
        for req, slot in enumerate(self.slots.tolist()):
            if slot < 0:
                continue  # Both real kernels skip PAD_SLOT_ID == -1.
            expected_conv[write_rows[req]] = conv_windows[req]
            if self.backend == "mamba2":
                expected_ssm[write_rows[req]] = ssm_windows[req]
            torch.testing.assert_close(
                output.reshape(self.bs, 3, -1)[req], outputs[req], rtol=1e-5, atol=1e-6
            )
        # The existing conv update advances live persistent windows through the
        # whole draft; commit then rolls them back to the accepted step.
        advanced_conv = self.initial_conv.clone()
        for req, slot in enumerate(self.slots.tolist()):
            if slot >= 0:
                advanced_conv[slot] = conv_windows[req][-1]
        torch.testing.assert_close(self.cache.conv[0], advanced_conv, rtol=0, atol=0)
        torch.testing.assert_close(
            self.cache.temporal, self.initial_ssm, rtol=0, atol=0
        )
        torch.testing.assert_close(
            self.cache.intermediate_conv_window[0], expected_conv, rtol=0, atol=0
        )
        torch.testing.assert_close(
            self.cache.intermediate_ssm, expected_ssm, rtol=1e-5, atol=1e-6
        )
        for kind, initial, scratch, snapshots in (
            (
                "conv",
                self.initial_conv,
                self.cache.intermediate_conv_window[0],
                conv_windows,
            ),
            ("ssm", self.initial_ssm, self.cache.intermediate_ssm, ssm_windows),
        ):
            if kind == "ssm" and self.backend != "mamba2":
                continue
            persistent = self.cache.conv[0] if kind == "conv" else self.cache.temporal
            committed = persistent.clone()[None]
            rows = self.rows if self.stable else torch.arange(self.bs, device="cuda")
            scatter = (
                fused_conv_window_scatter_with_mask
                if kind == "conv"
                else fused_mamba_state_scatter_with_mask
            )
            scatter(committed, scratch[None], self.slots, self.accepted, rows)
            expected = initial.clone()
            for req, step in enumerate(self.accepted.tolist()):
                if step >= 0:
                    expected[self.slots[req]] = snapshots[req][step]
            torch.testing.assert_close(
                committed[0],
                expected,
                rtol=1e-5 if kind == "ssm" else 0,
                atol=1e-6 if kind == "ssm" else 0,
            )
            print(
                f"{self.backend} {kind}: rows={rows.tolist()}, accepted={self.accepted.tolist()}, committed={committed[0, self.slots[:2]].flatten(1)[:, :2].tolist()}"
            )


@pytest.mark.parametrize("backend", ["mamba2", "lfm2"])
@pytest.mark.parametrize(
    "case", ["ordinary", "stable", "reordered", "nonzero", "padding", "last_row"]
)
@pytest.mark.parametrize("dedup", [False, True])
def test_verify_commit(backend, case, dedup):
    test = VerifyCase(backend, case, dedup)
    with patch(_STABLE_ROWS, return_value=test.stable):
        output = test.run()
    test.check(output)


@pytest.mark.parametrize("backend", ["mamba2", "lfm2"])
def test_batch_size_changes_keep_current_rows(backend):
    previous = None
    for bs in (2, 9, 2):
        test = VerifyCase(backend, "padding" if bs > 2 else "reordered", batch_size=bs)
        if previous is not None:
            test.conv = previous.conv
            test.mixer = previous.mixer
        with patch(_STABLE_ROWS, return_value=True):
            output = test.run()
        test.check(output)
        previous = test


@pytest.mark.parametrize("backend", ["mamba2", "lfm2"])
def test_delayed_microbatch_commit(backend):
    # Model-free, single-device simulation: two verifies before either commit.
    first, second = VerifyCase(backend, "nonzero"), VerifyCase(backend, "nonzero")
    second.rows.copy_(torch.tensor([1, 5], device="cuda"))
    second.slots.copy_(torch.tensor([7, 8], device="cuda"))
    # Share the full pool, with disjoint persistent slots and scratch rows.
    second.cache = first.cache
    with patch(_STABLE_ROWS, return_value=True):
        first_output = first.run()
        second_output = second.run()
    # Each check includes all unaffected scratch rows, including the other batch.
    first.initial_windows[second.rows] = first.cache.intermediate_conv_window[0][
        second.rows
    ]
    first.initial_ssm_windows[second.rows] = first.cache.intermediate_ssm[second.rows]
    second.initial_windows[first.rows] = first.cache.intermediate_conv_window[0][
        first.rows
    ]
    second.initial_ssm_windows[first.rows] = first.cache.intermediate_ssm[first.rows]
    first_windows = first.expected()[0]
    second_windows = second.expected()[0]
    for req, slot in enumerate(second.slots.tolist()):
        first.initial_conv[slot] = second_windows[req][-1]
    for req, slot in enumerate(first.slots.tolist()):
        second.initial_conv[slot] = first_windows[req][-1]
    first.check(first_output)
    second.check(second_output)
    # Commit both delayed batches into the SAME persistent pool.
    for persistent, scratch, reference_index in (
        (first.cache.conv[0], first.cache.intermediate_conv_window[0], 0),
        (first.cache.temporal, first.cache.intermediate_ssm, 1),
    ):
        if reference_index == 1 and backend != "mamba2":
            continue
        expected = persistent.clone()
        for test in (first, second):
            snapshots = test.expected()[reference_index]
            scatter = (
                fused_conv_window_scatter_with_mask
                if reference_index == 0
                else fused_mamba_state_scatter_with_mask
            )
            scatter(
                persistent[None], scratch[None], test.slots, test.accepted, test.rows
            )
            for req, slot in enumerate(test.slots.tolist()):
                expected[slot] = snapshots[req][test.accepted[req]]
        torch.testing.assert_close(persistent, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["mamba2", "lfm2"])
@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="CUDA JIT only")
def test_graph_replay_reads_updated_request_rows(backend):
    test = VerifyCase(backend, "nonzero")
    with patch(_STABLE_ROWS, return_value=True):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                test.cache.conv[0].copy_(test.initial_conv)
                test.run()
        torch.cuda.current_stream().wait_stream(stream)
        test.cache.conv[0].copy_(test.initial_conv)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = test.run()
        for rows in ([4, 2], [2, 4]):
            test.rows.copy_(torch.tensor(rows, device="cuda"))
            test.cache.conv[0].copy_(test.initial_conv)
            test.cache.intermediate_conv_window[0].copy_(test.initial_windows)
            test.cache.intermediate_ssm.copy_(test.initial_ssm_windows)
            graph.replay()
            test.check(output)


@pytest.mark.parametrize("backend", ["mamba2", "lfm2"])
def test_ordinary_verify_reuses_indices(backend):
    test = VerifyCase(backend, "ordinary")
    with patch(_STABLE_ROWS, return_value=False):
        test.run()
        test.cache.conv[0].copy_(test.initial_conv)
        with patch.object(
            torch, "arange", side_effect=AssertionError("reallocated rows")
        ):
            output = test.run()
    test.check(output)


def test_mamba2_graph_metadata_forwards_request_rows():
    test = VerifyCase("mamba2")
    backend = object.__new__(Mamba2AttnBackend)
    backend._replay_metadata = lambda *args, **kwargs: test.md
    batch = SimpleNamespace(
        **vars(test.batch),
        batch_size=test.bs,
        seq_lens_cpu=None,
        ragged_verify_layout=None,
    )
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    assert backend.forward_metadata.req_pool_indices is test.rows


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
