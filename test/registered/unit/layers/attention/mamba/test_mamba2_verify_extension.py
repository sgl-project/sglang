"""Mamba2 speculative target-verify extension - CPU only.

* ``Mamba2AttnBackend`` asks ``create_linear_attn_verify_extension(state)``
  once, with ``kind="mamba2"`` and the Mamba2 state dims, when its pool holds
  speculative state; builtins return None and keep the kernels.
* ``begin_forward`` gets a ``LinearAttnVerifyLayout`` for a target verify with
  tree links (eager and inside the CUDA graph), None otherwise; only then does
  the backend hand the extension to ``MambaMixer2.forward``.
* In the mixer's target-verify branch ``conv_update`` and ``ssm_update``
  replace ``causal_conv1d_update`` and ``selective_state_update`` when they
  handle the forward; otherwise the kernels run as before.

Backends are built with ``__new__`` or a patched base ``__init__``, the mixer
as a namespace with only the attributes ``forward`` reads; kernels are mocks.
"""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid_mod
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    Mamba2AttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear.verify_extension import (
    LinearAttnStateLayout,
    LinearAttnVerifyExtension,
    LinearAttnVerifyLayout,
)
from sglang.srt.layers.attention.mamba import mamba as mamba_mod
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# One request, verify width 3, TP 1: 2 heads x head_dim 4, 1 group, state 4.
BS, WIDTH, HEADS, HEAD_DIM, GROUPS, STATE = 1, 3, 2, 4, 1, 4
INTERMEDIATE = HEADS * HEAD_DIM
CONV_DIM = INTERMEDIATE + 2 * GROUPS * STATE


class _Recorder(LinearAttnVerifyExtension):
    def __init__(self, handle=True):
        self.handle = handle
        self.calls = []

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        self.calls.append(("init_cuda_graph_state", max_bs, max_num_tokens))

    def begin_forward(self, layout, *, in_graph):
        self.calls.append(("begin_forward", layout, in_graph))

    def conv_update(self, layer, mixed_qkv, **kwargs):
        self.calls.append(("conv_update", tuple(mixed_qkv.shape), sorted(kwargs)))
        return mixed_qkv if self.handle else None

    def ssm_update(self, layer, **kwargs):
        shapes = {k: tuple(v.shape) for k, v in kwargs.items()}
        self.calls.append(("ssm_update", shapes))
        if self.handle:
            kwargs["out"].fill_(7.0)
        return self.handle


def _speculative_state():
    return MambaPool.SpeculativeState(
        conv=[torch.zeros(2, 4, CONV_DIM, 3, dtype=torch.bfloat16)],
        temporal=torch.zeros(2, 4, HEADS, HEAD_DIM, STATE),
        intermediate_ssm=torch.zeros(2, BS, WIDTH, HEADS, HEAD_DIM, STATE),
        intermediate_conv_window=[torch.zeros(2, BS, WIDTH, CONV_DIM, 3)],
    )


class TestConstruction(CustomTestCase):
    def _build_backend(self, mamba_cache):
        asked = []
        extension = _Recorder()

        def create(state):
            asked.append(state)
            return extension

        model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                mamba_pool=SimpleNamespace(mamba_cache=mamba_cache)
            ),
            device="cpu",
            spec_algorithm=SimpleNamespace(create_linear_attn_verify_extension=create),
        )

        def base_init(self, runner):
            self.req_to_token_pool = runner.req_to_token_pool
            self.device = runner.device

        exec_args = SimpleNamespace(
            mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
        )
        with (
            mock.patch.object(MambaAttnBackendBase, "__init__", base_init),
            mock.patch.object(hybrid_mod, "get_exec", return_value=exec_args),
        ):
            backend = Mamba2AttnBackend(model_runner)
        return backend, extension, asked

    def test_backend_asks_the_algorithm_with_the_mamba2_layout(self):
        backend, extension, asked = self._build_backend(_speculative_state())
        self.assertIs(backend.verify_extension, extension)
        self.assertEqual(
            asked,
            [
                LinearAttnStateLayout(
                    conv_state_len=3,
                    conv_dtype=torch.bfloat16,
                    ssm_dtype=torch.float32,
                    num_v_heads=HEADS,
                    has_state_snapshots=True,
                    device=torch.device("cpu"),
                    kind="mamba2",
                    head_dim=HEAD_DIM,
                    ssm_state_size=STATE,
                    n_groups=GROUPS,
                )
            ],
        )

    def test_pool_without_speculative_state_gets_no_extension(self):
        state = _speculative_state()
        backend, _, asked = self._build_backend(
            MambaPool.State(conv=state.conv, temporal=state.temporal)
        )
        self.assertIsNone(backend.verify_extension)
        self.assertEqual(asked, [])


class TestLifecycle(CustomTestCase):
    def setUp(self):
        self.backend = Mamba2AttnBackend.__new__(Mamba2AttnBackend)
        self.extension = _Recorder()
        self.backend.verify_extension = self.extension
        self.backend._verify_extension_layout = None
        links = torch.zeros(BS, WIDTH, dtype=torch.int32)
        self.backend.forward_metadata = SimpleNamespace(
            retrieve_next_token=links,
            retrieve_next_sibling=links + 1,
            retrieve_parent_token=links + 2,
            mamba_cache_indices=torch.zeros(BS, dtype=torch.int32),
            query_start_loc=torch.tensor([0, WIDTH], dtype=torch.int32),
        )
        self.backend.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=_speculative_state())
        )

    def _begin(self, mode, in_graph=True):
        self.backend._begin_verify_extension(
            SimpleNamespace(
                forward_mode=mode, spec_info=SimpleNamespace(draft_token_num=WIDTH)
            ),
            in_graph=in_graph,
        )

    def test_cuda_graph_state(self):
        with mock.patch.object(MambaAttnBackendBase, "init_cuda_graph_state") as base:
            self.backend.init_cuda_graph_state(4, 32)
        base.assert_called_once_with(4, 32)
        self.assertEqual(self.extension.calls, [("init_cuda_graph_state", 4, 32)])

    def test_in_graph_metadata_begins_the_forward(self):
        with mock.patch.object(
            MambaAttnBackendBase, "init_forward_metadata_in_graph"
        ) as base:
            self.backend.init_forward_metadata_in_graph(
                SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    spec_info=SimpleNamespace(draft_token_num=WIDTH),
                )
            )
        base.assert_called_once()
        ((name, layout, in_graph),) = self.extension.calls
        self.assertEqual((name, in_graph), ("begin_forward", True))
        self.assertIsInstance(layout, LinearAttnVerifyLayout)

    def test_tree_verify_gets_its_layout(self):
        self._begin(ForwardMode.TARGET_VERIFY, in_graph=False)
        ((name, layout, in_graph),) = self.extension.calls
        metadata = self.backend.forward_metadata
        self.assertEqual((name, in_graph), ("begin_forward", False))
        self.assertIs(layout.parent_token, metadata.retrieve_parent_token)
        self.assertIs(layout.cache_indices, metadata.mamba_cache_indices)
        self.assertIs(layout.query_start_loc, metadata.query_start_loc)
        self.assertEqual(layout.width, WIDTH)
        self.assertIs(self.backend._verify_extension_layout, layout)

    def test_other_forwards_get_none(self):
        self._begin(ForwardMode.DECODE)
        self.backend.forward_metadata.retrieve_parent_token = None
        self._begin(ForwardMode.TARGET_VERIFY)
        self.assertEqual(
            [layout for _, layout, _ in self.extension.calls], [None, None]
        )
        self.assertIsNone(self.backend._verify_extension_layout)

    def test_tree_verify_without_snapshots_raises(self):
        pool = self.backend.req_to_token_pool.mamba_pool
        pool.mamba_cache = dataclasses.replace(pool.mamba_cache, intermediate_ssm=None)
        with self.assertRaisesRegex(RuntimeError, "state snapshot"):
            self._begin(ForwardMode.TARGET_VERIFY)

    def test_forward_hands_the_extension_to_the_mixer_only_with_a_layout(self):
        mixer = mock.Mock()
        mixer.forward.return_value = (torch.zeros(1), None, None)
        self.backend.req_to_token_pool.mamba2_layer_cache = lambda layer_id: None
        self.backend.forward_metadata = mock.Mock(spec=hybrid_mod.Mamba2Metadata)
        forward_batch = SimpleNamespace(mamba_track_mask=None)
        memory = SimpleNamespace(enable_page_major_kv_layout=False)
        with mock.patch.object(hybrid_mod, "get_memory", return_value=memory):
            for layout in (None, object()):
                self.backend._verify_extension_layout = layout
                self.backend.forward(mixer, torch.zeros(1), None, 0, forward_batch)
        handed = [c.kwargs["verify_extension"] for c in mixer.forward.call_args_list]
        self.assertEqual(handed, [None, self.extension])


def _mixer():
    """A MambaMixer2 stand-in with only what ``forward``'s verify path reads."""
    proj_dim = INTERMEDIATE + CONV_DIM + HEADS
    return SimpleNamespace(
        in_proj=lambda h: (torch.randn(h.shape[0], proj_dim), None),
        conv1d=SimpleNamespace(weight=torch.zeros(CONV_DIM, 1, 4), bias=None),
        intermediate_size=INTERMEDIATE,
        conv_dim=CONV_DIM,
        num_heads=HEADS,
        tp_size=1,
        n_groups=GROUPS,
        groups_ssm_state_size=GROUPS * STATE,
        head_dim=HEAD_DIM,
        ssm_state_size=STATE,
        activation="silu",
        A=torch.ones(HEADS),
        D=torch.ones(HEADS),
        dt_bias=torch.ones(HEADS),
        norm=lambda x, gate: x,
        out_proj=lambda h: (h, None),
    )


class TestMixerVerify(CustomTestCase):
    def _run(self, extension):
        metadata = SimpleNamespace(
            mamba_cache_indices=torch.zeros(BS, dtype=torch.int32),
            query_start_loc=torch.tensor([0, WIDTH], dtype=torch.int32),
            num_prefills=0,
            num_decodes=BS,
            num_prefill_tokens=0,
            draft_token_num=WIDTH,
            is_target_verify=True,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=torch.zeros(BS, WIDTH, dtype=torch.int32),
        )
        conv = mock.Mock(side_effect=lambda x, *a, **k: x)
        ssu = mock.Mock()
        with (
            mock.patch.object(
                mamba_mod, "causal_conv1d_update_triton", conv, create=True
            ),
            mock.patch.object(mamba_mod, "selective_state_update", ssu),
        ):
            out, _, _ = MambaMixer2.forward(
                _mixer(),
                hidden_states=torch.zeros(BS * WIDTH, 8),
                layer_cache=_speculative_state(),
                metadata=metadata,
                use_triton_causal_conv=True,
                verify_extension=extension,
            )
        return out, conv, ssu

    def test_handled_verify_skips_the_kernels(self):
        extension = _Recorder(handle=True)
        out, conv, ssu = self._run(extension)
        conv.assert_not_called()
        ssu.assert_not_called()
        (conv_call, ssm_call) = extension.calls
        self.assertEqual(conv_call[:2], ("conv_update", (BS, CONV_DIM, WIDTH)))
        self.assertEqual(
            conv_call[2],
            [
                "cache_indices",
                "conv_states",
                "intermediate_conv_window",
                "intermediate_state_indices",
            ],
        )
        name, shapes = ssm_call
        self.assertEqual(name, "ssm_update")
        verify = (BS, WIDTH, HEADS, HEAD_DIM)
        for key in ("x", "dt", "out"):
            self.assertEqual(shapes[key], verify)
        self.assertEqual(shapes["B"], (BS, WIDTH, GROUPS, STATE))
        self.assertEqual(shapes["A"], (HEADS, HEAD_DIM, STATE))
        self.assertTrue(torch.all(out == 7.0))

    def test_unhandled_verify_runs_the_kernels(self):
        extension = _Recorder(handle=False)
        _, conv, ssu = self._run(extension)
        conv.assert_called_once()
        ssu.assert_called_once()
        self.assertEqual(ssu.call_args.kwargs["cache_steps"], WIDTH)
        self.assertEqual([c[0] for c in extension.calls], ["conv_update", "ssm_update"])

    def test_no_extension_runs_the_kernels(self):
        _, conv, ssu = self._run(None)
        conv.assert_called_once()
        ssu.assert_called_once()


if __name__ == "__main__":
    unittest.main()
