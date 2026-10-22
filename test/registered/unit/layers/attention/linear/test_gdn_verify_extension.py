"""GDN speculative target-verify extension and variable verify width - CPU only.

* ``create_linear_attn_verify_extension(state)`` lets a speculative algorithm
  replace the GDN target-verify conv1d and recurrence of tree verifies
  (``LinearAttnVerifyExtension``); it sees the state layout and each forward's
  tree layout as small typed values. Builtins return None and keep the kernels.
* ``target_verify_takes_projection_views`` says whether a verify takes views
  of the layer's input projection: the extension answers for the tree verifies
  it handles (builtins take split copies), so the layer and the backend skip
  their split kernels.
* The linear-attention CUDA-graph metadata follows the forward's verify width,
  so graphs narrower than the captured width (a runner may capture several
  verify widths) get query starts and tree links of their own width.

Backends are built with ``__new__`` and only the attributes each method reads.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid_mod
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear import gdn_backend as gdn_mod
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.verify_extension import (
    LinearAttnStateLayout,
    LinearAttnVerifyExtension,
    LinearAttnVerifyLayout,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _Recorder(LinearAttnVerifyExtension):
    def __init__(self, handle=True, strided=False):
        self.handle = handle
        self.strided = strided
        self.calls = []
        self.attn_out = torch.full((1, 4, 1, 8), 7.0)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        self.calls.append(("init_cuda_graph_state", max_bs, max_num_tokens))

    def begin_forward(self, layout, *, in_graph):
        self.calls.append(("begin_forward", layout, in_graph))

    def reads_strided_inputs(self):
        return self.strided

    def conv_update(self, layer, mixed_qkv, **kwargs):
        self.calls.append(("conv_update", tuple(mixed_qkv.shape), sorted(kwargs)))
        return mixed_qkv if self.handle else None

    def recurrent_update(self, layer, **kwargs):
        self.calls.append(("recurrent_update", sorted(kwargs)))
        return self.attn_out if self.handle else None


class TestDefaults(CustomTestCase):
    def test_builtin_and_plugin_return_no_extension(self):
        self.assertIsNone(
            SpeculativeAlgorithm.EAGLE.create_linear_attn_verify_extension(None)
        )
        plugin = CustomSpecAlgo(name="PLUGIN", factory=lambda server_args: None)
        self.assertIsNone(plugin.create_linear_attn_verify_extension(None))

    def _build_backend(self, mamba_cache):
        asked = []
        extension = _Recorder()

        def create(state):
            asked.append(state)
            return extension

        model_runner = SimpleNamespace(
            linear_attn_backends=SimpleNamespace(
                decode=None, prefill=None, verify=None
            ),
            req_to_token_pool=SimpleNamespace(
                size=4, mamba_pool=SimpleNamespace(mamba_cache=mamba_cache)
            ),
            device="cpu",
            spec_algorithm=SimpleNamespace(create_linear_attn_verify_extension=create),
        )

        def base_init(self, runner):
            self.req_to_token_pool = runner.req_to_token_pool
            self.device = runner.device

        with (
            mock.patch.object(MambaAttnBackendBase, "__init__", base_init),
            mock.patch.object(gdn_mod, "_validate_gdn_linear_attn_backends"),
            mock.patch.object(gdn_mod, "validate_gdn_mis_backend"),
            mock.patch.object(gdn_mod, "GDNKernelDispatcher"),
            mock.patch.object(gdn_mod, "build_verify_intermediate_state_indices"),
            mock.patch.object(
                gdn_mod,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_mis=False)
                ),
            ),
        ):
            backend = GDNAttnBackend(model_runner)
        return backend, extension, asked

    def test_backend_asks_the_algorithm(self):
        backend, extension, asked = self._build_backend(
            MambaPool.SpeculativeState(
                conv=[torch.zeros(4, 8, 3, dtype=torch.bfloat16)],
                temporal=torch.zeros(2, 4, 6, 8, 8),
                intermediate_ssm=torch.zeros(1),
                intermediate_conv_window=[torch.zeros(1)],
            )
        )
        self.assertIs(backend.verify_extension, extension)
        self.assertEqual(
            asked,
            [
                LinearAttnStateLayout(
                    conv_state_len=3,
                    conv_dtype=torch.bfloat16,
                    ssm_dtype=torch.float32,
                    num_v_heads=6,
                    has_state_snapshots=True,
                    device=torch.device("cpu"),
                )
            ],
        )

    def test_pool_without_speculative_state_gets_no_extension(self):
        # A server without speculation holds a plain State (no snapshots).
        backend, _, asked = self._build_backend(
            MambaPool.State(
                conv=[torch.zeros(4, 8, 3, dtype=torch.bfloat16)],
                temporal=torch.zeros(2, 4, 6, 8, 8),
            )
        )
        self.assertIsNone(backend.verify_extension)
        self.assertEqual(asked, [])


class TestLifecycle(CustomTestCase):
    def setUp(self):
        self.backend = GDNAttnBackend.__new__(GDNAttnBackend)
        self.extension = _Recorder()
        self.backend.verify_extension = self.extension

    def test_cuda_graph_state(self):
        with mock.patch.object(MambaAttnBackendBase, "init_cuda_graph_state") as base:
            self.backend.init_cuda_graph_state(4, 32)
        base.assert_called_once_with(4, 32)
        self.assertEqual(self.extension.calls, [("init_cuda_graph_state", 4, 32)])

    def _tree_verify(self, mamba_pool):
        links = torch.zeros(1, 4, dtype=torch.int32)
        self.backend.forward_metadata = SimpleNamespace(
            retrieve_next_token=links,
            retrieve_next_sibling=links + 1,
            retrieve_parent_token=links + 2,
            mamba_cache_indices=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        )
        self.backend.req_to_token_pool = SimpleNamespace(mamba_pool=mamba_pool)
        self.backend._begin_verify_extension(
            SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=SimpleNamespace(draft_token_num=4),
            ),
            in_graph=True,
        )

    def test_tree_verify_gets_its_layout(self):
        snapshots = SimpleNamespace(mamba_cache=SimpleNamespace(intermediate_ssm=1))
        self._tree_verify(snapshots)
        ((name, layout, in_graph),) = self.extension.calls
        metadata = self.backend.forward_metadata
        self.assertEqual((name, in_graph), ("begin_forward", True))
        self.assertIsInstance(layout, LinearAttnVerifyLayout)
        self.assertIs(layout.parent_token, metadata.retrieve_parent_token)
        self.assertIs(layout.cache_indices, metadata.mamba_cache_indices)
        self.assertEqual(layout.width, 4)
        self.assertIs(self.backend._verify_extension_layout, layout)

    def test_tree_verify_off_the_snapshot_path_raises(self):
        for mamba_pool in (
            SimpleNamespace(mamba_cache=SimpleNamespace(intermediate_ssm=None)),
            SimpleNamespace(
                mamba_cache=SimpleNamespace(intermediate_ssm=1),
                replayssm_spec_fold=True,
            ),
        ):
            with self.assertRaises(RuntimeError):
                self._tree_verify(mamba_pool)

    def test_metadata_in_graph_and_eager(self):
        metadata = SimpleNamespace(has_mamba_track_mask=False)
        self.backend.forward_metadata = metadata
        self.backend.init_forward_metadata_in_graph(
            SimpleNamespace(forward_mode=ForwardMode.DECODE)
        )

        def base_eager(self, forward_batch):
            self.forward_metadata = metadata

        self.backend.enable_mis = False
        with mock.patch.object(
            MambaAttnBackendBase, "init_forward_metadata", base_eager
        ):
            self.backend.init_forward_metadata(
                SimpleNamespace(
                    multi_item_delimiter_indices=None,
                    forward_mode=ForwardMode.DECODE,
                )
            )
        self.assertEqual(
            self.extension.calls,
            [("begin_forward", None, True), ("begin_forward", None, False)],
        )


def _verify_backend(extension, bs=1, width=4, tree=True):
    backend = GDNAttnBackend.__new__(GDNAttnBackend)
    backend.verify_extension = extension
    # What begin_forward recorded for this forward (a tree verify or not).
    backend._verify_extension_layout = object() if tree else None
    backend._use_strided_target_verify_qkv = False
    backend.mis_metadata = None
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=torch.arange(bs + 1, dtype=torch.int32) * width,
        mamba_cache_indices=torch.zeros(bs, dtype=torch.int32),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    layer_cache = MambaPool.SpeculativeState(
        conv=[torch.zeros(2, 16, 3)],
        temporal=torch.zeros(2, 1, 4, 8),
        intermediate_ssm=torch.zeros(2, width, 1, 4, 8),
        intermediate_conv_window=[torch.zeros(2, width, 16, 3)],
    )
    backend.req_to_token_pool = SimpleNamespace(
        size=4,
        mamba2_layer_cache=lambda layer_id: layer_cache,
        mamba_pool=SimpleNamespace(),
    )
    backend.verify_intermediate_state_indices = torch.arange(4, dtype=torch.int32)
    backend.kernel_dispatcher = mock.MagicMock()
    layer = SimpleNamespace(
        layer_id=0,
        q_dim=4,
        k_dim=4,
        v_dim=8,
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=1,
        head_q_dim=4,
        head_k_dim=4,
        head_v_dim=8,
        conv_weights=None,
        bias=None,
        activation="silu",
        A_log=None,
        dt_bias=None,
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(draft_token_num=width),
        req_pool_indices=torch.zeros(bs, dtype=torch.int64),
    )
    mixed_qkv = torch.randn(bs * width, 16)
    return backend, layer, forward_batch, mixed_qkv


class TestTargetVerifyExtension(CustomTestCase):
    def setUp(self):
        # The inputs are CPU tensors: split q/k/v with torch on every host, as
        # on a CPU-only one, instead of the Triton kernel CUDA hosts pick.
        patcher = mock.patch.object(gdn_mod, "is_cuda", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_extension_replaces_conv_and_recurrence(self):
        extension = _Recorder(handle=True)
        backend, layer, fb, mixed_qkv = _verify_backend(extension)
        with mock.patch.object(gdn_mod, "causal_conv1d_update") as conv:
            out = backend.forward_extend(
                layer, fb, mixed_qkv, a=torch.zeros(4, 1), b=torch.zeros(4, 1)
            )
        conv.assert_not_called()
        backend.kernel_dispatcher.target_verify.assert_not_called()
        self.assertIs(out, extension.attn_out)
        names = [call[0] for call in extension.calls]
        self.assertEqual(names, ["conv_update", "recurrent_update"])
        # Conv input is (batch, dim, width).
        self.assertEqual(extension.calls[0][1], (1, 16, 4))
        self.assertIn("intermediate_states", extension.calls[1][1])

    def test_declining_extension_keeps_the_backend_kernels(self):
        extension = _Recorder(handle=False)
        backend, layer, fb, mixed_qkv = _verify_backend(extension)
        sentinel = torch.zeros(1)
        backend.kernel_dispatcher.target_verify.return_value = sentinel
        with mock.patch.object(
            gdn_mod, "causal_conv1d_update", side_effect=lambda x, *a, **k: x
        ) as conv:
            out = backend.forward_extend(
                layer, fb, mixed_qkv, a=torch.zeros(4, 1), b=torch.zeros(4, 1)
            )
        conv.assert_called_once()
        backend.kernel_dispatcher.target_verify.assert_called_once()
        self.assertIs(out, sentinel)

    def test_forward_without_a_tree_layout_skips_the_extension(self):
        extension = _Recorder(handle=True)
        backend, layer, fb, mixed_qkv = _verify_backend(extension, tree=False)
        backend.kernel_dispatcher.target_verify.return_value = torch.zeros(1)
        with mock.patch.object(
            gdn_mod, "causal_conv1d_update", side_effect=lambda x, *a, **k: x
        ):
            backend.forward_extend(
                layer, fb, mixed_qkv, a=torch.zeros(4, 1), b=torch.zeros(4, 1)
            )
        self.assertEqual(extension.calls, [])
        backend.kernel_dispatcher.target_verify.assert_called_once()


class TestStridedInputs(CustomTestCase):
    def test_predicate_follows_the_extension_of_a_tree_verify(self):
        verify = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        decode = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        base = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
        self.assertFalse(base.target_verify_takes_projection_views(verify))
        for strided, tree, fb, expected in (
            (True, True, verify, True),
            (False, True, verify, False),
            (True, False, verify, False),
            (True, True, decode, False),
        ):
            backend, *_ = _verify_backend(_Recorder(strided=strided), tree=tree)
            self.assertIs(backend.target_verify_takes_projection_views(fb), expected)

    def test_strided_extension_gets_views_instead_of_the_split_kernel(self):
        for strided in (True, False):
            extension = _Recorder(strided=strided)
            backend, layer, fb, mixed_qkv = _verify_backend(extension)
            # The backend's own routing never applies to a forward it hands out.
            backend._use_strided_target_verify_qkv = not strided
            split = mock.MagicMock(
                return_value=(
                    torch.zeros(1, 4, 1, 4),
                    torch.zeros(1, 4, 1, 4),
                    torch.zeros(1, 4, 1, 8),
                )
            )
            with (
                mock.patch.object(gdn_mod, "is_cuda", return_value=True),
                mock.patch.object(
                    gdn_mod, "fused_qkv_split_gdn_prefill", split, create=True
                ),
            ):
                backend.forward_extend(
                    layer, fb, mixed_qkv, a=torch.zeros(4, 1), b=torch.zeros(4, 1)
                )
            self.assertEqual(split.called, not strided)


def _hybrid_backend(max_bs=2, captured_width=4, topk=4):
    backend = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
    backend.device = "cpu"
    backend.pad_slot_id = -1
    backend.topk = topk
    backend.state_indices_list = []
    backend.query_start_loc_list = []
    backend.retrieve_next_token_list = []
    backend.retrieve_next_sibling_list = []
    backend.retrieve_parent_token_list = []
    backend.req_to_token_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(enable_linear_replayssm=False),
        get_mamba_indices=lambda idx: idx.to(torch.int32),
    )
    backend._translate_mamba_indices = lambda indices: indices
    backend._fused_state_indices_ok = False
    backend.init_cuda_graph_state(max_bs, max_bs * captured_width)
    return backend


def _tree_input(bs, width):
    return SimpleNamespace(
        draft_token_num=width,
        ragged_verify_layout=None,
        retrieve_next_token=torch.arange(bs * width, dtype=torch.int32).view(bs, width)
        + 1,
        retrieve_next_sibling=torch.arange(bs * width, dtype=torch.int32).view(
            bs, width
        )
        + 100,
    )


class TestVariableVerifyWidth(CustomTestCase):
    def _replay(self, backend, bs, spec_info):
        with mock.patch.object(
            hybrid_mod, "_validate_mamba_replay_state_indices", False
        ):
            return backend._replay_metadata(
                bs,
                torch.zeros(bs, dtype=torch.int64),
                ForwardMode.TARGET_VERIFY,
                spec_info,
                seq_lens_cpu=None,
                num_padding=0,
            )

    def test_narrow_verify_gets_its_own_query_starts_and_links(self):
        backend = _hybrid_backend()
        spec_info = _tree_input(bs=2, width=2)
        metadata = self._replay(backend, 2, spec_info)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 2, 4])
        links = backend.retrieve_next_token_list[1]
        self.assertEqual(links[:, :2].tolist(), spec_info.retrieve_next_token.tolist())
        self.assertEqual(links[:, 2:].tolist(), [[0, 0], [0, 0]])
        siblings = backend.retrieve_next_sibling_list[1]
        self.assertEqual(
            siblings[:, :2].tolist(), spec_info.retrieve_next_sibling.tolist()
        )

    def test_captured_width_is_unchanged(self):
        backend = _hybrid_backend()
        spec_info = _tree_input(bs=2, width=4)
        metadata = self._replay(backend, 2, spec_info)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 4, 8])
        self.assertIs(
            backend._verify_query_start_loc(4),
            backend.cached_cuda_graph_verify_query_start_loc,
        )
        self.assertEqual(
            backend.retrieve_next_token_list[1].tolist(),
            spec_info.retrieve_next_token.tolist(),
        )

    def test_plan_stream_refill_takes_narrow_links(self):
        backend = _hybrid_backend()
        spec_info = _tree_input(bs=1, width=2)
        with mock.patch.object(hybrid_mod, "EagleVerifyInput", SimpleNamespace):
            backend.update_verify_buffers_to_fill_after_draft(spec_info, 1)
        self.assertEqual(
            backend.retrieve_next_token_list[0][:, :2].tolist(),
            spec_info.retrieve_next_token.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
