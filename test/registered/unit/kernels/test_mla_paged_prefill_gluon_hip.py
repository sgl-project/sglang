import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention import mla_paged_prefill_gluon_hip as adapter


def make_inputs(rows=1024, requests=2):
    device = torch.device("meta")
    cache = torch.empty((655361, 1, 576), dtype=torch.float8_e4m3fn, device=device)
    lengths = [rows // requests] * requests
    lengths[-1] += rows - sum(lengths)
    return dict(
        q=torch.empty((rows, 12, 576), dtype=torch.bfloat16, device=device),
        k=torch.empty((rows, 1, 576), dtype=torch.bfloat16, device=device),
        v=torch.empty((rows, 1, 512), dtype=torch.bfloat16, device=device),
        out=torch.empty((rows, 12, 512), dtype=torch.bfloat16, device=device),
        k_buffer=cache,
        v_buffer=cache[..., :512],
        qo_indptr=torch.empty((requests + 1,), dtype=torch.int32, device=device),
        kv_indptr=torch.empty((requests + 1,), dtype=torch.int32, device=device),
        kv_indices=torch.empty((512,), dtype=torch.int64, device=device),
        lengths=lengths,
    )


def call_extend(extend, inputs, **overrides):
    kwargs = dict(
        logit_cap=0.0,
        sliding_window_size=-1,
        xai_temperature_len=-1,
        page_size=1,
        extend_seq_lens_cpu=inputs["lengths"],
    )
    kwargs.update(overrides)
    return extend(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["out"],
        inputs["k_buffer"],
        inputs["v_buffer"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        None,
        True,
        None,
        max(inputs["lengths"]),
        1.0,
        1.0,
        192**-0.5,
        **kwargs,
    )


def covered(inputs, **overrides):
    kwargs = dict(
        custom_mask=None,
        is_causal=True,
        mask_indptr=None,
        max_query_length=max(inputs["lengths"]),
        k_scale=1.0,
        v_scale=1.0,
        sm_scale=192**-0.5,
        logit_cap=0.0,
        sliding_window_size=-1,
        sinks=None,
        window_kv_offsets=None,
        xai_temperature_len=-1,
        lse=None,
        skip_prefix=False,
        skip_extend=False,
        page_size=1,
        score_mod=None,
        aux_tensors=None,
        lengths=inputs["lengths"],
        identity_kv_indices=False,
    )
    kwargs.update(overrides)
    backend = SimpleNamespace(max_context_len=1048576)
    with (
        mock.patch("torch.compiler.is_compiling", return_value=False),
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        return adapter.covered(
            backend,
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["out"],
            inputs["k_buffer"],
            inputs["v_buffer"],
            inputs["qo_indptr"],
            inputs["kv_indptr"],
            inputs["kv_indices"],
            **kwargs,
        )


class TestMlaPagedPrefill(unittest.TestCase):
    def test_runtime_contract_is_fail_closed(self):
        inputs = make_inputs()
        self.assertTrue(covered(inputs))
        self.assertFalse(covered(inputs, identity_kv_indices=True))
        self.assertFalse(covered(inputs, lse=inputs["qo_indptr"]))
        self.assertFalse(covered(inputs, lengths=[1024]))
        self.assertFalse(covered(make_inputs(512)))
        self.assertFalse(covered(make_inputs(1024, 9)))

    def test_install_preserves_fallback_and_propagates_launch_failure(self):
        native = mock.Mock(return_value="native")
        backend = SimpleNamespace(extend_attention_fwd=native, max_context_len=1048576)
        with (
            mock.patch.object(adapter, "can_install", return_value=True),
            mock.patch.object(adapter, "rank0_log"),
        ):
            self.assertTrue(adapter.install(backend, object()))

        unsupported = make_inputs(512)
        with (
            mock.patch("torch.compiler.is_compiling", return_value=False),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            self.assertEqual(
                call_extend(backend.extend_attention_fwd, unsupported), "native"
            )
        native.assert_called_once()

        supported = make_inputs()
        with (
            mock.patch("torch.compiler.is_compiling", return_value=False),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
            mock.patch.object(adapter, "run", side_effect=RuntimeError("launch")),
            self.assertRaisesRegex(RuntimeError, "launch"),
        ):
            call_extend(backend.extend_attention_fwd, supported)
        self.assertEqual(native.call_count, 1)

    def test_installed_kernel_must_fill_native_output(self):
        native = mock.Mock()
        backend = SimpleNamespace(extend_attention_fwd=native, max_context_len=1048576)
        with (
            mock.patch.object(adapter, "can_install", return_value=True),
            mock.patch.object(adapter, "rank0_log"),
        ):
            adapter.install(backend, object())
        inputs = make_inputs()
        with (
            mock.patch("torch.compiler.is_compiling", return_value=False),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
            mock.patch.object(adapter, "run", return_value=inputs["out"]),
        ):
            self.assertIsNone(call_extend(backend.extend_attention_fwd, inputs))
        native.assert_not_called()

        wrong = torch.empty((1024, 12, 512), dtype=torch.float32, device="meta")
        with (
            mock.patch("torch.compiler.is_compiling", return_value=False),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
            mock.patch.object(adapter, "run", return_value=wrong),
            self.assertRaisesRegex(RuntimeError, "output ABI"),
        ):
            call_extend(backend.extend_attention_fwd, inputs)


if __name__ == "__main__":
    unittest.main()
