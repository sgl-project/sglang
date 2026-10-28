"""The host-sync guards, and the shared hot paths they keep off the host.

Each shared-path test runs the same steps twice, once with plain tensors and
once with the device tensors as ``SyncOnRead``, and requires bitwise-equal
results. Model tests follow the same pattern with their own stage and model.
"""

import unittest
import warnings
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import torch

from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.layers.attention.backends import flash_attn
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.layers.rotary_embedding import utils as rope_utils
from sglang.multimodal_gen.runtime.layers.rotary_embedding.mrope import FluxPosEmbed
from sglang.multimodal_gen.runtime.models.schedulers.kandinsky6_piflow import (
    PiflowScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_helios import (
    HeliosScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import denoising_dmd
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import (
    SyncOnRead,
    assert_no_cuda_sync,
    forbid_host_syncs,
    no_cuda_sync,
)


def _on_device(tensor):
    return tensor.as_subclass(SyncOnRead)


def _plain(tensors):
    return [tensor.as_subclass(torch.Tensor) for tensor in tensors]


# --- the guards ---


def _read_device_value():
    # what torch's sync debug mode emits for an implicit sync
    warnings.warn("called a synchronizing CUDA operation")


def _synchronize_explicitly():
    torch.cuda.synchronize()


def _calls_a_sync():
    _read_device_value()


class TestSyncOnRead(unittest.TestCase):
    def test_host_reads_raise(self):
        values = torch.tensor([1.0, 0.0]).as_subclass(SyncOnRead)
        reads = {
            "item": lambda: values[0].item(),
            "float": lambda: float(values[0]),
            "bool": lambda: bool(values[0] > 0.5),
            "index": lambda: [0, 1][values.long()[0]],
            "tolist": values.tolist,
            "nonzero": values.nonzero,
            "boolean mask": lambda: values[values > 0],
        }
        for name, read in reads.items():
            with self.subTest(read=name), self.assertRaises(AssertionError):
                read()

    def test_device_math_and_host_copies_do_not_raise(self):
        values = torch.tensor([1.0, 0.5]).as_subclass(SyncOnRead)
        delta = values[1] - values[0]
        self.assertIsInstance(delta, SyncOnRead)
        self.assertEqual(values.cpu()[1].item(), 0.5)
        self.assertIsInstance(values.to(values.device), SyncOnRead)


class TestForbidHostSyncs(unittest.TestCase):
    def test_syncing_ops_raise(self):
        values = torch.tensor([1.0, 0.0, 2.0])
        mask = values > 0.5
        syncs = {
            "item": lambda: values[0].item(),
            "bool": lambda: bool(values[0] > 0.5),
            "nonzero": values.nonzero,
            "masked_select": lambda: values.masked_select(mask),
            "boolean index": lambda: values[mask],
            "boolean index_put": lambda: values.clone().index_put_((mask,), values[:1]),
            "repeat_interleave": lambda: torch.repeat_interleave(values.long()),
            "full from a tensor": lambda: torch.full((2,), values[0]),
        }
        for name, sync in syncs.items():
            with self.subTest(op=name), self.assertRaises(AssertionError):
                with forbid_host_syncs():
                    sync()

    def test_device_side_ops_pass(self):
        values = torch.tensor([1.0, 0.0, 2.0])
        mask = values > 0.5
        with forbid_host_syncs():
            selected = torch.where(mask, values * 0.0, values)
            gathered = values.index_select(0, torch.tensor([2, 0]))
            repeated = torch.repeat_interleave(
                values, torch.tensor([1, 0, 2]), output_size=3
            )
        torch.testing.assert_close(
            (selected, gathered, repeated),
            (torch.zeros(3), torch.tensor([2.0, 1.0]), torch.tensor([1.0, 2.0, 2.0])),
            rtol=0,
            atol=0,
        )


class TestNoCudaSync(unittest.TestCase):
    def _fake_cuda(self):
        modes = []
        synchronize = Mock()
        patcher = patch.multiple(
            torch.cuda,
            is_available=Mock(return_value=True),
            get_sync_debug_mode=Mock(return_value=0),
            set_sync_debug_mode=Mock(side_effect=modes.append),
            synchronize=synchronize,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return modes, synchronize

    def test_reports_each_sync_with_its_call_site(self):
        modes, synchronize = self._fake_cuda()
        with self.assertRaises(AssertionError) as raised:
            with no_cuda_sync():
                _read_device_value()
                _synchronize_explicitly()
        message = str(raised.exception)
        self.assertIn("2 host-device sync(s)", message)
        self.assertIn("_read_device_value", message)
        self.assertIn("_synchronize_explicitly", message)
        # the explicit synchronize still ran, and the debug mode was restored
        synchronize.assert_called_once()
        self.assertEqual(modes, ["warn", 0])

    def test_allow_names_intentional_syncs(self):
        self._fake_cuda()
        allow = (
            "test_host_sync.py:_read_device_value",
            "test_host_sync.py:_synchronize_explicitly",
        )
        with no_cuda_sync(allow=allow):
            _read_device_value()
            _synchronize_explicitly()
        self.assertEqual(assert_no_cuda_sync(lambda: 3, allow=allow), 3)

    def test_allow_does_not_cover_callees(self):
        self._fake_cuda()
        with self.assertRaises(AssertionError):
            with no_cuda_sync(allow=("test_host_sync.py:_calls_a_sync",)):
                _calls_a_sync()

    def test_other_warnings_pass_through(self):
        self._fake_cuda()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with no_cuda_sync():
                warnings.warn("unrelated")
        self.assertEqual([str(w.message) for w in caught], ["unrelated"])

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_reports_item_and_pageable_upload(self):
        device_value = torch.ones(2, device="cuda")
        with no_cuda_sync():
            device_value = device_value * 2 + 1
        for sync in (
            lambda: device_value.sum().item(),
            lambda: torch.tensor([1.0, 2.0], device="cuda"),
            lambda: device_value.nonzero(),
        ):
            with self.assertRaises(AssertionError), no_cuda_sync():
                sync()


# --- shared denoising stages, with a tiny DiT and the flow-match Euler scheduler ---


class _PipelineConfig:
    task_type = ModelTaskType.T2I

    def get_classifier_free_guidance_scale(self, batch, guidance_scale):
        return guidance_scale

    def slice_noise_pred(self, noise_pred, latents):
        return noise_pred

    def postprocess_cfg_noise(self, batch, noise_pred, noise_pred_cond):
        return noise_pred


class _TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        hidden = self.proj(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return hidden + encoder_hidden_states.mean() + timestep.view(-1, 1, 1, 1) / 1000


def _run_request(device, wrap_device_tensor=lambda t: t, guard=nullcontext()):
    """Denoise one two-branch CFG request through ``_run_denoising_loop``."""
    torch.manual_seed(0)
    transformer = _TinyDiT().to(device)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(4, device=device)
    scheduler.sigmas = wrap_device_tensor(scheduler.sigmas)
    server_args = SimpleNamespace(
        comfyui_mode=False,
        disable_autocast=True,
        enable_cfg_parallel=False,
        enable_breakable_cuda_graph=False,
        enable_layerwise_nvtx_marker=False,
        pipeline_config=_PipelineConfig(),
    )
    stage = DenoisingStage.__new__(DenoisingStage)
    stage.server_args = server_args
    stage.transformer = transformer
    stage.attn_backend = SDPABackend
    stage._extra_func_kwarg_names_cache = {}
    stage._manage_dit_use_site = Mock()
    stage.progress_bar = MagicMock()
    batch = SimpleNamespace(
        raw_latent_shape=(1, 4, 2, 2),
        image_latent=None,
        condition_image=None,
        do_classifier_free_guidance=True,
        guidance_scale=4.0,
        guidance_scale_2=None,
        cfg_normalization=0,
        guidance_rescale=0.0,
        is_cfg_negative=False,
        return_trajectory_latents=False,
        sampling_params=SimpleNamespace(cfg_gate_step=None),
        extra={},
        metrics=None,
        perf_dump_path=None,
        is_warmup=False,
    )
    prompt = torch.randn(1, 3, 4, device=device)
    ctx = DenoisingContext(
        scheduler=scheduler,
        extra_step_kwargs={},
        target_dtype=torch.float32,
        autocast_enabled=False,
        timesteps=wrap_device_tensor(scheduler.timesteps),
        num_inference_steps=4,
        num_warmup_steps=0,
        image_kwargs={},
        pos_cond_kwargs={"encoder_hidden_states": prompt},
        neg_cond_kwargs={"encoder_hidden_states": -prompt},
        latents=wrap_device_tensor(torch.randn(1, 4, 2, 2, device=device)),
        boundary_timestep=None,
        z=None,
        reserved_frames_mask=None,
        seq_len=None,
        guidance=None,
        is_warmup=False,
        cfg_policy=CFGPolicy().build(
            batch,
            {},
            {"encoder_hidden_states": prompt},
            {"encoder_hidden_states": -prompt},
        ),
    )
    with torch.no_grad(), guard:
        stage._run_denoising_loop(ctx, batch, server_args)
    return ctx.latents.as_subclass(torch.Tensor)


class _TinyVideoDiT(_TinyDiT):
    def forward(self, hidden_states, timestep, guidance=None):
        hidden = self.proj(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return hidden + timestep.view(-1, 1, 1, 1, 1) / 1000


def _run_dmd_request(device_tensor=lambda t: t, nan_latents=False):
    """One DMD request; device timesteps go through ``device_tensor``."""
    torch.manual_seed(0)
    stage = denoising_dmd.DmdDenoisingStage.__new__(denoising_dmd.DmdDenoisingStage)
    stage.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(should_use_guidance=False)
    )
    stage.scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    stage.transformer = _TinyVideoDiT()
    stage.transformer_2 = None
    stage.attn_backend = SDPABackend
    stage._extra_func_kwarg_names_cache = {}
    stage._manage_dit_use_site = Mock()
    stage.progress_bar = MagicMock()
    stage.log_info = Mock()
    final = {}
    stage._post_denoising_loop = lambda latents, **kwargs: final.update(latents=latents)
    latents = torch.randn(1, 2, 4, 2, 2)
    if nan_latents:
        latents[0, 1, 0, 0, 0] = float("nan")
    stage._prepare_denoising_loop = lambda batch, args: SimpleNamespace(
        target_dtype=torch.float32,
        autocast_enabled=False,
        num_warmup_steps=0,
        latents=latents,
        pos_cond_kwargs={},
    )
    batch = SimpleNamespace(
        image_embeds=[],
        image_latent=None,
        generator=[torch.Generator().manual_seed(1)],
        metrics=None,
        perf_dump_path=None,
        is_warmup=False,
        is_cfg_negative=False,
    )
    server_args = SimpleNamespace(
        disable_autocast=True,
        pipeline_config=SimpleNamespace(dmd_denoising_steps=[1000, 757, 522]),
    )
    tensor = torch.tensor

    def tensor_on_device(data, **kwargs):
        made = tensor(data, **kwargs)
        return device_tensor(made) if "device" in kwargs else made

    cpu = torch.device("cpu")
    with (
        torch.no_grad(),
        patch.object(denoising_dmd, "get_local_torch_device", return_value=cpu),
        patch.object(
            denoising_dmd.DmdDenoisingStage,
            "device",
            new_callable=PropertyMock,
            return_value=cpu,
        ),
        patch.object(torch, "tensor", tensor_on_device),
    ):
        stage.forward(batch, server_args)
    return final["latents"].as_subclass(torch.Tensor)


class TestDenoisingLoopHostSync(unittest.TestCase):
    def test_steps_never_read_device_schedule_on_host(self):
        expected = _run_request("cpu")
        actual = _run_request("cpu", wrap_device_tensor=_on_device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dmd_steps_never_read_device_timesteps(self):
        expected = _run_dmd_request()
        actual = _run_dmd_request(device_tensor=_on_device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # the NaN guard still fires, read once after the loop
        with self.assertRaisesRegex(AssertionError, "contains nan"):
            _run_dmd_request(nan_latents=True)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_loop_syncs_only_for_the_host_schedule(self):
        expected = _run_request("cuda")
        # the loop copies the schedule to the host once; steps must not sync
        guard = no_cuda_sync(allow=("stages/denoising.py:_run_denoising_loop",))
        actual = _run_request("cuda", guard=guard)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# --- shared schedulers: device schedule and latents as SyncOnRead ---

_full = torch.full


def _full_without_host_read(size, fill_value, **kwargs):
    if isinstance(fill_value, torch.Tensor):
        raise AssertionError("torch.full reads a tensor fill value on the host")
    return _full(size, fill_value, **kwargs)


class TestPiflowHostSync(unittest.TestCase):
    def _run(self, on_device):
        scheduler = PiflowScheduler(n_grid=10)
        scheduler.set_timesteps(4)
        scheduler.set_begin_index(0)
        generator = torch.Generator().manual_seed(0)
        sample = torch.randn(2, 5, 4, generator=generator)
        outputs = [torch.randn(2, 5, 40, generator=generator) for _ in range(4)]
        timesteps = scheduler.timesteps
        if on_device:
            scheduler._piflow_raw_timesteps = _on_device(
                scheduler._piflow_raw_timesteps
            )
            scheduler.sigmas = _on_device(scheduler.sigmas)
            timesteps, sample = _on_device(timesteps), _on_device(sample)
        samples = []
        for timestep, output in zip(timesteps, outputs):
            sample = scheduler.step(output, timestep, sample, return_dict=False)[0]
            samples.append(sample)
        return _plain(samples)

    def test_policy_rollout_never_reads_device_substeps(self):
        torch.testing.assert_close(self._run(True), self._run(False), rtol=0, atol=0)


class TestHeliosHostSync(unittest.TestCase):
    def _run_unipc(self, on_device, dtype):
        scheduler = HeliosScheduler(shift=3.0, solver_order=2, scheduler_type="unipc")
        scheduler.set_timesteps(6, device="cpu")
        generator = torch.Generator().manual_seed(1)
        sample = torch.randn(1, 4, 2, 3, 3, generator=generator).to(dtype)
        outputs = [
            torch.randn(1, 4, 2, 3, 3, generator=generator).to(dtype) for _ in range(6)
        ]
        timesteps = scheduler.timesteps
        if on_device:
            scheduler.sigmas = _on_device(scheduler.sigmas)
            timesteps, sample = _on_device(timesteps), _on_device(sample)
        samples = []
        for timestep, output in zip(timesteps, outputs):
            sample = scheduler.step(output, timestep, sample, return_dict=False)[0]
            samples.append(sample)
        return _plain(samples)

    def _run_dmd(self, on_device):
        scheduler = HeliosScheduler(scheduler_type="dmd")
        scheduler.set_timesteps(3, device="cpu")
        generator = torch.Generator().manual_seed(2)
        sample = torch.randn(1, 4, 2, 3, 3, generator=generator)
        noisy = torch.randn(1, 4, 2, 3, 3, generator=generator)
        outputs = [torch.randn(1, 4, 2, 3, 3, generator=generator) for _ in range(3)]
        sigmas, timesteps = scheduler.sigmas, scheduler.timesteps
        if on_device:
            sigmas, timesteps = _on_device(sigmas), _on_device(timesteps)
            sample, noisy = _on_device(sample), _on_device(noisy)
        samples = []
        for index, (timestep, output) in enumerate(zip(timesteps, outputs)):
            sample = scheduler.step(
                output,
                timestep,
                sample,
                return_dict=False,
                cur_sampling_step=index,
                dmd_noisy_tensor=noisy,
                dmd_sigmas=sigmas,
                dmd_timesteps=timesteps,
                all_timesteps=timesteps,
            )[0]
            samples.append(sample)
        return _plain(samples)

    def test_unipc_never_reads_device_coefficients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.testing.assert_close(
                    self._run_unipc(True, dtype),
                    self._run_unipc(False, dtype),
                    rtol=0,
                    atol=0,
                )

    def test_dmd_never_reads_device_timesteps(self):
        expected = self._run_dmd(False)
        with patch.object(torch, "full", _full_without_host_read):
            actual = self._run_dmd(True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# --- shared RoPE ---


def _grid_ids(height: int, width: int) -> torch.Tensor:
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = torch.arange(height)[:, None]
    ids[..., 2] = torch.arange(width)[None, :]
    return ids.reshape(-1, 3)


class TestNDRopeHostSync(unittest.TestCase):
    def test_never_reads_positions_on_host(self):
        embed = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])
        ids = _grid_ids(6, 8)
        expected = embed(ids)
        actual = _plain(embed(_on_device(ids)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestRopePositions(unittest.TestCase):
    def setUp(self):
        rope_utils._shared_rope_positions.cache_clear()
        self.addCleanup(rope_utils._shared_rope_positions.cache_clear)

    def test_shared_across_blocks(self):
        device = torch.device("cpu")
        with patch.object(
            rope_utils,
            "_build_rope_positions",
            wraps=rope_utils._build_rope_positions,
        ) as build:
            first = rope_utils.rope_positions(3, 5, 2, device)
            second = rope_utils.rope_positions(3, 5, 2, device)
        self.assertIs(first, second)
        build.assert_called_once()
        torch.testing.assert_close(first, torch.arange(3, 8).repeat(2), rtol=0, atol=0)

    def test_compile_and_capture_build_their_own_positions(self):
        build = Mock(side_effect=lambda *args: object())
        modes = (
            (torch.compiler, "is_compiling", torch.device("cpu")),
            (torch.cuda, "is_current_stream_capturing", torch.device("cuda")),
        )
        for owner, name, device in modes:
            with self.subTest(mode=name):
                with (
                    patch.object(rope_utils, "_build_rope_positions", build),
                    patch.object(owner, name, return_value=True),
                ):
                    first = rope_utils.rope_positions(0, 4, 1, device)
                    second = rope_utils.rope_positions(0, 4, 1, device)
                self.assertIsNot(first, second)
        self.assertEqual(rope_utils._shared_rope_positions.cache_info().currsize, 0)

    def test_qk_norm_rope_takes_shared_positions(self):
        torch.manual_seed(0)
        q = torch.randn(2, 6, 2, 8)
        k = torch.randn(2, 6, 2, 8)
        cos_sin_cache = torch.randn(16, 8)
        q_norm = layernorm.RMSNorm(8)
        k_norm = layernorm.RMSNorm(8)
        expected = layernorm.apply_qk_norm_rope(
            q.clone(),
            k.clone(),
            q_norm,
            k_norm,
            8,
            cos_sin_cache,
            positions=torch.arange(1, 7).repeat(2),
        )
        with patch.object(
            layernorm, "rope_positions", wraps=rope_utils.rope_positions
        ) as positions:
            actual = layernorm.apply_qk_norm_rope(
                q.clone(),
                k.clone(),
                q_norm,
                k_norm,
                8,
                cos_sin_cache,
                position_offset=1,
            )
        positions.assert_called_once_with(1, 6, 2, q.device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# --- shared attention ---


class TestRingKVChunkHostSync(unittest.TestCase):
    def test_ring_hops_reuse_device_cu_seqlens(self):
        # every ring hop of every block used to upload [0, n] with a pageable,
        # host-blocking copy; the bounds are fixed per shape, so they upload once
        impl = flash_attn.FlashAttentionImpl(
            num_heads=2, head_size=8, causal=False, softmax_scale=0.5
        )
        query = torch.randn(6, 2, 8)
        key = torch.randn(4, 2, 8)
        calls = []

        def varlen(q, k, v, **kwargs):
            calls.append(kwargs)
            return q, torch.zeros(2, q.shape[0])

        with patch.object(flash_attn, "flash_attn_varlen_func", varlen):
            for _ in range(2):
                impl.forward_ring_kv_chunk(query, key, key)
        first, second = calls
        self.assertEqual(first["cu_seqlens_q"].tolist(), [0, 6])
        self.assertEqual(first["cu_seqlens_k"].tolist(), [0, 4])
        self.assertEqual(first["cu_seqlens_q"].dtype, torch.int32)
        self.assertIs(first["cu_seqlens_q"], second["cu_seqlens_q"])
        self.assertIs(first["cu_seqlens_k"], second["cu_seqlens_k"])


if __name__ == "__main__":
    unittest.main()
