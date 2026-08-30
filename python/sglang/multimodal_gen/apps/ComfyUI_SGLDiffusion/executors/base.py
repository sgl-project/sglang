"""
Base executor class for SGLang Diffusion ComfyUI integration.
"""

import uuid

import torch

try:
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
except ImportError:
    print(
        "Error: sglang.multimodal_gen is not installed. Please install it using 'pip install sglang[diffusion]'"
    )


class SGLDiffusionExecutor(torch.nn.Module):
    """Shared ComfyUI DiT-forward executor. Per-model logic lives on the adapter."""

    adapter_cls = None

    def __init__(self, generator, model_path, model, config):
        super(SGLDiffusionExecutor, self).__init__()
        self.generator = generator
        self.model_path = model_path
        self.model = model
        self.dtype = config.unet_config["dtype"]
        self.config = config
        self.loras = []
        self._lora_input = None
        self._sgld_reload = None
        self._ensure_runtime = None
        if self.adapter_cls is None:
            raise TypeError(f"{type(self).__name__} must set adapter_cls")
        self.adapter = self.adapter_cls()
        self.session_id = uuid.uuid4().hex
        self._run_id = 0
        self._sent_conds: set[tuple] = set()

    @staticmethod
    def should_suppress_logs(timestep):
        """Determine if logs should be suppressed based on timestep value."""
        if torch.is_tensor(timestep):
            return bool((timestep < 1.0).item())
        return bool(timestep < 1.0)

    def set_lora(self, lora_nickname=None, lora_path=None, strength=None, target=None):
        """Set LoRA adapter using SGLang Diffusion API."""
        self._lora_input = {
            "lora_nickname": lora_nickname,
            "lora_path": lora_path,
            "strength": strength,
            "target": target,
        }
        if lora_nickname and len(lora_nickname) > 0:
            self.generator.set_lora(
                lora_nickname=lora_nickname,
                lora_path=lora_path,
                strength=strength,
                target=target,
            )

    def begin_sampler_run(self) -> None:
        """One ComfyUI ``sampler.sample()`` invocation is one cache lifetime."""
        self._run_id += 1
        self._sent_conds = set()

    def end_sampler_run(self) -> None:
        """Run cache is evicted on the next bind of a newer id for this executor."""

    def sampler_sample_wrapper(self, executor, *args, **kwargs):
        self.begin_sampler_run()
        try:
            return executor(*args, **kwargs)
        finally:
            self.end_sampler_run()

    def comfyui_session_id(self) -> str:
        return f"{self.session_id}:{self._run_id}"

    def _cond_key(self, packed) -> tuple | None:
        embeds = packed.prompt_embeds
        if not embeds:
            return None
        tensor = embeds[0]
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return None
        flat = tensor.reshape(-1)
        return (
            tuple(int(dim) for dim in tensor.shape),
            float(flat[0].item()),
            float(flat[-1].item()),
        )

    def _mark_and_maybe_drop(self, packed) -> None:
        key = self._cond_key(packed)
        if key is not None:
            packed.extra_req["comfyui_cond_key"] = repr(key)
            if key in self._sent_conds:
                self.adapter.drop_cached_fields(packed)
            else:
                self._sent_conds.add(key)

    def _sampling_params_kwargs(self, packed, timestep) -> dict:
        return {
            "prompt": " ",
            "guidance_scale": packed.guidance_scale,
            "height": packed.height,
            "width": packed.width,
            "num_frames": 1,
            "num_inference_steps": 1,
            "save_output": False,
            "suppress_logs": self.should_suppress_logs(timestep),
        }

    def _execute_packed(self, packed, x, timestep):
        ensure = getattr(self, "_ensure_runtime", None)
        if ensure is not None:
            ensure(self)
        self._mark_and_maybe_drop(packed)
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            **self._sampling_params_kwargs(packed, timestep),
        )
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        self.adapter.fill_req(req, packed)
        extra = dict(req.extra or {})
        extra["comfyui_session_id"] = self.comfyui_session_id()
        for key in ("comfyui_cond_key", "comfyui_cache_fp"):
            value = packed.extra_req.get(key)
            if value is not None:
                extra[key] = value
        req.extra = extra
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]
        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        return self.adapter.unpack(output_batch.noise_pred, packed, x)

    def forward(self, x, timestep, context, **kwargs):
        packed = self.adapter.pack(x, timestep, context, **kwargs)
        return self._execute_packed(packed, x, timestep)
