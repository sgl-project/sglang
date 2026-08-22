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
        if self.adapter_cls is None:
            raise TypeError(f"{type(self).__name__} must set adapter_cls")
        self.adapter = self.adapter_cls()
        self.session_id = uuid.uuid4().hex
        self._run_id = 0
        self._conditioning_sent = False
        self._last_spatial = None

    @staticmethod
    def should_suppress_logs(timestep):
        """Determine if logs should be suppressed based on timestep value."""
        if torch.is_tensor(timestep):
            return bool((timestep < 1.0).item())
        return bool(timestep < 1.0)

    def set_lora(self, lora_nickname=None, lora_path=None, strength=None, target=None):
        """Set LoRA adapter using SGLang Diffusion API."""
        if len(lora_nickname) > 0:
            self.generator.set_lora(
                lora_nickname=lora_nickname,
                lora_path=lora_path,
                strength=strength,
                target=target,
            )

    def _begin_run_if_needed(self, timestep, kwargs, spatial=None) -> None:
        opts = kwargs.get("transformer_options") or {}
        sigmas = opts.get("sample_sigmas")
        if sigmas is None:
            is_first = not self._conditioning_sent
        else:
            if torch.is_tensor(timestep):
                sigma = float(timestep.reshape(-1)[0].item())
            else:
                sigma = float(timestep)
            if sigma > 2.0:
                sigma = sigma / 1000.0
            first = float(sigmas.reshape(-1)[0].item() if torch.is_tensor(sigmas) else sigmas[0])
            is_first = abs(sigma - first) < 1e-4
        size_changed = spatial is not None and spatial != getattr(
            self, "_last_spatial", None
        )
        if is_first or size_changed:
            self._run_id += 1
            self._conditioning_sent = False
        if spatial is not None:
            self._last_spatial = spatial

    def forward(self, x, timestep, context, **kwargs):
        self._begin_run_if_needed(timestep, kwargs)
        packed = self.adapter.pack(x, timestep, context, **kwargs)
        if self._conditioning_sent:
            self.adapter.drop_cached_fields(packed)
        else:
            self._conditioning_sent = True
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            prompt=" ",
            guidance_scale=packed.guidance_scale,
            height=packed.height,
            width=packed.width,
            num_frames=1,
            num_inference_steps=1,
            save_output=False,
            suppress_logs=self.should_suppress_logs(timestep),
        )
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        self.adapter.fill_req(req, packed)
        extra = dict(req.extra or {})
        extra["comfyui_session_id"] = f"{self.session_id}:{self._run_id}"
        req.extra = extra
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]
        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        return self.adapter.unpack(output_batch.noise_pred, packed, x)
