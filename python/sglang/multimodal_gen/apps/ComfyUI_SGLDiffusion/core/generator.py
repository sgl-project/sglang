"""
Generator for SGLang Diffusion ComfyUI integration.
"""

import logging
import os

import psutil
from comfy import model_detection, model_management
from comfy.patcher_extension import WrappersMP
from comfy.utils import (
    calculate_parameters,
    load_torch_file,
    state_dict_prefix_replace,
    unet_to_diffusers,
)

logger = logging.getLogger(__name__)


class _HeaderTensor:
    """Shape-only stand-in so detect_unet_config does not load 66GB weights."""

    def __init__(self, shape):
        import torch

        self.shape = torch.Size(shape)

    def numel(self):
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return n

    nelement = numel


def _looks_like_gguf(path: str) -> bool:
    if not path:
        return False
    return path.lower().endswith(".gguf") or (
        ":" in path and path.count("/") == 1 and not os.path.isabs(path)
    )


def _h3_detect_companion(gguf_ref: str) -> str:
    """ComfyUI detect needs H3 safetensors keys; GGUF only overrides the DiT."""
    name = os.path.basename(gguf_ref).lower()
    prefer = []
    if "ref2va" in name:
        prefer.append("minimax_h3_ref2va_bf16.safetensors")
    prefer.extend(
        [
            "minimax_h3_fl2va_bf16.safetensors",
            "minimax_h3_ref2va_bf16.safetensors",
        ]
    )
    folders = [os.path.dirname(os.path.abspath(gguf_ref))] if os.path.dirname(gguf_ref) else []
    try:
        import folder_paths

        folders.extend(folder_paths.get_folder_paths("diffusion_models"))
    except Exception:
        pass
    seen: set[str] = set()
    for folder in folders:
        if not folder:
            continue
        for candidate in prefer:
            path = os.path.join(folder, candidate)
            if path in seen:
                continue
            seen.add(path)
            if os.path.isfile(path):
                return path
    raise ValueError(
        "GGUF transformer needs a MiniMax-H3 BF16 safetensors in "
        "models/diffusion_models for ComfyUI architecture detect "
        f"(looked for {prefer}). Keep --model-path / unet_name on the "
        "BF16 file and pass the GGUF via transformer_weights_path."
    )


def _load_state_dict_for_detection(model_path: str):
    if _looks_like_gguf(model_path):
        raise ValueError(
            f"Cannot detect architecture from GGUF {model_path!r}. "
            "Use a MiniMax-H3 BF16 safetensors for detect."
        )
    if model_path.endswith(".safetensors"):
        from safetensors import safe_open

        sd = {}
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                sd[key] = _HeaderTensor(handle.get_slice(key).get_shape())
        return sd
    return load_torch_file(model_path)


try:
    from sglang.multimodal_gen import DiffGenerator
except ImportError:
    logger.error(
        "Error: sglang.multimodal_gen is not installed. Please install it using 'pip install sglang[diffusion]'"
    )

from ..executors import (
    FluxExecutor,
    MiniMaxH3Executor,
    QwenImageEditExecutor,
    QwenImageExecutor,
    ZImageExecutor,
)
from .model_patcher import SGLDModelPatcher

_EXECUTOR_CLASSES = (
    FluxExecutor,
    ZImageExecutor,
    QwenImageExecutor,
    QwenImageEditExecutor,
    MiniMaxH3Executor,
)


class SGLDiffusionGenerator:
    """Generator for SGLang Diffusion models in ComfyUI."""

    def __init__(self):
        self.model_path = None
        self.generator = None
        self.executor = None
        self.last_options = None

        # Native pipelines, run under comfyui_mode as a DiT-only forward service.
        self.pipeline_class_dict = {}
        self.executor_class_dict = {}
        for executor_cls in _EXECUTOR_CLASSES:
            for model_type in executor_cls.adapter_cls.model_types:
                self.executor_class_dict[model_type] = executor_cls
                self.pipeline_class_dict[model_type] = (
                    executor_cls.adapter_cls.pipeline_class_name
                )

    def __del__(self):
        self.close_generator()

    def init_generator(
        self, model_path: str, pipeline_class_name: str, kwargs: dict = None
    ):
        """Initialize the diffusion generator."""
        if self.generator is not None:
            return self.generator
        if kwargs is None:
            kwargs = {}
        # Set comfyui_mode for ComfyUI integration
        kwargs = dict(kwargs)
        kwargs["comfyui_mode"] = True
        # ComfyUI already keeps CLIP/VAE in the parent process. Auto image
        # policy otherwise sets dit_cpu_offload=True and every sampler step
        # reloads the DiT from CPU.
        kwargs.setdefault("dit_cpu_offload", False)
        kwargs = self._server_args_kwargs(kwargs)
        self.generator = DiffGenerator.from_pretrained(
            model_path=model_path,
            pipeline_class_name=pipeline_class_name,
            **kwargs,
        )
        return self.generator

    @staticmethod
    def _server_args_kwargs(kwargs: dict) -> dict:
        """Drop plugin-only / stale flags that ServerArgs no longer accepts."""
        import dataclasses

        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        valid = {f.name for f in dataclasses.fields(ServerArgs)}
        aliases = {"dp_degree": "dp_size", "cache_strategy": None, "model_type": None}
        cleaned = {}
        for key, value in kwargs.items():
            dest = aliases.get(key, key)
            if dest is None or dest not in valid:
                continue
            cleaned[dest] = value
        return cleaned

    def kill_generator(self):
        """Kill worker processes manually because generator shutdown cannot terminate them."""
        current_pid = os.getpid()
        worker_processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                # Look for sglang-diffusionWorker processes
                if proc.info["cmdline"]:
                    cmdline = " ".join(proc.info["cmdline"])
                    if "sgl_diffusion::" in cmdline:
                        if proc.info["pid"] != current_pid:
                            worker_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if worker_processes:
            logger.info(
                f"Found {len(worker_processes)} worker processes to terminate..."
            )
            for proc in worker_processes:
                try:
                    logger.info(
                        f"Terminating worker process {proc.info['pid']}: {proc.info['name']}"
                    )
                    proc.terminate()
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    logger.warning(
                        f"Process {proc.info['pid']} did not terminate, forcing kill..."
                    )
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

    def close_generator(self):
        """Close and cleanup the generator and all associated resources."""
        if self.generator is not None:
            self.generator.shutdown()
            self.kill_generator()
            # Clear other references
            self.last_options = None
            self.model_path = None
            self.generator = None
            self.executor = None
            self._patcher = None

    def get_comfyui_model(self, model_path: str, model_options: dict = None):
        """Get ComfyUI model from model path."""
        if model_options is None:
            model_options = {}
        dtype = model_options.get("dtype", None)
        # Header-only read: H3 BF16 is ~66GB and we only need keys + shapes
        # for detect_unet_config. SGLD loads the real weights later.
        sd = _load_state_dict_for_detection(model_path)
        diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
        temp_sd = state_dict_prefix_replace(
            sd, {diffusion_model_prefix: ""}, filter_keys=True
        )
        if len(temp_sd) > 0:
            sd = temp_sd

        parameters = calculate_parameters(sd)
        load_device = model_management.get_torch_device()

        model_detect_config = model_detection.detect_unet_config(sd, "")
        model_type = model_detect_config.get("image_model", None)
        if model_type is None or model_type not in self.pipeline_class_dict:
            raise ValueError(f"Unsupported model type: {model_type}")
        model_config = model_detection.model_config_from_unet(sd, "")

        if model_config is not None:
            new_sd = sd
        else:
            new_sd = model_detection.convert_diffusers_mmdit(sd, "")
            if new_sd is not None:  # diffusers mmdit
                model_config = model_detection.model_config_from_unet(new_sd, "")
                if model_config is None:
                    return None
            else:  # diffusers unet
                model_config = model_detection.model_config_from_diffusers_unet(sd)
                if model_config is None:
                    return None

                diffusers_keys = unet_to_diffusers(model_config.unet_config)
                new_sd = {}
                for k in diffusers_keys:
                    if k in sd:
                        new_sd[diffusers_keys[k]] = sd.pop(k)
        if dtype is None:
            unet_dtype = model_management.unet_dtype(
                model_params=parameters,
                supported_dtypes=model_config.supported_inference_dtypes,
            )
        else:
            unet_dtype = dtype

        manual_cast_dtype = model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes
        )
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
        model_config.custom_operations = model_options.get("custom_operations", None)
        model_config.unet_config["disable_unet_model_creation"] = True
        comfyui_model = model_config.get_model({})
        return comfyui_model, model_config, model_type

    def load_model(
        self, model_path: str, model_options: dict = None, sgld_options: dict = None
    ):
        """Load model and return model patcher."""
        sgld_options = dict(sgld_options) if sgld_options else {}
        model_options = dict(model_options) if model_options else {}
        plugin_flags = {}
        if "enable_cache_dit" in sgld_options:
            plugin_flags["enable_cache_dit"] = sgld_options.pop("enable_cache_dit")
        set_model_type = sgld_options.pop("model_type", None)
        override = (sgld_options.get("transformer_weights_path") or "").strip()
        if override:
            sgld_options["transformer_weights_path"] = override
        elif _looks_like_gguf(model_path):
            sgld_options["transformer_weights_path"] = model_path
        else:
            sgld_options.pop("transformer_weights_path", None)
        detect_path = model_path
        if _looks_like_gguf(model_path):
            detect_path = _h3_detect_companion(
                sgld_options["transformer_weights_path"]
            )
        gather_options = {
            "model_path": detect_path,
            "model_options": model_options,
            "sgld_options": sgld_options,
            "set_model_type": set_model_type,
        }
        if (
            self.last_options is not None
            and self.last_options == gather_options
            and self.generator is not None
            and getattr(self, "_patcher", None) is not None
            and self.executor is not None
        ):
            self.executor.enable_cache_dit = plugin_flags.get("enable_cache_dit")
            return self._patcher

        self.close_generator()
        self.last_options = gather_options
        self.model_path = detect_path

        comfyui_model, model_config, model_type = self.get_comfyui_model(
            detect_path, model_options
        )
        if model_type is None or model_type not in self.pipeline_class_dict:
            raise ValueError(f"Unsupported model type: {model_type}")
        if set_model_type is not None and set_model_type in self.pipeline_class_dict:
            model_type = set_model_type

        pipeline_class_name = self.pipeline_class_dict[model_type]
        self.generator = self.init_generator(
            detect_path, pipeline_class_name, sgld_options
        )

        executor_class = self.executor_class_dict[model_type]
        self.executor = executor_class(
            self.generator, detect_path, comfyui_model, model_config
        )
        self.executor.enable_cache_dit = plugin_flags.get("enable_cache_dit")
        comfyui_model.diffusion_model = self.executor

        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        self._patcher = SGLDModelPatcher(
            comfyui_model, load_device, offload_device, model_type=model_type
        )
        self._patcher.add_wrapper_with_key(
            WrappersMP.SAMPLER_SAMPLE,
            "sgld_session",
            self.executor.sampler_sample_wrapper,
        )
        return self._patcher
