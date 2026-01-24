"""
Generator for SGLang Diffusion ComfyUI integration.
"""

import logging
import os

import psutil
from comfy import model_detection, model_management
from comfy.utils import (
    calculate_parameters,
    load_torch_file,
    state_dict_prefix_replace,
    unet_to_diffusers,
)

logger = logging.getLogger(__name__)

try:
    from sglang.multimodal_gen import DiffGenerator
except ImportError:
    logger.error(
        "Error: sglang.multimodal_gen is not installed. Please install it using 'pip install sglang[diffusion]'"
    )

from ..executors import FluxExecutor, ZImageExecutor
from .model_patcher import SGLDModelPatcher


class SGLDiffusionGenerator:
    """Generator for SGLang Diffusion models in ComfyUI."""

    def __init__(self):
        self.model_path = None
        self.generator = None
        self.executor = None
        self.last_options = None

        self.pipeline_class_dict = {
            "flux": "ComfyUIFluxPipeline",
            "lumina2": "ComfyUIZImagePipeline",  # zimage
        }
        self.executor_class_dict = {
            "flux": FluxExecutor,
            "lumina2": ZImageExecutor,
        }

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
        kwargs["comfyui_mode"] = True
        self.generator = DiffGenerator.from_pretrained(
            model_path=model_path,
            pipeline_class_name=pipeline_class_name,
            **kwargs,
        )
        return self.generator

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

    def get_comfyui_model(self, model_path: str, model_options: dict = None):
        """Get ComfyUI model from model path."""
        if model_options is None:
            model_options = {}
        dtype = model_options.get("dtype", None)
        # Allow loading unets from checkpoint files
        sd = load_torch_file(model_path)
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
        offload_device = model_management.unet_offload_device()
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
        gather_options = {
            "model_path": model_path,
            "model_options": model_options,
            "sgld_options": sgld_options,
        }
        if (
            self.last_options is not None
            and self.last_options == gather_options
            and self.generator is not None
        ):
            return self.generator
        else:
            self.close_generator()

        self.last_options = gather_options
        self.model_path = model_path

        comfyui_model, model_config, model_type = self.get_comfyui_model(
            model_path, model_options
        )
        if model_type is None or model_type not in self.pipeline_class_dict:
            raise ValueError(f"Unsupported model type: {model_type}")

        pipeline_class_name = self.pipeline_class_dict[model_type]
        self.generator = self.init_generator(
            model_path, pipeline_class_name, sgld_options
        )

        executor_class = self.executor_class_dict[model_type]
        self.executor = executor_class(
            self.generator, model_path, comfyui_model, model_config
        )
        comfyui_model.diffusion_model = self.executor

        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        return SGLDModelPatcher(
            comfyui_model, load_device, offload_device, model_type=model_type
        )
