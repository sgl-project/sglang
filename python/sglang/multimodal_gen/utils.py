# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py

import argparse
import ctypes
import importlib
import importlib.util
import inspect
import math
import os
import signal
import socket
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache, partial, wraps
from typing import Any, TypeVar, cast

import cloudpickle
import imageio
import numpy as np
import torch
import torchvision
import yaml
from einops import rearrange
from remote_pdb import RemotePdb
from torch.distributed.fsdp import MixedPrecisionPolicy

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    SortedHelpFormatter,
    init_logger,
)

logger = init_logger(__name__)

T = TypeVar("T")

# TODO(will): used to convert server_args.precision to torch.dtype. Find a
# cleaner way to do this.
PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

STR_BACKEND_ENV_VAR: str = "SGL_DIFFUSION_ATTENTION_BACKEND"
STR_ATTN_CONFIG_ENV_VAR: str = "SGL_DIFFUSION_ATTENTION_CONFIG"


def find_nccl_library() -> str:
    """
    We either use the library file specified by the `VLLM_NCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libnccl.so.2` or `librccl.so.1` can be
    found by `ctypes` automatically.
    """
    so_file = envs.SGL_DIFFUSION_NCCL_SO_PATH

    # manually load the nccl library
    if so_file:
        logger.info(
            "Found nccl from environment variable SGL_DIFFUSION_NCCL_SO_PATH=%s",
            so_file,
        )
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.info("Found nccl from library %s", so_file)
    return str(so_file)


prev_set_stream = torch.cuda.set_stream

_current_stream = None


def _patched_set_stream(stream: torch.cuda.Stream | None) -> None:
    global _current_stream
    _current_stream = stream
    if stream is not None:
        prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


def current_stream() -> torch.cuda.Stream | None:
    """
    replace `torch.cuda.current_stream()` with `sglang.multimodal_gen.utils.current_stream()`.
    it turns out that `torch.cuda.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.cuda.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.cuda.current_stream()`.

    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
    from C/C++ code.
    """
    from sglang.multimodal_gen.runtime.platforms import current_platform

    # For non-CUDA platforms, return None
    if not current_platform.is_cuda_alike():
        return None

    global _current_stream
    if _current_stream is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        # On ROCm using the default 0 stream in combination with RCCL
        # is hurting performance. Therefore creating a dedicated stream
        # per process
        _current_stream = (
            torch.cuda.Stream()
            if current_platform.is_rocm()
            else torch.cuda.current_stream()
        )
    return _current_stream


class StoreBoolean(argparse.Action):

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs="?",
            const=True,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, True)
        elif isinstance(values, str):
            if values.lower() == "true":
                setattr(namespace, self.dest, True)
            elif values.lower() == "false":
                setattr(namespace, self.dest, False)
            else:
                raise ValueError(
                    f"Invalid boolean value: {values}. " "Expected 'true' or 'false'."
                )
        else:
            setattr(namespace, self.dest, bool(values))


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def __init__(self, *args, **kwargs) -> None:
        # Set the default 'formatter_class' to SortedHelpFormatter
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = SortedHelpFormatter
        super().__init__(*args, **kwargs)

    def parse_args(  # type: ignore[override]
        self, args=None, namespace=None
    ) -> argparse.Namespace:
        if args is None:
            args = sys.argv[1:]

        if any(arg.startswith("--config") for arg in args):
            args = self._pull_args_from_config(args)

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = "--" + key[len("--") :].replace("_", "-")
                    processed_args.append(f"{key}={value}")
                else:
                    processed_args.append("--" + arg[len("--") :].replace("_", "-"))
            elif arg.startswith("-O") and arg != "-O" and len(arg) == 2:
                # allow -O flag to be used without space, e.g. -O3
                processed_args.append("-O")
                processed_args.append(arg[2:])
            else:
                processed_args.append(arg)

        namespace = super().parse_args(processed_args, namespace)

        # Track which arguments were explicitly provided
        namespace._provided = set()

        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                # Handle --key=value format
                if "=" in arg:
                    key = arg.split("=")[0][2:].replace("-", "_")
                    namespace._provided.add(key)
                    i += 1
                # Handle --key value format
                else:
                    key = arg[2:].replace("-", "_")
                    namespace._provided.add(key)
                    # Skip the value if there is one
                    if i + 1 < len(args) and not args[i + 1].startswith("-"):
                        i += 2
                    else:
                        i += 1
            else:
                i += 1

        return namespace  # type: ignore[no-any-return]

    def _pull_args_from_config(self, args: list[str]) -> list[str]:
        """Method to pull arguments specified in the config file
        into the command-line args variable.

        The arguments in config file will be inserted between
        the argument list.

        example:
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        ```python
        $: vllm {serve,chat,complete} "facebook/opt-12B" \
            --config config.yaml -tp 2
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--config', 'config.yaml',
            '-tp', '2'
        ]
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--port', '12323',
            '--tp-size', '4',
            '-tp', '2'
            ]
        ```

        Please note how the config args are inserted after the sub command.
        this way the order of priorities is maintained when these are args
        parsed by super().
        """
        index = -1
        config_arg = None
        for i, arg in enumerate(args):
            if arg.startswith("--config"):
                if index != -1:
                    raise ValueError("More than one config file specified!")
                index = i
                config_arg = arg

        if config_arg is None:
            return args
        args_before_config = args[:index]
        if "=" in config_arg:
            file_path = config_arg.split("=", 1)[1]
            args_after_config = args[index + 1 :]
        else:
            if index == len(args) - 1:
                raise ValueError(
                    "No config file specified! "
                    "Please check your command-line arguments."
                )
            file_path = args[index + 1]
            args_after_config = args[index + 2 :]

        config_args = self._load_config_file(file_path)

        # 0th index is for {serve,chat,complete}
        # followed by model_tag (only for serve)
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        if args[0] == "serve":
            if index == 1:
                raise ValueError(
                    "No model_tag specified! Please check your command-line"
                    " arguments."
                )
            command = args_before_config[0]
            model_tag = args_before_config[1]
            other_args_before = args_before_config[2:]
            args = (
                [command, model_tag]
                + config_args
                + other_args_before
                + args_after_config
            )
        else:
            command = args_before_config[0]
            other_args_before = args_before_config[1:]
            args = [command] + config_args + other_args_before + args_after_config

        return args

    def _load_config_file(self, file_path: str) -> list[str]:
        """Loads a yaml file and returns the key value pairs as a
        flattened list with argparse like pattern
        ```yaml
            port: 12323
            tensor-parallel-size: 4
            vae_config:
                load_encoder: false
                load_decoder: true
        ```
        returns:
            processed_args: list[str] = [
                '--port': '12323',
                '--tp-size': '4',
                '--vae-config.load-encoder': 'false',
                '--vae-config.load-decoder': 'true'
            ]
        """

        extension: str = file_path.split(".")[-1]
        if extension not in ("yaml", "yml", "json"):
            raise ValueError(
                "Config file must be of a yaml/yml/json type.\
                              %s supplied",
                extension,
            )

        processed_args: list[str] = []

        config: dict[str, Any] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. \
                Make sure path is correct",
                file_path,
            )
            raise ex

        store_boolean_arguments = [
            action.dest for action in self._actions if isinstance(action, StoreBoolean)
        ]

        def process_dict(prefix: str, d: dict[str, Any]):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, bool) and full_key not in store_boolean_arguments:
                    if value:
                        processed_args.append("--" + full_key)
                    else:
                        processed_args.append("--" + full_key)
                        processed_args.append("false")
                elif isinstance(value, list):
                    processed_args.append("--" + full_key)
                    for item in value:
                        processed_args.append(str(item))
                elif isinstance(value, dict):
                    process_dict(full_key, value)
                else:
                    processed_args.append("--" + full_key)
                    processed_args.append(str(value))

        process_dict("", config)

        return processed_args


def warn_for_unimplemented_methods(cls: type[T]) -> type[T]:
    """
    A replacement for `abc.ABC`.
    When we use `abc.ABC`, subclasses will fail to instantiate
    if they do not implement all abstract methods.
    Here, we only require `raise NotImplementedError` in the
    base class, and log a warning if the method is not implemented
    in the subclass.
    """

    original_init = cls.__init__

    def find_unimplemented_methods(self: object):
        unimplemented_methods = []
        for attr_name in dir(self):
            # bypass inner method
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(self, attr_name)
                # get the func of callable method
                if callable(attr):
                    attr_func = attr.__func__
            except AttributeError:
                continue
            src = inspect.getsource(attr_func)
            if "NotImplementedError" in src:
                unimplemented_methods.append(attr_name)
        if unimplemented_methods:
            method_names = ",".join(unimplemented_methods)
            msg = f"Methods {method_names} not implemented in {self}"
            logger.warning(msg)

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        find_unimplemented_methods(self)

    type.__setattr__(cls, "__init__", wrapped_init)
    return cls


def align_to(value: int, alignment: int) -> int:
    """align height, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)


def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully qualified name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# From vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py
def import_pynvml():
    """
    Historical comments:

    libnvml.so is the library behind nvidia-smi, and
    pynvml is a Python wrapper around it. We use it to get GPU
    status without initializing CUDA context in the current process.
    Historically, there are two packages that provide pynvml:
    - `nvidia-ml-py` (https://pypi.org/project/nvidia-ml-py/): The official
        wrapper. It is a dependency of sgl-diffusion, and is installed when users
        install sgl-diffusion. It provides a Python module named `pynvml`.
    - `pynvml` (https://pypi.org/project/pynvml/): An unofficial wrapper.
        Prior to version 12.0, it also provides a Python module `pynvml`,
        and therefore conflicts with the official one which is a standalone Python file.
        This causes errors when both of them are installed.
        Starting from version 12.0, it migrates to a new module
        named `pynvml_utils` to avoid the conflict.
    It is so confusing that many packages in the community use the
    unofficial one by mistake, and we have to handle this case.
    For example, `nvcr.io/nvidia/pytorch:24.12-py3` uses the unofficial
    one, and it will cause errors, see the issue
    https://github.com/vllm-project/vllm/issues/12847 for example.
    After all the troubles, we decide to copy the official `pynvml`
    module to our codebase, and use it directly.
    """
    import sglang.multimodal_gen.third_party.pynvml as pynvml

    return pynvml


def update_environment_variables(envs: dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s " "from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


def run_method(
    obj: Any, method: str | bytes | Callable, args: tuple[Any], kwargs: dict[str, Any]
) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not" " implemented."
            ) from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)


def shallow_asdict(obj) -> dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError("Expected dataclass instance")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


# TODO: validate that this is fine
def kill_itself_when_parent_died() -> None:
    # if sys.platform == "linux":
    # sigkill this process when parent worker manager dies
    PR_SET_PDEATHSIG = 1
    import platform

    if platform.system() == "Linux":
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    # elif platform.system() == "Darwin":
    #     libc = ctypes.CDLL("libc.dylib")
    #     logger.warning("kill_itself_when_parent_died is only supported in linux.")
    else:
        logger.warning("kill_itself_when_parent_died is only supported in linux.")


def get_exception_traceback() -> str:
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


class TypeBasedDispatcher:

    def __init__(self, mapping: list[tuple[type, Callable]]):
        self._mapping = mapping

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)
        raise ValueError(f"Invalid object: {obj}")


# For non-torch.distributed debugging
def remote_breakpoint() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", 0))  # Let the OS pick an ephemeral port.
        port = s.getsockname()[1]
        RemotePdb(host="localhost", port=port).set_trace()


@dataclass
class MixedPrecisionState:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    compute_dtype: torch.dtype | None = None
    mp_policy: MixedPrecisionPolicy | None = None


# Thread-local storage for mixed precision state
_mixed_precision_state = threading.local()


def get_mixed_precision_state() -> MixedPrecisionState:
    """Get the current mixed precision state."""
    if not hasattr(_mixed_precision_state, "state"):
        raise ValueError("Mixed precision state not set")
    return cast(MixedPrecisionState, _mixed_precision_state.state)


def set_mixed_precision_policy(
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    output_dtype: torch.dtype | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
):
    """Set mixed precision policy globally.

    Args:
        param_dtype: Parameter dtype used for training
        reduce_dtype: Reduction dtype used for gradients
        output_dtype: Optional output dtype
    """
    state = MixedPrecisionState(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        mp_policy=mp_policy,
    )
    _mixed_precision_state.state = state


def get_compute_dtype() -> torch.dtype:
    """Get the current compute dtype from mixed precision policy.

    Returns:
        torch.dtype: The compute dtype to use, defaults to get_default_dtype() if no policy set
    """
    if not hasattr(_mixed_precision_state, "state"):
        return torch.get_default_dtype()
    else:
        state = get_mixed_precision_state()
        return state.param_dtype


def dict_to_3d_list(
    mask_strategy: dict[str, Any] | None = None,
    t_max: int | None = None,
    l_max: int | None = None,
    h_max: int | None = None,
) -> list[list[list[torch.Tensor | None]]]:
    """
    Convert a dictionary of mask indices to a 3D list of tensors.
    Args:
        mask_strategy: keys are "t_l_h", values are torch.Tensor masks.
        t_max, l_max, h_max: if provided (all three), force the output shape to (t_max, l_max, h_max).
                            If all three are None, infer shape from the data.
    """
    # Case 1: no data, but fixed shape requested
    if mask_strategy is None:
        assert (
            t_max is not None and l_max is not None and h_max is not None
        ), "If mask_strategy is None, you must provide t_max, l_max, and h_max"
        return [
            [[None for _ in range(h_max)] for _ in range(l_max)] for _ in range(t_max)
        ]

    # Parse all keys into integer tuples
    indices = [tuple(map(int, key.split("_"))) for key in mask_strategy]

    # Decide on dimensions
    if t_max is None and l_max is None and h_max is None:
        # fully dynamic: infer from data
        max_timesteps_idx = max(t for t, _, _ in indices) + 1
        max_layer_idx = max(l for _, l, _ in indices) + 1  # noqa: E741
        max_head_idx = max(h for _, _, h in indices) + 1
    else:
        # require all three to be provided
        assert t_max is not None and l_max is not None and h_max is not None, (
            "Either supply none of (t_max, l_max, h_max) to infer dimensions, "
            "or supply all three to fix the shape."
        )
        max_timesteps_idx = t_max
        max_layer_idx = l_max
        max_head_idx = h_max

    # Preallocate
    result = [
        [[None for _ in range(max_head_idx)] for _ in range(max_layer_idx)]
        for _ in range(max_timesteps_idx)
    ]

    # Fill in, skipping any out-of-bounds entries
    for key, value in mask_strategy.items():
        t, l, h = map(int, key.split("_"))  # noqa: E741
        if (
            0 <= t < max_timesteps_idx
            and 0 <= l < max_layer_idx
            and 0 <= h < max_head_idx
        ):
            result[t][l][h] = value
        # else: silently ignore any key that doesn't fit

    return result


def set_random_seed(seed: int) -> None:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    current_platform.seed_everything(seed)


@lru_cache(maxsize=1)
def is_vsa_available() -> bool:
    return importlib.util.find_spec("vsa") is not None


@lru_cache(maxsize=1)
def is_vmoba_available() -> bool:
    if importlib.util.find_spec("kernel.csrc.attn.vmoba_attn.vmoba") is None:
        return False
    try:
        import flash_attn

        return flash_attn.__version__ >= "2.7.4"
    except Exception:
        return False


# adapted from: https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/utils.py
def masks_like(
    tensor, zero=False, generator=None, p=0.2
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2, strict=False):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device
                ).item()
                if random_num < p:
                    u[:, 0] = (
                        torch.normal(
                            mean=-3.5,
                            std=0.5,
                            size=(1,),
                            device=u.device,
                            generator=generator,
                        )
                        .expand_as(u[:, 0])
                        .exp()
                    )
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]

        else:
            for u, v in zip(out1, out2, strict=False):
                u[:, 0] = torch.zeros_like(u[:, 0])
                v[:, 0] = torch.zeros_like(v[:, 0])

    return out1, out2


# adapted from: https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/utils.py
def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2


def save_decoded_latents_as_video(
    decoded_latents: list[torch.Tensor], output_path: str, fps: int
):
    # Process outputs
    videos = rearrange(decoded_latents, "b c t h w -> t b c h w")
    frames = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        frames.append((x * 255).numpy().astype(np.uint8))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, format="mp4")
