import copy
import dataclasses
import functools
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import helion  # noqa: F401
import torch
from triton.testing import do_bench

logger = logging.getLogger(__name__)
AutotuneInputFn = Callable[[], Iterable[tuple[Any, ...]]]


def get_model_depths() -> list[int]:
    return [8, 16, 24, 32, 40, 48, 64]


def get_cuda_device_capability() -> tuple[int, int]:
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        cc_major, cc_minor = torch.cuda.get_device_capability(dev)
        return cc_major, cc_minor
    return (0, 0)


class AOTAutotuneMode(Enum):
    NONE = "none"
    CREATE = "create"
    RETUNE = "retune"

    @classmethod
    def from_str(cls, mode: str) -> "AOTAutotuneMode":
        return cls[mode.upper()]


def load_autotune_data_from_json(
    path: Path,
) -> tuple[dict[int, Any], dict[tuple[Any, Any, Any], int]]:
    with open(path, "r") as f:
        json_obj = json.load(f)
        useful_configs = {
            int(k): eval(v) for k, v in json_obj["useful_configs"].items()
        }
        hash_configs = {tuple(eval(k)): v for k, v in json_obj["hash_configs"].items()}
    return useful_configs, hash_configs


def bind_and_compile_kernel(
    kernel: helion.Kernel, args: Any, config: helion.Config
) -> Callable:
    has_int64 = any([i.numel() >= 2**31 for i in args if isinstance(i, torch.Tensor)])
    if has_int64:
        new_settings = dataclasses.replace(kernel.settings, index_dtype=torch.int64)
        kernel = helion.Kernel(
            kernel.fn,
            configs=kernel.configs,
            settings=new_settings,
            key=kernel._key_fn,
        )
    config = copy.deepcopy(config)
    return kernel.bind(args).compile_config(config, allow_print=False)


def helion_aot_autotune(
    config_path: str,
    kernel_key: Callable[..., tuple[Any, Any, Any]],
    primary_inputs: AutotuneInputFn,
    secondary_inputs: AutotuneInputFn | None = None,
    int64_threshold: Callable[..., bool] | None = None,
    warn_on_hash_miss: bool = False,
):
    """
    A decorator that automatically tunes and dispatches a Helion kernel based off of the kernel_key and provided inputs.

    The general flow is this:
    1. We first run helion_kernel.autotune on all primary_inputs. This will give us a list of configs, one per primary_input.
    2. We benchmark every config on every primary_input and secondary_input.
    3. For every primary input/secondary input, we keep the config that is the fastest (NB: We aim to do some deduplication by reusing configs if they're within some threshold of the fastest config).
    4. We'll save the configs and the dispatch choices to a json file.
    5. We create a dispatch function that will lookup to see whether each kernel is one we've tuned for in step 2. If so, we'll use that config. Otherwise, we'll use some heuristic to (deterministically) find a reasonable config for the kernel.

    There are 3 modes (set by env variable HELION_AOT_AUTOTUNE):
    - none: No autotuning will be done. We will skip to step 5. If the config json file doesn't exist, we'll raise an error.
    - retune: We only benchmark the existing useful configs on the primary/secondary inputs. We'll skip to step 2. This finishes much faster than create (albeit won't fully retune for each shape), but requires create to have been run first. For example, for rmsnorm, retune takes maybe one minute for 10 shapes, but create might take 30 minutes.
    - create: The kernel will be fully autotuned, starting from step 1.

    You can also minimize the kernels to be autotuned by setting the env variable HELION_AOT_AUTOTUNE_KERNEL to the name of the kernel. For example, if you want to only autotune rmsnorm, you can set HELION_AOT_AUTOTUNE_KERNEL="rms_norm_fwd".

    kernel_key: Callable[..., tuple[Any, Any, Any]]
    returns:
        (numeric_key, hash_key, exact_key)
    The semantics are that if all 3 match a saved key, we'll use that config.
    Otherwise, we'll use some heuristic to find a reasonable config for the
    kernel. The heuristic is:
    - We require exact_key to match the saved config.
    - We prioritize hash_key that match the saved config.
    - We then prioritize the highest numeric_key that's <= the current numeric_key.
    """
    # Threshold for how much faster a config has to be for some shape to be considered "useful"
    threshold = 1.01
    # How many ms to run the kernel when retuning
    retune_rep_ms = 1000

    def inner_autotune(kernel: helion.Kernel):
        helion_dir = Path(__file__).parent / "configs"

        cc_major, cc_minor = get_cuda_device_capability()
        if cc_minor == 3:
            cc_minor = 0
        gpu_arch = f"sm_{cc_major}{cc_minor}"

        path = helion_dir / Path(f"{config_path}_{gpu_arch}.json")

        @functools.wraps(kernel_key)
        def wrapped_kernel_key(*inps: Any) -> tuple[Any, Any, Any]:
            """
            A wrapper that handles dtype specially (since dtype is not
            serializable to json).
            """

            numeric_key, hash_key, exact_key = kernel_key(*inps)
            assert isinstance(hash_key, tuple)
            assert isinstance(exact_key, tuple)
            hash_key = tuple(
                i.itemsize if isinstance(i, torch.dtype) else i for i in hash_key
            )
            if exact_key is not None:
                exact_key = tuple(
                    i.itemsize if isinstance(i, torch.dtype) else i for i in exact_key
                )
            return numeric_key, hash_key, exact_key

        autotune_mode_str = os.environ.get("HELION_AOT_AUTOTUNE", "none")
        autotune_mode = AOTAutotuneMode.from_str(autotune_mode_str)
        autotune_kernel = os.environ.get("HELION_AOT_AUTOTUNE_KERNEL", "all")
        if autotune_kernel != "all" and autotune_kernel != config_path:
            autotune_mode = AOTAutotuneMode.NONE

        @functools.cache
        def get_configs_from_autotuning(autotune_mode: AOTAutotuneMode):
            if autotune_mode == AOTAutotuneMode.NONE:
                if not path.exists():
                    raise RuntimeError(
                        f"Helion kernel not tuned yet. Run with HELION_AOT_AUTOTUNE=create HELION_AOT_AUTOTUNE_KERNEL={config_path} to generate the config at {path}"
                    )
                return load_autotune_data_from_json(path)

            inputs = sorted(
                list(primary_inputs()), key=lambda x: wrapped_kernel_key(*x)[0]
            )
            if autotune_mode == AOTAutotuneMode.CREATE:
                useful_configs = []
                for idx, input in enumerate(inputs):
                    print(
                        f"Autotuning for {config_path} with key: ",
                        wrapped_kernel_key(*input),
                    )
                    config = kernel.autotune(input)
                    useful_configs.append(repr(config))
            elif autotune_mode == AOTAutotuneMode.RETUNE:
                with open(path, "r") as f:
                    json_obj = json.load(f)
                    useful_configs = list(json_obj["useful_configs"].values())
            else:
                raise RuntimeError(f"Unexpected autotune mode: {autotune_mode}")

            logger.info("Candidate useful configs: ")
            for idx, config in enumerate(useful_configs):
                logger.info(f"{idx}:, Config: {config}")

            input_timings = []
            if secondary_inputs is not None:
                inputs += list(secondary_inputs())
                inputs = sorted(inputs, key=lambda x: kernel_key(*x)[0])

            for input in inputs:
                cur_input_key = wrapped_kernel_key(*input)
                timings = []
                for idx, config in enumerate(useful_configs):
                    try:
                        cur_kernel = bind_and_compile_kernel(
                            kernel, input, eval(config)
                        )
                        timings.append(
                            do_bench(lambda: cur_kernel(*input), rep=retune_rep_ms)
                        )  # noqa: B023
                    except Exception as e:
                        logger.info(f"Error compiling config {config}: {e}")
                        timings.append(float("inf"))
                input_timings.append((cur_input_key, timings))
            for idx, (key, timing) in enumerate(input_timings):
                logger.info(
                    f"Key {key} timings: {' '.join([f'{i:.5f}' for i in timing])}"
                )

            hash_configs_timings: dict[tuple[Any, Any], tuple[float, int | None]] = (
                defaultdict(lambda: (float("inf"), None))
            )
            for cur_kernel_key, input_timings in input_timings:
                for config_idx, (config, timing) in enumerate(
                    zip(useful_configs, input_timings)
                ):
                    if timing < hash_configs_timings[cur_kernel_key][0] * threshold:
                        hash_configs_timings[cur_kernel_key] = (timing, config_idx)

            kept_configs = {}
            hash_configs = {k: v[1] for k, v in hash_configs_timings.items()}
            for key, config_idx in hash_configs.items():
                assert config_idx is not None
                kept_configs[config_idx] = useful_configs[config_idx]

            json_obj = {
                "useful_configs": kept_configs,
                "hash_configs": {repr(k): v for k, v in hash_configs.items()},
            }

            with open(path, "w") as f:
                json.dump(json_obj, f, indent=2)
                f.write("\n")
            return load_autotune_data_from_json(path)

        cached_kernels = {}

        def default_int64_threshold(*args: Any) -> bool:
            return any(
                [i.numel() >= 2**31 for i in args if isinstance(i, torch.Tensor)]
            )

        used_int64_threshold = (
            int64_threshold if int64_threshold is not None else default_int64_threshold
        )

        def wrapped_func(*args: Any):
            nonlocal cached_kernels
            cur_kernel_key = wrapped_kernel_key(*args)
            has_int64 = used_int64_threshold(*args)
            size1 = tuple(
                tuple(shape == 1 for shape in arg.shape)
                for arg in args
                if isinstance(arg, torch.Tensor)
            )
            dtypes = tuple(arg.dtype for arg in args if isinstance(arg, torch.Tensor))
            scalar_args = tuple(a for a in args if not isinstance(a, torch.Tensor))
            key = (cur_kernel_key, has_int64, size1, dtypes, scalar_args)
            if key in cached_kernels:
                out = cached_kernels[key](*args)
                return out
            if has_int64:
                kernel.settings = dataclasses.replace(
                    kernel.settings, index_dtype=torch.int64
                )
            useful_configs, hash_configs = get_configs_from_autotuning(autotune_mode)
            key_to_config = {k: useful_configs[v] for k, v in hash_configs.items()}
            if key not in cached_kernels and cur_kernel_key in hash_configs:
                cached_kernels[key] = bind_and_compile_kernel(
                    kernel, args, key_to_config[cur_kernel_key]
                )
            else:
                if warn_on_hash_miss:
                    logger.warning(
                        f"No config found for key {cur_kernel_key} for kernel={config_path}. Finding best match. This is *not* a correctness issue, but means that the performance of this kernel could potentially be improved."
                    )
                used_config = None

                def config_key_sort(
                    config_key: tuple[Any, Any, Any],
                ) -> tuple[Any, ...]:
                    return (
                        config_key[2] == cur_kernel_key[2],
                        config_key[1] == cur_kernel_key[1],
                        config_key[0] <= cur_kernel_key[0],
                        config_key[0],
                    )

                sorted_keys = sorted(
                    hash_configs.keys(), key=config_key_sort, reverse=True
                )
                used_config = key_to_config[sorted_keys[0]]
                if len(used_config) == 3:
                    assert (
                        used_config[2] == cur_kernel_key[2]
                    ), "Exact key not found in configs"
                cached_kernels[key] = bind_and_compile_kernel(kernel, args, used_config)
            out = cached_kernels[key](*args)
            return out

        return wrapped_func

    return inner_autotune
