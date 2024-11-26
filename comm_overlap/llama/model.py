import pkgutil
import importlib
import os
import sys
import sglang
import logging
from sglang.srt.utils import (
    crash_on_warnings,
    enable_show_time_cost,
    get_available_gpu_memory,
    monkey_patch_vllm_model_config,
    monkey_patch_vllm_p2p_access_check,
)

# print("sys.path:")
# print(sys.path)

# print("sglang path:")
# print(sglang.__file__)

# package_name = "sglang.srt.models"
# package = importlib.import_module(package_name)

# print("Files in package path:")
# for path in package.__path__:
#     print(f"Directory: {path}")
#     print(os.listdir(path))

# print("All Modules:")

# for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
#     print(f"Module: {name}, Is package: {ispkg}")

logger = logging.getLogger(__name__)

model_arch_name_to_cls = {}
package_name = "sglang.srt.models"
package = importlib.import_module(package_name)
for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
    print(name)
    if not ispkg:
        try:
            module = importlib.import_module(name)
        except Exception as e:
            logger.warning(f"Ignore import error when loading {name}. {e}")
            if crash_on_warnings():
                raise ValueError(f"Ignore import error when loading {name}. {e}")
            continue
        if hasattr(module, "EntryClass"):
            entry = module.EntryClass
            if isinstance(
                entry, list
            ):  # To support multiple model classes in one module
                for tmp in entry:
                    assert (
                        tmp.__name__ not in model_arch_name_to_cls
                    ), f"Duplicated model implementation for {tmp.__name__}"
                    model_arch_name_to_cls[tmp.__name__] = tmp
            else:
                assert (
                    entry.__name__ not in model_arch_name_to_cls
                ), f"Duplicated model implementation for {entry.__name__}"
                model_arch_name_to_cls[entry.__name__] = entry

print(model_arch_name_to_cls)