import importlib
import logging
import pkgutil

from sglang.srt.dllm.config import DllmConfig

logger = logging.getLogger(__name__)


def import_algorithms():
    mapping = {}
    package_name = "sglang.srt.dllm.algorithm"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if ispkg:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as e:
            logger.warning(f"Ignore import error when loading {name}: {e}")
            continue
        if not hasattr(module, "Algorithm"):
            continue

        algo = module.Algorithm
        mapping[algo.__name__] = algo

    return mapping


def get_algorithm(config: DllmConfig):
    try:
        name = config.algorithm
        return algo_name_to_cls[name](config)
    except:
        raise RuntimeError(f"Unknown diffusion LLM algorithm: {name}")


algo_name_to_cls = import_algorithms()
