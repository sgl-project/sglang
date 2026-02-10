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


def get_algorithm_fdfo_requirement(algorithm_name: str) -> bool:
    """Query whether an algorithm requires first_done_first_out_mode.

    Determines the requirement by reading the algorithm class's requires_fdfo_mode
    class attribute, avoiding hardcoding algorithm names in config.py.

    Args:
        algorithm_name: The name of the algorithm

    Returns:
        True if the algorithm requires FDFO mode, False otherwise
    """
    if algorithm_name in algo_name_to_cls:
        return getattr(algo_name_to_cls[algorithm_name], "requires_fdfo_mode", False)
    return False


algo_name_to_cls = import_algorithms()
