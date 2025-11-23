import importlib
import logging
import pkgutil

from sglang.srt.diffusion.config import DiffusionConfig

logger = logging.getLogger(__name__)


def import_algorithms():
    mapping = {}
    package_name = "sglang.srt.diffusion.algorithm"
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


def get_algorithm(config: DiffusionConfig):
    try:
        name = config.algorithm
        return algo_name_to_cls[name](config)
    except:
        raise RuntimeError(f"Unknown diffusion algorithm: {name}")


algo_name_to_cls = import_algorithms()
