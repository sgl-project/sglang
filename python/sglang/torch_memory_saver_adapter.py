try:
    from torch_memory_saver import *
except ImportError:
    from .torch_memory_saver_noop import *
