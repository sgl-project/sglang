from . import compare, generate, precision, utils

try:
    from . import bench
    from .bench import bench_by_cuda_events, bench_kineto
except ImportError:
    bench = None

from .compare import (
    check_is_allclose,
    check_is_allclose_comparator,
    check_is_bitwise_equal,
    check_is_bitwise_equal_comparator,
    get_cos_diff,
)
from .generate import (
    gen_non_contiguous_randn_tensor,
    gen_non_contiguous_tensor,
    non_contiguousify,
)
from .precision import (
    LowPrecisionMode,
    is_low_precision_mode,
    optional_cast_to_bf16_and_cast_back,
)
from .utils import Counter, cdiv, colors, is_using_profiling_tools, set_random_seed
