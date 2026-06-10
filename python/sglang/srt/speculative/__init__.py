# Importing this package registers plugin speculative-decoding algorithms.
# SUFFIX (arctic_inference SuffixDecoding) registers itself on import; the
# worker / arctic_inference are loaded lazily only when SUFFIX is selected.
from sglang.srt.speculative import sgl_suffix_plugin  # noqa: F401
