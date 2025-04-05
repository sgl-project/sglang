MOONCAKE = "mooncake"
PYVERBS = "pyverbs"
mode = PYVERBS

if mode == MOONCAKE:
    from sglang.srt.disaggregation.mooncake_conn import *
elif mode == PYVERBS:
    from sglang.srt.disaggregation.verbs_conn import *
