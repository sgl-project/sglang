"""Teach the vendored CUTLASS headers about SM107 until upstream CUTLASS does.

CUTLASS lists supported architectures explicitly in three headers. Without 1070
in all three, an sm_107a build compiles but loses the SM100 MMA/TMA features,
`setmaxnreg` register reconfiguration, and PDL.

Usage: patch_cutlass_sm107.py <cutlass include root>
"""

import os
import sys

REPL = {
    "cutlass/arch/config.h": [
        (
            "defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000)",
            "defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1070))",
        ),
        (
            "!defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM100_ALL))",
            "!defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM107_ALL)))",
        ),
        (
            "!defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) && CUDA_ARCH_FAMILY(1000))",
            "!defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) && (CUDA_ARCH_FAMILY(1000) || CUDA_ARCH_FAMILY(1070)))",
        ),
        (
            "(defined(__CUDA_ARCH_FEAT_SM90_ALL)  || defined(__CUDA_ARCH_FEAT_SM100_ALL) || \\",
            "(defined(__CUDA_ARCH_FEAT_SM90_ALL)  || defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM107_ALL) || \\",
        ),
    ],
    "cutlass/arch/reg_reconfig.h": [
        (
            "      || (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))     \\\n",
            "      || (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))     \\\n"
            "      || (__CUDA_ARCH__ == 1070 && defined(__CUDA_ARCH_FEAT_SM107_ALL))     \\\n",
        ),
        (
            "         (__CUDA_ARCH__ == 1000 && CUDA_ARCH_FAMILY(1000))  \\\n",
            "         (__CUDA_ARCH__ == 1000 && CUDA_ARCH_FAMILY(1000))  \\\n"
            "      || (__CUDA_ARCH__ == 1070 && CUDA_ARCH_FAMILY(1070))  \\\n",
        ),
    ],
    "cutlass/arch/grid_dependency_control.h": [
        (
            "__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))\n",
            "(__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1070) && (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM107_ALL)))\n",
        ),
        (
            "     (__CUDA_ARCH__ == 1010 &&\\\n",
            "     (__CUDA_ARCH__ == 1070 &&\\\n"
            "        (defined(__CUDA_ARCH_FEAT_SM107_ALL) || CUDA_ARCH_FAMILY(1070))) || \\\n"
            "     (__CUDA_ARCH__ == 1010 &&\\\n",
        ),
    ],
}

for root in sys.argv[1:]:
    for rel, repl in REPL.items():
        path = os.path.join(root, rel)
        s = open(path).read()
        if "1070" in s:
            print("already patched", path)
            continue
        for old, new in repl:
            assert s.count(old) == 1, (path, old, s.count(old))
            s = s.replace(old, new)
        open(path, "w").write(s)
        print("patched", path)
