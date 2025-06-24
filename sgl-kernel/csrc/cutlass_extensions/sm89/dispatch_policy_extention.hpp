#pragma once
#include <cutlass/gemm/dispatch_policy.hpp>


namespace cutlass::gemm {
    // FP8 related policies (including Blocked Scaled Accumulation)
    struct KernelMultistageBlockScaledAccumExtension { };

    
    // n-buffer in smem (cp.async), pipelined with registers, with predicated gmem loads
    template<int Stages_, class ClusterShape_ = Shape<_1,_1,_1> >
    struct MainloopSm80CpAsyncBlockScalingExtension {
        constexpr static int Stages = Stages_;
        using ArchTag = cute::conditional_t<(size(ClusterShape_{}) > 1), arch::Sm90, arch::Sm80>;
        using Schedule = KernelMultistageBlockScaledAccumExtension;
        using ClusterShape = ClusterShape_;
    };


}



