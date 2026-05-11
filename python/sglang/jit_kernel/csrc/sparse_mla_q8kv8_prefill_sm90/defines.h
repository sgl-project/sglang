#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
