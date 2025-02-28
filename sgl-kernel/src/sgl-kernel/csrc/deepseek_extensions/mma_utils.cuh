#pragma once

#include <cuda.h>

#include "utils.cuh"

namespace deep_gemm {

struct SM90_64x16x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %10, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},"
                     " %8,"
                     " %9,"
                     " p   , 1,    1;\n"
                     "}\n"
                     : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07)
                     : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 16;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x24x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %14, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7, "
                     " %8,   %9,   %10,  %11},"
                     " %12,"
                     " %13,"
                     " p   , 1,    1;\n"
                     "}\n"
                     : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                       "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11)
                     : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 24;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x32x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %18, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7, "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},"
                     " %16,"
                     " %17,"
                     " p   , 1,    1;\n"
                     "}\n"
                     : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                       "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
                     : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 32;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x40x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %22, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n40k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7, "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15, "
                     " %16,  %17,  %18,  %19},"
                     " %20,"
                     " %21,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 40;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x48x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %26, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7, "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15, "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23},"
                     " %24,"
                     " %25,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 48;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x56x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %30, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n56k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27}, "
                     " %28,"
                     " %29,"
                     " p   , 1,    1;\n"
                     "}\n"
                     : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                       "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                       "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                       "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27)
                     : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 56;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x64x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %34, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31}, "
                     " %32,"
                     " %33,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 64;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x72x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 float& d32, float& d33, float& d34, float& d35,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %38, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
                     " %32,  %33,  %34,  %35}, "
                     " %36,"
                     " %37,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
                      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              d[32], d[33], d[34], d[35],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 72;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x80x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 float& d32, float& d33, float& d34, float& d35, float& d36, float& d37, float& d38, float& d39,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %42, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
                     " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39}, "
                     " %40,"
                     " %41,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
                      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 80;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x88x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 float& d32, float& d33, float& d34, float& d35, float& d36, float& d37, float& d38, float& d39,
                                 float& d40, float& d41, float& d42, float& d43,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %46, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n88k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
                     " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
                     " %40,  %41,  %42,  %43}, "
                     " %44,"
                     " %45,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
                      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39),
                      "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
              d[40], d[41], d[42], d[43],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 88;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x96x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 float& d32, float& d33, float& d34, float& d35, float& d36, float& d37, float& d38, float& d39,
                                 float& d40, float& d41, float& d42, float& d43, float& d44, float& d45, float& d46, float& d47,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %50, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n96k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
                     " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
                     " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47}, "
                     " %48,"
                     " %49,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
                      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39),
                      "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
              d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 96;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x104x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 float& d32, float& d33, float& d34, float& d35, float& d36, float& d37, float& d38, float& d39,
                                 float& d40, float& d41, float& d42, float& d43, float& d44, float& d45, float& d46, float& d47,
                                 float& d48, float& d49, float& d50, float& d51,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %54, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n104k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
                     " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
                     " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
                     " %48,  %49,  %50,  %51}, "
                     " %52,"
                     " %53,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
                      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39),
                      "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
                      "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
              d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
              d[48], d[49], d[50], d[51],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 104;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x112x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 float& d32, float& d33, float& d34, float& d35, float& d36, float& d37, float& d38, float& d39,
                                 float& d40, float& d41, float& d42, float& d43, float& d44, float& d45, float& d46, float& d47,
                                 float& d48, float& d49, float& d50, float& d51, float& d52, float& d53, float& d54, float& d55,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %58, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n112k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
                     " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
                     " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
                     " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55}, "
                     " %56,"
                     " %57,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
                      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39),
                      "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
                      "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
              d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
              d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 112;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x120x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 float& d32, float& d33, float& d34, float& d35, float& d36, float& d37, float& d38, float& d39,
                                 float& d40, float& d41, float& d42, float& d43, float& d44, float& d45, float& d46, float& d47,
                                 float& d48, float& d49, float& d50, float& d51, float& d52, float& d53, float& d54, float& d55,
                                 float& d56, float& d57, float& d58, float& d59,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %62, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n120k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
                     " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
                     " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
                     " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
                     " %56,  %57,  %58,  %59}, "
                     " %60,"
                     " %61,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
                      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39),
                      "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
                      "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55),
                      "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
              d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
              d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
              d[56], d[57], d[58], d[59],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 120;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x128x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 float& d32, float& d33, float& d34, float& d35, float& d36, float& d37, float& d38, float& d39,
                                 float& d40, float& d41, float& d42, float& d43, float& d44, float& d45, float& d46, float& d47,
                                 float& d48, float& d49, float& d50, float& d51, float& d52, float& d53, float& d54, float& d55,
                                 float& d56, float& d57, float& d58, float& d59, float& d60, float& d61, float& d62, float& d63,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %66, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
                     " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
                     " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
                     " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
                     " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
                     " %64,"
                     " %65,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
                      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39),
                      "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
                      "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55),
                      "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
              d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
              d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
              d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 128;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x192x32_F32E4M3E4M3_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
                                 float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
                                 float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
                                 float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
                                 float& d32, float& d33, float& d34, float& d35, float& d36, float& d37, float& d38, float& d39,
                                 float& d40, float& d41, float& d42, float& d43, float& d44, float& d45, float& d46, float& d47,
                                 float& d48, float& d49, float& d50, float& d51, float& d52, float& d53, float& d54, float& d55,
                                 float& d56, float& d57, float& d58, float& d59, float& d60, float& d61, float& d62, float& d63,
                                 float& d64, float& d65, float& d66, float& d67, float& d68, float& d69, float& d70, float& d71,
                                 float& d72, float& d73, float& d74, float& d75, float& d76, float& d77, float& d78, float& d79,
                                 float& d80, float& d81, float& d82, float& d83, float& d84, float& d85, float& d86, float& d87,
                                 float& d88, float& d89, float& d90, float& d91, float& d92, float& d93, float& d94, float& d95,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %98, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n192k32.f32.e4m3.e4m3"
                     "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                     " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                     " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                     " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
                     " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
                     " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
                     " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
                     " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
                     " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
                     " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
                     " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
                     " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},  "
                     " %96,"
                     " %97,"
                     " p   , 1,    1;\n"
                     "}\n"
                    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
                      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
                      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
                      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
                      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39),
                      "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
                      "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55),
                      "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63),
                      "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71),
                      "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79),
                      "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87),
                      "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95)
                    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
              d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
              d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
              d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
              d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
              d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
              d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63],
              d[64], d[65], d[66], d[67], d[68], d[69], d[70], d[71],
              d[72], d[73], d[74], d[75], d[76], d[77], d[78], d[79],
              d[80], d[81], d[82], d[83], d[84], d[85], d[86], d[87],
              d[88], d[89], d[90], d[91], d[92], d[93], d[94], d[95],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 192;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                     :: "l"(smem_dst), "r"(src[0]), "r"(src[1]));
    }
};

template <typename dtype_t>
struct SM90_U32x4_STSM_N {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void* smem_dst) {
        const uint32_t src[4] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
                                 *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                     :: "l"(smem_dst), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

__forceinline__ __device__ uint32_t get_lane_id() {
    uint32_t lane_id;
    asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__  __forceinline__ uint32_t ld_shared(const uint32_t* __restrict__ ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ int4 ld_shared(const int4* __restrict__ ptr) {
    int4 ret;
    asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float ld_shared(const float* __restrict__ ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(ptr), "f"(val));
}

__device__ __forceinline__ void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(ptr), "r"(val));
}

template <int N>
__device__ void warpgroup_wait() {
    DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

union GmmaDescriptor {
    __host__ __device__ constexpr GmmaDescriptor() noexcept: desc_(0) {}

    __host__ __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept: desc_(desc) {}

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept: desc_(t.desc_) {}

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept: desc_(t.desc_) {}

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor const &t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    uint64_t desc_;
    uint32_t reg32_[2];
    uint16_t reg16_[4];

    struct {
        uint16_t start_address_: 14, : 2;
        uint16_t leading_byte_offset_: 14, : 2;
        uint16_t stride_byte_offset_: 14, : 2;
        uint8_t : 1, base_offset_: 3, : 4;
        uint8_t : 6, layout_type_: 2;
    } bitfield;

    // Decay to an `uint64_t`
    __host__ __device__ constexpr operator uint64_t() const noexcept { return desc_; }
};

template <class PointerType>
__device__ GmmaDescriptor make_smem_desc(PointerType smem_ptr, int layout_type,
                                         int leading_byte_offset = 0,
                                         int stride_byte_offset = 1024) {
    GmmaDescriptor desc;
    auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

template <int N>
struct FP8MMASelector {
    static constexpr auto select_type() {
        if constexpr (N == 16) return SM90_64x16x32_F32E4M3E4M3_SS();
        if constexpr (N == 24) return SM90_64x24x32_F32E4M3E4M3_SS();
        if constexpr (N == 32) return SM90_64x32x32_F32E4M3E4M3_SS();
        if constexpr (N == 40) return SM90_64x40x32_F32E4M3E4M3_SS();
        if constexpr (N == 48) return SM90_64x48x32_F32E4M3E4M3_SS();
        if constexpr (N == 56) return SM90_64x56x32_F32E4M3E4M3_SS();
        if constexpr (N == 64) return SM90_64x64x32_F32E4M3E4M3_SS();
        if constexpr (N == 72) return SM90_64x72x32_F32E4M3E4M3_SS();
        if constexpr (N == 80) return SM90_64x80x32_F32E4M3E4M3_SS();
        if constexpr (N == 88) return SM90_64x88x32_F32E4M3E4M3_SS();
        if constexpr (N == 96) return SM90_64x96x32_F32E4M3E4M3_SS();
        if constexpr (N == 104) return SM90_64x104x32_F32E4M3E4M3_SS();
        if constexpr (N == 112) return SM90_64x112x32_F32E4M3E4M3_SS();
        if constexpr (N == 120) return SM90_64x120x32_F32E4M3E4M3_SS();
        if constexpr (N == 128) return SM90_64x128x32_F32E4M3E4M3_SS();
        if constexpr (N == 192) return SM90_64x192x32_F32E4M3E4M3_SS();
    }

    using type = decltype(select_type());
};

} // namespace deep_gemm
