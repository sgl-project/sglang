// Copyright (C) 2026 Tencent.
//
// Shared host helpers and architecture dispatch.
//
// One .so per architecture. HPC_ARCH_DISPATCH expands only the pair matching
// HPC_TARGET_ARCH; other pairs become unevaluated sizeof. Operators that
// compile for every known architecture call *_async directly.

#ifndef SRC_UTILS_UTILS_H_
#define SRC_UTILS_UTILS_H_

#ifndef HPC_TARGET_ARCH
#error "HPC_TARGET_ARCH is not defined; this file must be compiled by the project's CMake."
#endif

namespace hpc {

int get_sm_count();

int get_sm_major_version();

// Device SM architecture, major * 10 + minor. Cached; assumed same as device 0.
int get_sm_arch();

// Unreached dispatch: not implemented, or the loaded module does not match.
[[noreturn]] void throw_arch_not_supported(const char *op, const char *impl_archs);

}  // namespace hpc

// HPC_ARCH_DISPATCH("op", 90, expr, 100, expr)
// Evaluates the expr matching this module; raises if nothing launched.
// Compile-time selects the pair; runtime checks the device matches this module.

#define HPC_ARCH_UNEVAL(expr) static_cast<void>(sizeof((expr, 0)))

#define HPC_ARCH_EVAL_LIVE(arch, expr)    \
  do {                                    \
    if (::hpc::get_sm_arch() == (arch)) { \
      expr;                               \
      hpc_arch_dispatched_ = true;        \
    }                                     \
  } while (0)

#if HPC_TARGET_ARCH == 90
#define HPC_ARCH_EVAL_90(expr) HPC_ARCH_EVAL_LIVE(90, expr)
#else
#define HPC_ARCH_EVAL_90(expr) HPC_ARCH_UNEVAL(expr)
#endif

#if HPC_TARGET_ARCH == 100
#define HPC_ARCH_EVAL_100(expr) HPC_ARCH_EVAL_LIVE(100, expr)
#else
#define HPC_ARCH_EVAL_100(expr) HPC_ARCH_UNEVAL(expr)
#endif

#if HPC_TARGET_ARCH == 103
#define HPC_ARCH_EVAL_103(expr) HPC_ARCH_EVAL_LIVE(103, expr)
#else
#define HPC_ARCH_EVAL_103(expr) HPC_ARCH_UNEVAL(expr)
#endif

#define HPC_ARCH_CAT(a, b) HPC_ARCH_CAT_(a, b)
#define HPC_ARCH_CAT_(a, b) a##b

#define HPC_ARCH_NARGS(...) HPC_ARCH_NARGS_(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define HPC_ARCH_NARGS_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

#define HPC_ARCH_EVALS_2(a, e) HPC_ARCH_EVAL_##a(e)
#define HPC_ARCH_EVALS_4(a, e, ...) \
  HPC_ARCH_EVAL_##a(e);             \
  HPC_ARCH_EVALS_2(__VA_ARGS__)
#define HPC_ARCH_EVALS_6(a, e, ...) \
  HPC_ARCH_EVAL_##a(e);             \
  HPC_ARCH_EVALS_4(__VA_ARGS__)
#define HPC_ARCH_EVALS_8(a, e, ...) \
  HPC_ARCH_EVAL_##a(e);             \
  HPC_ARCH_EVALS_6(__VA_ARGS__)
#define HPC_ARCH_EVALS_10(a, e, ...) \
  HPC_ARCH_EVAL_##a(e);              \
  HPC_ARCH_EVALS_8(__VA_ARGS__)

#define HPC_ARCH_IMPL_STR_2(a, e) #a
#define HPC_ARCH_IMPL_STR_4(a, e, ...) #a ", " HPC_ARCH_IMPL_STR_2(__VA_ARGS__)
#define HPC_ARCH_IMPL_STR_6(a, e, ...) #a ", " HPC_ARCH_IMPL_STR_4(__VA_ARGS__)
#define HPC_ARCH_IMPL_STR_8(a, e, ...) #a ", " HPC_ARCH_IMPL_STR_6(__VA_ARGS__)
#define HPC_ARCH_IMPL_STR_10(a, e, ...) #a ", " HPC_ARCH_IMPL_STR_8(__VA_ARGS__)

#define HPC_ARCH_DISPATCH(op, ...) HPC_ARCH_DISPATCH_(op, HPC_ARCH_NARGS(__VA_ARGS__), __VA_ARGS__)
#define HPC_ARCH_DISPATCH_(op, n, ...) HPC_ARCH_DISPATCH__(op, n, __VA_ARGS__)
#define HPC_ARCH_DISPATCH__(op, n, ...)                                                      \
  do {                                                                                       \
    bool hpc_arch_dispatched_ = false;                                                       \
    HPC_ARCH_CAT(HPC_ARCH_EVALS_, n)(__VA_ARGS__);                                           \
    if (!hpc_arch_dispatched_) {                                                             \
      ::hpc::throw_arch_not_supported(op, HPC_ARCH_CAT(HPC_ARCH_IMPL_STR_, n)(__VA_ARGS__)); \
    }                                                                                        \
  } while (0)

#endif  // SRC_UTILS_UTILS_H_
