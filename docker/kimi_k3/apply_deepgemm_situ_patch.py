"""Patch deep_gemm's mega-MoE JIT header to support Kimi-K3 SiTU activation.

Mechanism (no _C.so rebuild needed — the kernel body is a runtime-JIT header):
the host passes `activation='swiglu'` with the magic `activation_clamp =
0.03125` (2^-5: exactly representable, round-trips through the host's float
stringification, and no legitimate swiglu clamp uses it; the host asserts
clamp >= 0 so a negative sentinel is not possible). In-kernel:
kActivationClamp == 0.03125f selects SiTU with K3 constants baked in:
  beta = 4.0, linear_beta = 25.0  (config activation_situ_{beta,linear_beta})
SiTU(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * (linear_beta*tanh(up/linear_beta))

A distinct clamp value produces a distinct JIT template instantiation, so the
new variant compiles fresh; cached swiglu kernels are unaffected. If you edit
the SiTU math itself, clear /root/.cache/deep_gemm first (same sentinel value
would otherwise hit a stale cache entry).

Idempotent; also migrates the deprecated negative-sentinel V1 patch.
Run on every node: python3 apply_deepgemm_situ_patch.py
"""

P = "/usr/local/lib/python3.12/dist-packages/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"

OLD = """                        // Apply SwiGLU: silu(gate) * up
                        // Gate/up pairs: (0, 2), (1, 3), (4, 6), (5, 7)
                        auto fp32_values = reinterpret_cast<float*>(values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
                            auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }
"""

NEW = """                        // Apply activation: SwiGLU, or Kimi-K3 SiTU via sentinel
                        // Gate/up pairs: (0, 2), (1, 3), (4, 6), (5, 7)
                        // K3-SITU-PATCH: kActivationClamp == 0.03125f (2^-5 magic;
                        // host asserts clamp >= 0 so negatives can't sentinel) selects SiTU:
                        //   act = kSituBeta * tanh(gate/kSituBeta) * sigmoid(gate)
                        //   up' = kSituLinearBeta * tanh(up/kSituLinearBeta)
                        // K3 config constants baked in (activation_situ_{beta,linear_beta}).
                        constexpr bool kUseSitu = (kActivationClamp == 0.03125f);
                        constexpr float kSituBeta = 4.0f;
                        constexpr float kSituLinearBeta = 25.0f;
                        auto fp32_values = reinterpret_cast<float*>(values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
                            auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));

                            // Clamp (SwiGLU-with-limit only; SiTU soft-clips below)
                            if constexpr (!kUseSitu && kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // sigmoid(gate)
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            float2 sig;
                            if constexpr (kFastMath) {
                                sig = {math::fast_rcp(denom.x), math::fast_rcp(denom.y)};
                            } else {
                                sig = {1.0f / denom.x, 1.0f / denom.y};
                            }
                            auto up = __bfloat1622float2(bf16_up);
                            if constexpr (kUseSitu) {
                                // K3-SITU-PATCH: tanh-bounded gate, soft-clipped up
                                gate = {kSituBeta * tanhf(gate.x / kSituBeta) * sig.x,
                                        kSituBeta * tanhf(gate.y / kSituBeta) * sig.y};
                                up = {kSituLinearBeta * tanhf(up.x / kSituLinearBeta),
                                      kSituLinearBeta * tanhf(up.y / kSituLinearBeta)};
                            } else {
                                // SwiGLU: silu(gate) * up
                                gate = __fmul2_rn(gate, sig);
                            }
                            activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }
"""

V1_LINES = """                        constexpr bool kUseSitu = kActivationClamp < 0.0f;
                        constexpr float kSituBeta = kUseSitu ? -kActivationClamp : 1.0f;
                        constexpr float kSituLinearBeta = 25.0f;"""
V2_LINES = """                        constexpr bool kUseSitu = (kActivationClamp == 0.03125f);
                        constexpr float kSituBeta = 4.0f;
                        constexpr float kSituLinearBeta = 25.0f;"""

s = open(P).read()
if V2_LINES in s:
    print("already patched (v2)")
elif V1_LINES in s:
    # migrate deprecated negative-sentinel v1 -> magic-value v2
    assert s.count(V1_LINES) == 1
    s = s.replace(V1_LINES, V2_LINES)
    s = s.replace(
        """                        // K3-SITU-PATCH: kActivationClamp < 0 selects SiTU:
                        //   act = kSituBeta * tanh(gate/kSituBeta) * sigmoid(gate)
                        //   up' = kSituLinearBeta * tanh(up/kSituLinearBeta)
                        // with kSituBeta = -kActivationClamp (host passes -beta).
""",
        """                        // K3-SITU-PATCH: kActivationClamp == 0.03125f (2^-5 magic;
                        // host asserts clamp >= 0 so negatives can't sentinel) selects SiTU:
                        //   act = kSituBeta * tanh(gate/kSituBeta) * sigmoid(gate)
                        //   up' = kSituLinearBeta * tanh(up/kSituLinearBeta)
                        // K3 config constants baked in (activation_situ_{beta,linear_beta}).
""",
    )
    open(P, "w").write(s)
    print("migrated v1 -> v2")
elif OLD in s:
    assert s.count(OLD) == 1
    open(P, "w").write(s.replace(OLD, NEW))
    print("patched")
else:
    raise SystemExit(
        "ERROR: expected SwiGLU epilogue block not found — header layout changed"
    )
