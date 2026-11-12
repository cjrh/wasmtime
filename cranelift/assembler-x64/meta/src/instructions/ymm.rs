//! 256-bit (VEX.L=1) forms of the AVX/AVX2 instructions Cranelift uses for
//! `F64X4` and `I64X4` values.
//!
//! These are kept in one file, apart from their 128-bit forms, so that the
//! fork's 256-bit support stays separate from upstream definitions. Formats
//! are named `Y*` to give distinct `<mnemonic>_<format>` names.

use crate::dsl::{Customization::*, Feature::*, Inst, Length::*, Location::*};
use crate::dsl::{fmt, inst, r, vex, w};

#[rustfmt::skip] // Keeps instructions on a single line.
pub fn list() -> Vec<Inst> {
    vec![
        // Moves: unaligned load and store, and register-to-register.
        inst("vmovupd", fmt("YA", [w(ymm1), r(ymm_m256)]), vex(L256)._66()._0f().op(0x10).r(), (_64b | compat) & avx),
        inst("vmovupd", fmt("YB", [w(ymm_m256), r(ymm1)]), vex(L256)._66()._0f().op(0x11).r(), (_64b | compat) & avx),
        inst("vmovapd", fmt("YR", [w(ymm1), r(ymm2)]), vex(L256)._66()._0f().op(0x28).r(), (_64b | compat) & avx),

        // Arithmetic.
        inst("vaddpd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256)]), vex(L256)._66()._0f().op(0x58).r(), (_64b | compat) & avx),
        inst("vsubpd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256)]), vex(L256)._66()._0f().op(0x5C).r(), (_64b | compat) & avx),
        inst("vmulpd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256)]), vex(L256)._66()._0f().op(0x59).r(), (_64b | compat) & avx),
        inst("vdivpd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256)]), vex(L256)._66()._0f().op(0x5E).r(), (_64b | compat) & avx),
        inst("vsqrtpd", fmt("Y", [w(ymm1), r(ymm_m256)]), vex(L256)._66()._0f().op(0x51).r(), (_64b | compat) & avx),

        // Comparison and bitwise mask operations.
        inst("vcmppd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256), r(imm8)]), vex(L256)._66()._0f().op(0xC2).r().ib(), (_64b | compat) & avx).custom(Display),
        inst("vandpd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256)]), vex(L256)._66()._0f().op(0x54).r(), (_64b | compat) & avx),
        inst("vandnpd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256)]), vex(L256)._66()._0f().op(0x55).r(), (_64b | compat) & avx),
        inst("vorpd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256)]), vex(L256)._66()._0f().op(0x56).r(), (_64b | compat) & avx),
        inst("vxorpd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256)]), vex(L256)._66()._0f().op(0x57).r(), (_64b | compat) & avx),

        // Lanes: broadcast, blend, and 128-bit half extract/insert.
        inst("vbroadcastsd", fmt("Y", [w(ymm1), r(xmm2)]), vex(L256)._66()._0f38().w0().op(0x19).r(), (_64b | compat) & avx2),
        inst("vpbroadcastq", fmt("Y", [w(ymm1), r(xmm2)]), vex(L256)._66()._0f38().w0().op(0x59).r(), (_64b | compat) & avx2),
        inst("vblendpd", fmt("Y", [w(ymm1), r(ymm2), r(ymm_m256), r(imm8)]), vex(L256)._66()._0f3a().wig().op(0x0D).r().ib(), (_64b | compat) & avx),
        inst("vextractf128", fmt("Y", [w(xmm_m128), r(ymm1), r(imm8)]), vex(L256)._66()._0f3a().w0().op(0x19).r().ib(), (_64b | compat) & avx),
        inst("vinsertf128", fmt("Y", [w(ymm1), r(ymm2), r(xmm_m128), r(imm8)]), vex(L256)._66()._0f3a().w0().op(0x18).r().ib(), (_64b | compat) & avx),
    ]
}
