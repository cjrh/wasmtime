//! Admission checks for 256-bit vectors (fork-only).
//!
//! The x64 backend lowers `F64X4` and `I64X4` to `ymm` registers for a closed
//! set of operations (`lower_ymm.isle`). Generic rules elsewhere would match a
//! 256-bit type and emit 128-bit instructions for it, so a function that uses
//! one must be admitted here first; anything outside the set is rejected with
//! `CodegenError::Unsupported` instead of being miscompiled.
//!
//! Register state: a function that uses 256-bit values moves, spills and
//! reloads every float register as 256 bits (`wide_float_canonical_type`) and
//! executes `vzeroupper` before each call and return. The System V and Tail
//! conventions preserve no vector register across a call, so no 256-bit value
//! is live there; arguments and results are at most 128 bits, which
//! `vzeroupper` keeps. Conventions that preserve vector registers save only
//! their low 128 bits, so they are rejected.

use crate::ir::{
    Endianness, Function, InstructionData, Opcode, Signature, Type, condcodes::FloatCC, types,
};
use crate::isa::CallConv;
use crate::isa::x64::settings as x64_settings;
use crate::result::{CodegenError, CodegenResult};
use alloc::format;

fn is_ymm(ty: Type) -> bool {
    ty.is_vector() && ty.bits() == 256
}

/// The 256-bit types that `lower_ymm.isle` lowers.
fn is_lowered_type(ty: Type) -> bool {
    ty == types::F64X4 || ty == types::I64X4
}

/// Whether any value in `f` has a 256-bit vector type.
pub(crate) fn uses_ymm(f: &Function) -> bool {
    f.dfg.values().any(|v| is_ymm(f.dfg.value_type(v)))
}

fn unsupported(f: &Function, what: impl core::fmt::Display) -> CodegenError {
    CodegenError::Unsupported(format!("256-bit vectors in {}: {what}", f.name))
}

fn check_signature(f: &Function, sig: &Signature) -> CodegenResult<()> {
    let wide = sig
        .params
        .iter()
        .chain(&sig.returns)
        .any(|p| is_ymm(p.value_type));
    if wide {
        return Err(unsupported(f, "256-bit arguments and results"));
    }
    Ok(())
}

/// Calling conventions that preserve no vector register.
fn convention_allowed(cc: CallConv) -> bool {
    matches!(cc, CallConv::SystemV | CallConv::Tail | CallConv::Fast)
}

/// Admit `f` for lowering. A function without 256-bit values is unaffected.
pub(crate) fn check_function(f: &Function, flags: &x64_settings::Flags) -> CodegenResult<()> {
    check_signature(f, &f.signature)?;
    for (_, sig) in f.dfg.signatures.iter() {
        check_signature(f, sig)?;
    }
    if !uses_ymm(f) {
        return Ok(());
    }
    if !(flags.has_avx() && flags.has_avx2() && flags.has_sse41()) {
        return Err(unsupported(f, "requires AVX2"));
    }
    if !convention_allowed(f.signature.call_conv) {
        return Err(unsupported(
            f,
            format!("calling convention {}", f.signature.call_conv),
        ));
    }
    for (_, sig) in f.dfg.signatures.iter() {
        if !convention_allowed(sig.call_conv) {
            return Err(unsupported(
                f,
                format!("call with convention {}", sig.call_conv),
            ));
        }
    }
    for block in f.layout.blocks() {
        for inst in f.layout.block_insts(block) {
            let dfg = &f.dfg;
            let wide = dfg
                .inst_results(inst)
                .iter()
                .any(|&v| is_ymm(dfg.value_type(v)))
                || dfg.inst_values(inst).any(|v| is_ymm(dfg.value_type(v)));
            if !wide {
                continue;
            }
            let lowered = |v| !is_ymm(dfg.value_type(v)) || is_lowered_type(dfg.value_type(v));
            if !(dfg.inst_results(inst).iter().all(|&v| lowered(v))
                && dfg.inst_values(inst).all(lowered))
            {
                return Err(unsupported(f, format!("`{}`", dfg.display_inst(inst))));
            }
            let data = &dfg.insts[inst];
            let opcode = data.opcode();
            let arg_ty = |i: usize| dfg.value_type(dfg.inst_args(inst)[i]);
            let result_ty = || dfg.value_type(dfg.first_result(inst));
            let ok = match opcode {
                // Control flow only forwards values; a call or return with a
                // 256-bit value fails the signature check above.
                Opcode::Jump | Opcode::Brif | Opcode::BrTable => true,
                Opcode::Load | Opcode::Store => data
                    .memflags_data(dfg)
                    .is_some_and(|m| m.explicit_endianness() != Some(Endianness::Big)),
                Opcode::Vconst
                | Opcode::Band
                | Opcode::Bor
                | Opcode::Bxor
                | Opcode::Bnot
                | Opcode::Bitselect
                | Opcode::Splat
                | Opcode::Insertlane
                | Opcode::Extractlane => true,
                Opcode::Fadd | Opcode::Fsub | Opcode::Fmul | Opcode::Fdiv | Opcode::Sqrt => {
                    result_ty() == types::F64X4
                }
                Opcode::Fcmp => match data {
                    InstructionData::FloatCompare { cond, .. } => {
                        !matches!(cond, FloatCC::OrderedNotEqual | FloatCC::UnorderedOrEqual)
                    }
                    _ => false,
                },
                // Between the two 256-bit types; both live in one register.
                Opcode::Bitcast => is_ymm(arg_ty(0)) && is_ymm(result_ty()),
                Opcode::X86Vextractf128 => {
                    let half = result_ty();
                    let index = match data {
                        InstructionData::BinaryImm8 { imm, .. } => *imm,
                        _ => u8::MAX,
                    };
                    index <= 1
                        && is_ymm(arg_ty(0))
                        && half.bits() == 128
                        && half.lane_type() == arg_ty(0).lane_type()
                }
                Opcode::X86Vinsertf128 => {
                    let lane = result_ty().lane_type();
                    (0..2).all(|i| arg_ty(i).bits() == 128 && arg_ty(i).lane_type() == lane)
                }
                _ => false,
            };
            if !ok {
                return Err(unsupported(f, format!("`{}`", dfg.display_inst(inst))));
            }
        }
    }
    Ok(())
}
