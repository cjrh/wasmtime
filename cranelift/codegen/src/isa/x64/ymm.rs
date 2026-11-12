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
use crate::settings::Flags;
use alloc::format;
use target_lexicon::Triple;

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

/// The start of every `CodegenError::Unsupported` message with which
/// [`check_function`] rejects a function. An embedder that builds a 256-bit
/// form speculatively can match it and build the function again without
/// 256-bit types.
pub const UNSUPPORTED_256_BIT: &str = "256-bit vectors in ";

fn unsupported(f: &Function, what: impl core::fmt::Display) -> CodegenError {
    CodegenError::Unsupported(format!("{UNSUPPORTED_256_BIT}{}: {what}", f.name))
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
pub(crate) fn check_function(
    f: &Function,
    shared: &Flags,
    flags: &x64_settings::Flags,
    triple: &Triple,
) -> CodegenResult<()> {
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
    // Lowering adds calls of its own: libcalls (`emit_vm_call`, for example
    // a scalar `fma` without FMA) and `memcpy` for struct arguments. Their
    // conventions come from `libcall_call_conv`, not from `f`.
    for default in [CallConv::triple_default(triple), f.signature.call_conv] {
        let cc = CallConv::for_libcall(shared, default);
        if !convention_allowed(cc) {
            return Err(unsupported(f, format!("libcall convention {cc}")));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::{Cursor, FuncCursor};
    use crate::ir::{AbiParam, InstBuilder, UserFuncName};
    use crate::settings::{self, Configurable};
    use alloc::string::String;
    use alloc::vec::Vec;
    use core::str::FromStr;

    fn flags(libcall: &str) -> Flags {
        let mut b = settings::builder();
        b.set("libcall_call_conv", libcall).unwrap();
        Flags::new(b)
    }

    fn x64_flags(avx2: bool) -> x64_settings::Flags {
        let mut b = x64_settings::builder();
        for name in ["has_avx", "has_sse41"] {
            b.enable(name).unwrap();
        }
        if avx2 {
            b.enable("has_avx2").unwrap();
        }
        x64_settings::Flags::new(&flags("isa_default"), &b)
    }

    /// A function that splats an `F64X4` and applies `op` to it.
    fn wide(cc: CallConv, op: impl FnOnce(&mut FuncCursor, crate::ir::Value)) -> Function {
        let mut f = Function::with_name_signature(UserFuncName::testcase("f"), Signature::new(cc));
        let block = f.dfg.make_block();
        f.layout.append_block(block);
        let mut cursor = FuncCursor::new(&mut f);
        cursor.goto_bottom(block);
        let zero = cursor.ins().f64const(0.0);
        let v = cursor.ins().splat(types::F64X4, zero);
        op(&mut cursor, v);
        cursor.ins().return_(&[]);
        f
    }

    #[test]
    fn every_rejection_starts_with_the_exported_prefix() {
        let triple = Triple::from_str("x86_64-unknown-linux-gnu").unwrap();
        let ok = x64_flags(true);
        let default = flags("isa_default");
        let mut wide_param = wide(CallConv::SystemV, |_, _| {});
        wide_param
            .signature
            .params
            .push(AbiParam::new(types::F64X4));
        let mut wide_call = wide(CallConv::SystemV, |_, _| {});
        let mut sig = Signature::new(CallConv::SystemV);
        sig.returns.push(AbiParam::new(types::I64X4));
        wide_call.import_signature(sig);
        let mut windows_call = wide(CallConv::SystemV, |_, _| {});
        windows_call.import_signature(Signature::new(CallConv::WindowsFastcall));
        let cases: Vec<(&str, Function, Flags, x64_settings::Flags)> = Vec::from([
            ("256-bit arguments", wide_param, default.clone(), ok.clone()),
            ("and results", wide_call, default.clone(), ok.clone()),
            (
                "requires AVX2",
                wide(CallConv::SystemV, |_, _| {}),
                default.clone(),
                x64_flags(false),
            ),
            (
                "calling convention",
                wide(CallConv::WindowsFastcall, |_, _| {}),
                default.clone(),
                ok.clone(),
            ),
            (
                "call with convention",
                windows_call,
                default.clone(),
                ok.clone(),
            ),
            (
                "libcall convention",
                wide(CallConv::SystemV, |_, _| {}),
                flags("preserve_all"),
                ok.clone(),
            ),
            (
                "f32x8",
                wide(CallConv::SystemV, |c, _| {
                    let zero = c.ins().f32const(0.0);
                    c.ins().splat(types::F32X8, zero);
                }),
                default.clone(),
                ok.clone(),
            ),
            (
                "fneg",
                wide(CallConv::SystemV, |c, v| {
                    c.ins().fneg(v);
                }),
                default.clone(),
                ok.clone(),
            ),
        ]);
        // Each case names text of the path that must reject it.
        for (name, f, shared, x64) in cases {
            let message = match check_function(&f, &shared, &x64, &triple) {
                Err(CodegenError::Unsupported(message)) => message,
                other => panic!("{name}: expected a rejection, got {other:?}"),
            };
            assert!(
                message.starts_with(UNSUPPORTED_256_BIT) && message.contains(name),
                "{name}: {message}"
            );
        }
        let admitted: Result<(), String> =
            check_function(&wide(CallConv::SystemV, |_, _| {}), &default, &ok, &triple)
                .map_err(|e| alloc::format!("{e:?}"));
        assert_eq!(admitted, Ok(()));
    }
}
