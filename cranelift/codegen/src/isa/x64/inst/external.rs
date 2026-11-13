//! Interface with the external assembler crate.

use super::{
    Amode, Gpr, Inst, LabelUse, MachBuffer, MachLabel, OperandVisitor, OperandVisitorImpl,
    SyntheticAmode, VCodeConstant, WritableGpr, WritableXmm, Xmm, args::FromWritableReg,
};
use crate::{Reg, Writable, ir::TrapCode};
use alloc::string::String;
use cranelift_assembler_x64 as asm;
use regalloc2::{PReg, RegClass};

/// Define the types of registers Cranelift will use.
#[derive(Clone, Debug)]
pub struct CraneliftRegisters;
impl asm::Registers for CraneliftRegisters {
    type ReadGpr = Gpr;
    type ReadWriteGpr = PairedGpr;
    type WriteGpr = WritableGpr;
    type ReadXmm = Xmm;
    type ReadWriteXmm = PairedXmm;
    type WriteXmm = WritableXmm;
}

/// Convenience type alias of `asm::inst::Inst` with `R = CraneliftRegisters`
/// filled in.
pub type AsmInst = asm::inst::Inst<CraneliftRegisters>;

/// A pair of registers, one for reading and one for writing.
///
/// Due to how Cranelift's SSA form, we must track the read and write registers
/// separately prior to register allocation. Once register allocation is
/// complete, we expect the hardware encoding for both `read` and `write` to be
/// the same.
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(missing_docs, reason = "self-describing variants")]
pub struct PairedGpr {
    pub read: Gpr,
    pub write: WritableGpr,
}

impl From<WritableGpr> for PairedGpr {
    fn from(wgpr: WritableGpr) -> Self {
        let read = wgpr.to_reg();
        let write = wgpr;
        Self { read, write }
    }
}

/// For ABI ergonomics.
impl From<WritableGpr> for asm::Gpr<PairedGpr> {
    fn from(wgpr: WritableGpr) -> Self {
        asm::Gpr::new(wgpr.into())
    }
}

// For ABI ergonomics.
impl From<Writable<Reg>> for asm::GprMem<PairedGpr, Gpr> {
    fn from(wgpr: Writable<Reg>) -> Self {
        assert!(wgpr.to_reg().class() == RegClass::Int);
        let wgpr = WritableGpr::from_writable_reg(wgpr).unwrap();
        Self::Gpr(wgpr.into())
    }
}

// For ABI ergonomics.
impl From<Reg> for asm::GprMem<Gpr, Gpr> {
    fn from(gpr: Reg) -> Self {
        assert!(gpr.class() == RegClass::Int);
        let gpr = Gpr::unwrap_new(gpr);
        Self::Gpr(gpr)
    }
}

// For ABI ergonomics.
impl From<Writable<Reg>> for asm::GprMem<Gpr, Gpr> {
    fn from(wgpr: Writable<Reg>) -> Self {
        wgpr.to_reg().into()
    }
}

// For ABI ergonomics.
impl From<Writable<Reg>> for asm::Gpr<PairedGpr> {
    fn from(wgpr: Writable<Reg>) -> Self {
        assert!(wgpr.to_reg().class() == RegClass::Int);
        let wgpr = WritableGpr::from_writable_reg(wgpr).unwrap();
        Self::new(wgpr.into())
    }
}

impl From<Writable<Reg>> for asm::Gpr<WritableGpr> {
    fn from(wgpr: Writable<Reg>) -> Self {
        assert!(wgpr.to_reg().class() == RegClass::Int);
        let wgpr = WritableGpr::from_writable_reg(wgpr).unwrap();
        Self::new(wgpr)
    }
}

impl asm::AsReg for PairedGpr {
    fn enc(&self) -> u8 {
        let PairedGpr { read, write } = self;
        let read = enc_gpr(read);
        let write = enc_gpr(&write.to_reg());
        assert_eq!(read, write);
        write
    }

    fn to_string(&self, size: Option<asm::Size>) -> String {
        if self.read.is_real() {
            asm::gpr::enc::to_string(self.enc(), size.unwrap()).into()
        } else {
            let read = self.read.to_reg();
            let write = self.write.to_reg().to_reg();
            format!("(%{write:?} <- %{read:?})")
        }
    }

    fn new(_: u8) -> Self {
        panic!("disallow creation of new assembler registers")
    }
}

/// A pair of XMM registers, one for reading and one for writing.
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(missing_docs, reason = "self-describing variants")]
pub struct PairedXmm {
    pub read: Xmm,
    pub write: WritableXmm,
}

impl From<WritableXmm> for PairedXmm {
    fn from(wxmm: WritableXmm) -> Self {
        let read = wxmm.to_reg();
        let write = wxmm;
        Self { read, write }
    }
}

/// For ABI ergonomics.
impl From<WritableXmm> for asm::Xmm<PairedXmm> {
    fn from(wgpr: WritableXmm) -> Self {
        asm::Xmm::new(wgpr.into())
    }
}

// For emission ergonomics.
impl From<Writable<Reg>> for asm::Xmm<PairedXmm> {
    fn from(wxmm: Writable<Reg>) -> Self {
        assert!(wxmm.to_reg().class() == RegClass::Float);
        let wxmm = WritableXmm::from_writable_reg(wxmm).unwrap();
        Self::new(wxmm.into())
    }
}

// For emission ergonomics.
impl From<Reg> for asm::Xmm<Xmm> {
    fn from(xmm: Reg) -> Self {
        assert!(xmm.class() == RegClass::Float);
        let xmm = Xmm::unwrap_new(xmm);
        Self::new(xmm)
    }
}

// For emission ergonomics.
impl From<Reg> for asm::XmmMem<Xmm, Gpr> {
    fn from(xmm: Reg) -> Self {
        assert!(xmm.class() == RegClass::Float);
        let xmm = Xmm::unwrap_new(xmm);
        Self::Xmm(xmm)
    }
}

impl asm::AsReg for PairedXmm {
    fn enc(&self) -> u8 {
        let PairedXmm { read, write } = self;
        let read = enc_xmm(read);
        let write = enc_xmm(&write.to_reg());
        assert_eq!(read, write);
        write
    }

    fn to_string(&self, size: Option<asm::Size>) -> String {
        assert!(size.is_none(), "XMM registers do not have size variants");
        if self.read.is_real() {
            asm::xmm::enc::to_string(self.enc()).into()
        } else {
            let read = self.read.to_reg();
            let write = self.write.to_reg().to_reg();
            format!("(%{write:?} <- %{read:?})")
        }
    }

    fn new(_: u8) -> Self {
        panic!("disallow creation of new assembler registers")
    }
}

/// This bridges the gap between codegen and assembler for general purpose register types.
impl asm::AsReg for Gpr {
    fn enc(&self) -> u8 {
        enc_gpr(self)
    }

    fn to_string(&self, size: Option<asm::Size>) -> String {
        if self.is_real() {
            asm::gpr::enc::to_string(self.enc(), size.unwrap()).into()
        } else {
            format!("%{:?}", self.to_reg())
        }
    }

    fn new(_: u8) -> Self {
        panic!("disallow creation of new assembler registers")
    }
}

/// This bridges the gap between codegen and assembler for xmm register types.
impl asm::AsReg for Xmm {
    fn enc(&self) -> u8 {
        enc_xmm(self)
    }

    fn to_string(&self, size: Option<asm::Size>) -> String {
        assert!(size.is_none(), "XMM registers do not have size variants");
        if self.is_real() {
            asm::xmm::enc::to_string(self.enc()).into()
        } else {
            format!("%{:?}", self.to_reg())
        }
    }

    fn new(_: u8) -> Self {
        panic!("disallow creation of new assembler registers")
    }
}

/// A helper method for extracting the hardware encoding of a general purpose register.
#[inline]
fn enc_gpr(gpr: &Gpr) -> u8 {
    if let Some(real) = gpr.to_reg().to_real_reg() {
        real.hw_enc()
    } else {
        unreachable!()
    }
}

/// A helper method for extracting the hardware encoding of an xmm register.
#[inline]
fn enc_xmm(xmm: &Xmm) -> u8 {
    if let Some(real) = xmm.to_reg().to_real_reg() {
        real.hw_enc()
    } else {
        unreachable!()
    }
}

/// A wrapper to implement the `cranelift-assembler-x64` register allocation trait,
/// `RegallocVisitor`, in terms of the trait used in Cranelift,
/// `OperandVisitor`.
pub(crate) struct RegallocVisitor<'a, T>
where
    T: OperandVisitorImpl,
{
    pub collector: &'a mut T,
}

impl<'a, T: OperandVisitor> asm::RegisterVisitor<CraneliftRegisters> for RegallocVisitor<'a, T> {
    fn read_gpr(&mut self, reg: &mut Gpr) {
        self.collector.reg_use(reg);
    }

    fn read_write_gpr(&mut self, reg: &mut PairedGpr) {
        let PairedGpr { read, write } = reg;
        self.collector.reg_use(read);
        self.collector.reg_reuse_def(write, 0);
    }

    fn write_gpr(&mut self, reg: &mut WritableGpr) {
        self.collector.reg_def(reg);
    }

    fn fixed_read_gpr(&mut self, reg: &mut Gpr, enc: u8) {
        self.collector
            .reg_fixed_use(reg, fixed_reg(enc, RegClass::Int));
    }

    fn fixed_read_write_gpr(&mut self, reg: &mut PairedGpr, enc: u8) {
        let PairedGpr { read, write } = reg;
        self.collector
            .reg_fixed_use(read, fixed_reg(enc, RegClass::Int));
        self.collector
            .reg_fixed_def(write, fixed_reg(enc, RegClass::Int));
    }

    fn fixed_write_gpr(&mut self, reg: &mut WritableGpr, enc: u8) {
        self.collector
            .reg_fixed_def(reg, fixed_reg(enc, RegClass::Int));
    }

    fn read_xmm(&mut self, reg: &mut Xmm) {
        self.collector.reg_use(reg);
    }

    fn read_write_xmm(&mut self, reg: &mut PairedXmm) {
        let PairedXmm { read, write } = reg;
        self.collector.reg_use(read);
        self.collector.reg_reuse_def(write, 0);
    }

    fn write_xmm(&mut self, reg: &mut WritableXmm) {
        self.collector.reg_def(reg);
    }

    fn fixed_read_xmm(&mut self, reg: &mut Xmm, enc: u8) {
        self.collector
            .reg_fixed_use(reg, fixed_reg(enc, RegClass::Float));
    }

    fn fixed_read_write_xmm(&mut self, reg: &mut PairedXmm, enc: u8) {
        let PairedXmm { read, write } = reg;
        self.collector
            .reg_fixed_use(read, fixed_reg(enc, RegClass::Float));
        self.collector
            .reg_fixed_def(write, fixed_reg(enc, RegClass::Float));
    }

    fn fixed_write_xmm(&mut self, reg: &mut WritableXmm, enc: u8) {
        self.collector
            .reg_fixed_def(reg, fixed_reg(enc, RegClass::Float));
    }
}

/// Folds a reload of a spilled XMM value into the one operand that reads it.
///
/// Visits an instruction whose registers are all allocated and counts the
/// reads of `reg`. The reload can be folded when `reg` is read exactly once,
/// by a read-only register-or-memory operand that reads the same bytes from
/// the spill slot as from the register; with `apply` set, that operand
/// becomes the slot's address. Used by [`Inst::fold_reload`].
pub(crate) struct ReloadFolder {
    /// The register the reload writes.
    pub reg: Reg,
    /// The spill slot's offset from the stack pointer, less the outgoing
    /// argument area, as in `SyntheticAmode::SlotOffset`.
    pub slot_offset: i32,
    /// The bytes of the register the slot holds.
    pub slot_bytes: u32,
    /// Whether the slot's address is 16-byte aligned.
    pub slot_aligned: bool,
    /// Replace the operand rather than only count it.
    pub apply: bool,
    /// Operands that could read the slot in place of `reg`.
    pub foldable_reads: u32,
    /// All other reads of `reg`.
    pub other_reads: u32,
}

impl ReloadFolder {
    /// Whether the visited instruction can read the slot in place of `reg`.
    pub(crate) fn can_fold(&self) -> bool {
        self.foldable_reads == 1 && self.other_reads == 0
    }

    fn read(&mut self, reg: &Xmm) {
        if reg.to_reg() == self.reg {
            self.other_reads += 1;
        }
    }
}

impl asm::RegisterVisitor<CraneliftRegisters> for ReloadFolder {
    // `reg` is an XMM register, so no general-purpose operand reads it.
    fn read_gpr(&mut self, _: &mut Gpr) {}
    fn read_write_gpr(&mut self, _: &mut PairedGpr) {}
    fn write_gpr(&mut self, _: &mut WritableGpr) {}
    fn fixed_read_gpr(&mut self, _: &mut Gpr, _: u8) {}
    fn fixed_read_write_gpr(&mut self, _: &mut PairedGpr, _: u8) {}
    fn fixed_write_gpr(&mut self, _: &mut WritableGpr, _: u8) {}

    fn read_xmm(&mut self, reg: &mut Xmm) {
        self.read(reg);
    }
    fn read_write_xmm(&mut self, reg: &mut PairedXmm) {
        self.read(&reg.read);
    }
    fn write_xmm(&mut self, _: &mut WritableXmm) {}
    fn fixed_read_xmm(&mut self, reg: &mut Xmm, _: u8) {
        self.read(reg);
    }
    fn fixed_read_write_xmm(&mut self, reg: &mut PairedXmm, _: u8) {
        self.read(&reg.read);
    }
    fn fixed_write_xmm(&mut self, _: &mut WritableXmm, _: u8) {}

    fn read_xmm_mem_foldable(&mut self, op: &mut asm::XmmMem<Xmm, Gpr>, bytes: u8, align: bool) {
        let asm::XmmMem::Xmm(reg) = op else {
            return;
        };
        if reg.to_reg() != self.reg {
            return;
        }
        if u32::from(bytes) > self.slot_bytes || (align && !self.slot_aligned) {
            self.other_reads += 1;
            return;
        }
        self.foldable_reads += 1;
        if self.apply {
            *op = asm::XmmMem::Mem(asm::Amode::ImmReg {
                base: Gpr::RSP,
                simm32: asm::AmodeOffsetPlusKnownOffset {
                    simm32: self.slot_offset.into(),
                    offset: Some(offsets::KEY_SLOT_OFFSET),
                },
                trap: None,
            });
        }
    }
}

/// Decides whether an instruction that defines an XMM register may leave
/// some of the register's old bits in place. Used by
/// [`Inst::def_keeps_bits`]; anything not known to write the whole register
/// counts as keeping bits.
pub(crate) struct DefKeepsBits {
    /// The defined register.
    pub reg: Reg,
    /// The bytes of the register that matter: those a spill slot holds.
    pub slot_bytes: u32,
    /// Set when a write of `reg` may keep old bits.
    pub keeps: bool,
}

impl DefKeepsBits {
    fn write(&mut self, reg: &WritableXmm, keeps: bool) {
        if keeps && reg.to_reg().to_reg() == self.reg {
            self.keeps = true;
        }
    }
}

impl asm::RegisterVisitor<CraneliftRegisters> for DefKeepsBits {
    fn read_gpr(&mut self, _: &mut Gpr) {}
    fn read_write_gpr(&mut self, _: &mut PairedGpr) {}
    fn write_gpr(&mut self, _: &mut WritableGpr) {}
    fn fixed_read_gpr(&mut self, _: &mut Gpr, _: u8) {}
    fn fixed_read_write_gpr(&mut self, _: &mut PairedGpr, _: u8) {}
    fn fixed_write_gpr(&mut self, _: &mut WritableGpr, _: u8) {}
    fn read_xmm(&mut self, _: &mut Xmm) {}
    fn fixed_read_xmm(&mut self, _: &mut Xmm, _: u8) {}

    // A read-write operand is a use as well, which the caller sees.
    fn read_write_xmm(&mut self, reg: &mut PairedXmm) {
        self.write(&reg.write, true);
    }
    fn fixed_read_write_xmm(&mut self, reg: &mut PairedXmm, _: u8) {
        self.write(&reg.write, true);
    }
    // A VEX write: every bit up to 256 is the result or zero.
    fn write_xmm(&mut self, reg: &mut WritableXmm) {
        self.write(reg, false);
    }
    // A legacy write keeps the bits above 128, and a scalar one the bits
    // above its result.
    fn write_xmm_legacy(&mut self, reg: &mut WritableXmm, scalar: bool) {
        self.write(reg, scalar || self.slot_bytes > 16);
    }
    fn fixed_write_xmm(&mut self, reg: &mut WritableXmm, _: u8) {
        self.write(reg, true);
    }
    // The encoding and width of a register destination that could also be
    // memory are not reported.
    fn write_xmm_mem(&mut self, op: &mut asm::XmmMem<WritableXmm, Gpr>) {
        if let asm::XmmMem::Xmm(reg) = op {
            self.write(reg, true);
        }
    }
}

/// A helper for building a fixed register from its hardware encoding.
fn fixed_reg(enc: u8, class: RegClass) -> Reg {
    let preg = PReg::new(usize::from(enc), class);
    Reg::from_real_reg(preg)
}

impl From<SyntheticAmode> for asm::Amode<Gpr> {
    fn from(amode: SyntheticAmode) -> asm::Amode<Gpr> {
        match amode {
            SyntheticAmode::Real(amode) => amode.into(),
            SyntheticAmode::IncomingArg { offset } => asm::Amode::ImmReg {
                base: Gpr::RBP,
                simm32: asm::AmodeOffsetPlusKnownOffset {
                    simm32: (-i32::try_from(offset).unwrap()).into(),
                    offset: Some(offsets::KEY_INCOMING_ARG),
                },
                trap: None,
            },
            SyntheticAmode::SlotOffset { simm32 } => asm::Amode::ImmReg {
                base: Gpr::RSP,
                simm32: asm::AmodeOffsetPlusKnownOffset {
                    simm32: simm32.into(),
                    offset: Some(offsets::KEY_SLOT_OFFSET),
                },
                trap: None,
            },
            SyntheticAmode::ConstantOffset(vcode_constant) => asm::Amode::RipRelative {
                target: asm::DeferredTarget::Constant(asm::Constant(vcode_constant.as_u32())),
            },
        }
    }
}

impl From<Amode> for asm::Amode<Gpr> {
    fn from(amode: Amode) -> asm::Amode<Gpr> {
        match amode {
            Amode::ImmReg {
                simm32,
                base,
                flags,
            } => asm::Amode::ImmReg {
                simm32: asm::AmodeOffsetPlusKnownOffset {
                    simm32: simm32.into(),
                    offset: None,
                },
                base: Gpr::unwrap_new(base),
                trap: flags.trap_code().map(Into::into),
            },
            Amode::ImmRegRegShift {
                simm32,
                base,
                index,
                shift,
                flags,
            } => asm::Amode::ImmRegRegShift {
                base,
                index: asm::NonRspGpr::new(index),
                scale: asm::Scale::new(shift),
                simm32: simm32.into(),
                trap: flags.trap_code().map(Into::into),
            },
            Amode::RipRelative { target } => asm::Amode::RipRelative {
                target: asm::DeferredTarget::Label(asm::Label(target.as_u32())),
            },
        }
    }
}

impl<R: asm::AsReg> From<SyntheticAmode> for asm::XmmMem<R, Gpr> {
    fn from(amode: SyntheticAmode) -> Self {
        asm::XmmMem::Mem(amode.into())
    }
}

impl<R: asm::AsReg> From<SyntheticAmode> for asm::GprMem<R, Gpr> {
    fn from(amode: SyntheticAmode) -> Self {
        asm::GprMem::Mem(amode.into())
    }
}

impl<R: asm::AsReg> From<Amode> for asm::XmmMem<R, Gpr> {
    fn from(amode: Amode) -> Self {
        asm::XmmMem::Mem(amode.into())
    }
}

impl<R: asm::AsReg> From<Amode> for asm::GprMem<R, Gpr> {
    fn from(amode: Amode) -> Self {
        asm::GprMem::Mem(amode.into())
    }
}

/// Keep track of the offset slots to fill in during emission; see
/// `KnownOffsetTable`.
#[expect(missing_docs, reason = "self-describing keys")]
pub mod offsets {
    pub const KEY_INCOMING_ARG: u8 = 0;
    pub const KEY_SLOT_OFFSET: u8 = 1;
}

/// Implementor of the [`asm::CodeSink`] trait.
pub struct AsmCodeSink<'a> {
    /// The buffer this is emitting into.
    pub sink: &'a mut MachBuffer<Inst>,
    /// The value of `KEY_INCOMING_ARG`.
    pub incoming_arg_offset: i32,
    /// The value of `KEY_SLOT_OFFSET`.
    pub slot_offset: i32,
}

impl asm::CodeSink for AsmCodeSink<'_> {
    fn put1(&mut self, value: u8) {
        self.sink.put1(value)
    }

    fn put2(&mut self, value: u16) {
        self.sink.put2(value)
    }

    fn put4(&mut self, value: u32) {
        self.sink.put4(value)
    }

    fn put8(&mut self, value: u64) {
        self.sink.put8(value)
    }

    fn add_trap(&mut self, code: asm::TrapCode) {
        self.sink.add_trap(code.into());
    }

    fn use_target(&mut self, target: asm::DeferredTarget) {
        let offset = self.sink.cur_offset();
        match target {
            asm::DeferredTarget::Label(label) => {
                self.sink
                    .use_label_at_offset(offset, label.into(), LabelUse::JmpRel32);
            }
            asm::DeferredTarget::Constant(constant) => {
                let label = self.sink.get_label_for_constant(constant.into());
                self.sink
                    .use_label_at_offset(offset, label, LabelUse::JmpRel32);
            }
            asm::DeferredTarget::None => {}
        }
    }

    fn known_offset(&self, offset: asm::KnownOffset) -> i32 {
        match offset {
            offsets::KEY_INCOMING_ARG => self.incoming_arg_offset,
            offsets::KEY_SLOT_OFFSET => self.slot_offset,
            other => panic!("unknown \"known\" offset {other}"),
        }
    }
}

impl From<asm::TrapCode> for TrapCode {
    fn from(value: asm::TrapCode) -> Self {
        Self::from_raw(value.0)
    }
}

impl From<TrapCode> for asm::TrapCode {
    fn from(value: TrapCode) -> Self {
        Self(value.as_raw())
    }
}

impl From<asm::Label> for MachLabel {
    fn from(value: asm::Label) -> Self {
        Self::from_u32(value.0)
    }
}

impl From<MachLabel> for asm::Label {
    fn from(value: MachLabel) -> Self {
        Self(value.as_u32())
    }
}

impl From<asm::Constant> for VCodeConstant {
    fn from(value: asm::Constant) -> Self {
        Self::from_u32(value.0)
    }
}

// Include code generated by `cranelift-codegen/meta/src/gen_asm.rs`. This file
// contains a `isle_assembler_methods!` macro with Rust implementations of all
// the assembler instructions exposed to ISLE.
include!(concat!(env!("OUT_DIR"), "/assembler-isle-macro.rs"));
pub(crate) use isle_assembler_methods;

#[cfg(test)]
mod tests {
    use super::PairedGpr;
    use super::asm::{AsReg, Size};
    use crate::isa::x64::args::{FromWritableReg, Gpr, WritableGpr, WritableXmm, Xmm};
    use crate::isa::x64::inst::external::PairedXmm;
    use crate::{Reg, Writable};
    use regalloc2::{RegClass, VReg};

    #[test]
    fn pretty_print_registers() {
        // For logging, we need to be able to pretty-print the virtual registers
        // that Cranelift uses before register allocation. This test ensures
        // that these remain printable using the `AsReg::to_string` interface
        // (see issue #10631).

        let v200: Reg = VReg::new(200, RegClass::Int).into();
        let gpr200 = Gpr::new(v200).unwrap();
        assert_eq!(gpr200.to_string(Some(Size::Quadword)), "%v200");

        let v300: Reg = VReg::new(300, RegClass::Int).into();
        let wgpr300 = WritableGpr::from_writable_reg(Writable::from_reg(v300)).unwrap();
        let pair = PairedGpr {
            read: gpr200,
            write: wgpr300,
        };
        assert_eq!(pair.to_string(Some(Size::Quadword)), "(%v300 <- %v200)");

        let v400: Reg = VReg::new(400, RegClass::Float).into();
        let xmm400 = Xmm::new(v400).unwrap();
        assert_eq!(xmm400.to_string(None), "%v400");

        let v500: Reg = VReg::new(500, RegClass::Float).into();
        let wxmm500 = WritableXmm::from_writable_reg(Writable::from_reg(v500)).unwrap();
        let pair = PairedXmm {
            read: xmm400,
            write: wxmm500,
        };
        assert_eq!(pair.to_string(None), "(%v500 <- %v400)");
    }

    /// The destinations reload folding must treat as reads, and the EVEX
    /// forms it must not fold.
    #[test]
    fn reload_fold_destinations() {
        use crate::isa::x64::inst::regs::{xmm1, xmm2, xmm3};
        use crate::isa::x64::inst::{Inst, asm};
        use crate::machinst::MachInst;

        let dst = || WritableXmm::from_writable_reg(Writable::from_reg(xmm1())).unwrap();
        let src = || Xmm::new(xmm2()).unwrap();
        let other = || Xmm::new(xmm3()).unwrap();
        let reg = xmm1().to_real_reg().unwrap();
        let keeps = |inst: asm::inst::Inst<super::CraneliftRegisters>, bytes| {
            Inst::External { inst }.def_keeps_bits(reg, bytes)
        };

        // Legacy scalar forms with a write-only destination merge into it.
        assert!(keeps(
            asm::inst::roundsd_rmi::new(dst(), src(), 3).into(),
            16
        ));
        assert!(keeps(
            asm::inst::roundss_rmi::new(dst(), src(), 3).into(),
            16
        ));
        assert!(keeps(asm::inst::rcpss_rm::new(dst(), src()).into(), 16));
        assert!(keeps(asm::inst::rsqrtss_rm::new(dst(), src()).into(), 16));
        // A legacy 128-bit write replaces a 16-byte slot's worth of bits but
        // keeps the upper half of a 32-byte one.
        assert!(!keeps(asm::inst::movaps_a::new(dst(), src()).into(), 16));
        assert!(keeps(asm::inst::movaps_a::new(dst(), src()).into(), 32));
        // VEX forms zero everything above their result.
        let vex: asm::inst::Inst<_> =
            asm::inst::vroundsd_rvmi::new(dst(), other(), src(), 3).into();
        assert!(!keeps(vex.clone(), 16));
        assert!(!keeps(vex, 32));

        // A reload of the source register folds into the VEX form of
        // `vpabsd` and not into the EVEX one.
        let source = xmm2().to_real_reg().unwrap();
        let mut vex = Inst::External {
            inst: asm::inst::vpabsd_a::new(dst(), src()).into(),
        };
        assert!(vex.fold_reload(source, 16, 16, true, false));
        let mut evex = Inst::External {
            inst: asm::inst::vpabsd_c::new(dst(), src()).into(),
        };
        assert!(!evex.fold_reload(source, 16, 16, true, false));
    }

    /// A legacy SSE packed operand faults on an address that is not a
    /// multiple of 16, so it takes only a slot known to be aligned. Scalar
    /// and VEX operands take any slot.
    #[test]
    fn reload_fold_needs_an_aligned_slot_for_legacy_packed_operands() {
        use crate::isa::x64::inst::regs::{xmm1, xmm2, xmm3};
        use crate::isa::x64::inst::{Inst, asm};
        use crate::machinst::MachInst;

        let wdst = || WritableXmm::from_writable_reg(Writable::from_reg(xmm1())).unwrap();
        let dst = || PairedXmm::from(wdst());
        let src = || Xmm::new(xmm2()).unwrap();
        let other = || Xmm::new(xmm3()).unwrap();
        let source = xmm2().to_real_reg().unwrap();
        let folds = |inst: asm::inst::Inst<super::CraneliftRegisters>, aligned| {
            Inst::External { inst }.fold_reload(source, 16, 16, aligned, false)
        };

        let packed = || asm::inst::addpd_a::new(dst(), src()).into();
        assert!(folds(packed(), true));
        assert!(!folds(packed(), false));
        let scalar = || asm::inst::addsd_a::new(dst(), src()).into();
        assert!(folds(scalar(), true));
        assert!(folds(scalar(), false));
        let vex = || asm::inst::vaddpd_b::new(wdst(), other(), src()).into();
        assert!(folds(vex(), true));
        assert!(folds(vex(), false));

        // Committing an unaligned fold is refused too, and leaves the
        // instruction as it was.
        let mut inst = Inst::External { inst: packed() };
        let before = format!("{inst:?}");
        assert!(!inst.fold_reload(source, 16, 16, false, true));
        assert_eq!(format!("{inst:?}"), before);
    }
}
