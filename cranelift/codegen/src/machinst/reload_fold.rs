//! The decisions and the checker behind the reload rewrites of
//! `VCode::emit`.
//!
//! This module sees a block as a sequence of register allocator edits and of
//! instructions described by their operands, allocations and clobbers. What
//! an instruction can do with a spill slot is asked through [`ReloadInsts`],
//! so the logic here is independent of the backend and is tested on its own.
//!
//! # What the rewrites depend on
//!
//! A register keeps its value until an instruction declares a def or a
//! clobber of it, or an edit writes it. regalloc2 depends on the same
//! property for every allocatable register. For x64, every `Inst` variant's
//! emission was compared with its operand collection when this pass was
//! added:
//!
//! - multi-instruction pseudo-instructions declare their temporaries as
//!   defs;
//! - the TLS and stack-switch sequences declare clobbers;
//! - return calls take their `r11` temporary as a fixed def;
//! - instructions of the assembler crate report implicit operands as fixed
//!   operands;
//! - a call writes the temporary of `ABIMachineSpec::retval_temp_reg`
//!   (`r11`) when it copies a stack-returned value to a spill slot. `r11` is
//!   in the clobber set of a call to every convention except a
//!   `preserve_all` callee (`get_regs_clobbered_by_call`: System V, Windows
//!   fastcall, `fast`, `tail`, `winch`, `apple_aarch64`, `probestack` and
//!   every `try_call`), and the verifier does not let a `preserve_all`
//!   signature return values. So that this does not rest on the verifier,
//!   `MachInst::call_temp` names the temporary and the rewrites take its
//!   contents as unknown after every call (`may_write` below).
//!
//! A backend change that makes an instruction write an allocatable register
//! without declaring it breaks these rewrites, as it breaks the register
//! allocator.

use crate::FxHashMap;
use crate::machinst::InsnIndex;
use alloc::vec::Vec;
use regalloc2::{Allocation, Operand, OperandKind, PReg, PRegSet, SpillSlot};

/// What the instructions of a block can do with a spill slot.
pub(crate) trait ReloadInsts {
    /// Whether instruction `inst` reads `reg` only through one operand that
    /// can address `slot` instead.
    fn can_fold(&mut self, inst: InsnIndex, reg: PReg, slot: SpillSlot) -> bool;
    /// Rewrite that operand of `inst` to address `slot`. `false` leaves the
    /// instruction as it was.
    fn fold(&mut self, inst: InsnIndex, reg: PReg, slot: SpillSlot) -> bool;
    /// Whether `inst`, which defines `reg`, may keep some of the register's
    /// bits (`MachInst::def_keeps_bits`).
    fn def_keeps_bits(&self, inst: InsnIndex, reg: PReg) -> bool;
}

/// A register allocator edit that loaded a spill slot into a register which
/// has not been written since.
#[derive(Clone, Copy)]
struct Reload {
    reg: PReg,
    slot: SpillSlot,
    /// The index of the edit in `regalloc2::Output::edits`.
    edit: usize,
    /// The one instruction that has read `reg` so far, through an operand
    /// that can address the slot instead. Any other read abandons the
    /// reload.
    reader: Option<InsnIndex>,
}

/// What to emit for an edit changed by the rewrites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EditFold {
    /// Nothing: the reload's reader takes the slot, or the register
    /// already holds its value.
    Skip,
    /// A copy from this register, which holds the value of the slot the
    /// edit would have loaded.
    MoveFrom(PReg),
}

/// The reload rewrites of one block; the state is reused across blocks.
///
/// Feed a block's edits and instructions in program order to
/// [`ReloadFolds::edit`], [`ReloadFolds::inst`] and [`ReloadFolds::ret`]
/// after [`ReloadFolds::start_block`]. The decisions are then in `edits` and
/// `folded`.
#[derive(Default)]
pub(crate) struct ReloadFolds {
    /// Remove a reload whose one reader can address the spill slot.
    pub(crate) fold: bool,
    /// Reuse a register that still holds a spilled value.
    pub(crate) forward: bool,
    /// The pending reloads; at most one per register.
    reloads: Vec<Reload>,
    /// The registers that hold the value of a spill slot.
    held: Vec<(PReg, SpillSlot)>,
    /// The decisions for this block, keyed by edit index.
    pub(crate) edits: FxHashMap<usize, EditFold>,
    /// The instructions of this block that now read a slot in place of a
    /// register, for [`ReloadCheck`].
    pub(crate) folded: Vec<(InsnIndex, PReg, SpillSlot)>,
}

impl ReloadFolds {
    /// Forget the previous block.
    pub(crate) fn start_block(&mut self) {
        self.reloads.clear();
        self.held.clear();
        self.edits.clear();
        self.folded.clear();
    }

    /// Whether no reload is pending and no register is known to hold a
    /// slot's value, so that an instruction cannot change the state.
    pub(crate) fn is_idle(&self) -> bool {
        self.reloads.is_empty() && self.held.is_empty()
    }

    /// The edit with index `edit` moves `from` to `to`.
    pub(crate) fn edit(
        &mut self,
        edit: usize,
        from: Allocation,
        to: Allocation,
        insts: &mut impl ReloadInsts,
    ) {
        if let (Some(slot), Some(reg)) = (from.as_stack(), to.as_reg()) {
            if self.held.contains(&(reg, slot)) {
                // The register already holds the slot's value.
                self.edits.insert(edit, EditFold::Skip);
                return;
            }
        }
        // The edit reads `from`.
        if let Some(from) = from.as_reg() {
            if let Some(i) = self.reloads.iter().position(|r| r.reg == from) {
                self.reloads.swap_remove(i);
            }
        }
        // The edit writes `to`.
        if let Some(to) = to.as_reg() {
            self.reg_written(to, insts);
            self.held.retain(|(reg, _)| *reg != to);
            if let Some(from) = from.as_stack() {
                let holder = self
                    .held
                    .iter()
                    .find(|(reg, slot)| *slot == from && reg.class() == to.class());
                if let Some(&(src, _)) = holder {
                    // A register still holds what was stored to the slot:
                    // copy it. This comes before folding because the copy
                    // does not wait for the store.
                    self.edits.insert(edit, EditFold::MoveFrom(src));
                    self.held.push((to, from));
                } else if self.fold {
                    self.reloads.push(Reload {
                        reg: to,
                        slot: from,
                        edit,
                        reader: None,
                    });
                }
            }
        } else if let Some(to) = to.as_stack() {
            self.slot_written(to);
            if let (Some(from), true) = (from.as_reg(), self.forward) {
                self.held.push((from, to));
            }
        }
    }

    /// The block ends in a return that reads `read`. The epilogue replaces
    /// a return, and no register it does not read is live after it.
    pub(crate) fn ret(&mut self, read: &[Allocation], insts: &mut impl ReloadInsts) {
        while let Some(reload) = self.reloads.last() {
            let reg = reload.reg;
            if read.iter().any(|alloc| alloc.as_reg() == Some(reg)) {
                self.reloads.pop();
            } else {
                self.reg_written(reg, insts);
            }
        }
    }

    /// Instruction `inst` has these operands at these allocations and
    /// writes the registers of `clobbers` besides its defs. It may or may
    /// not write those of `may_write`.
    pub(crate) fn inst(
        &mut self,
        inst: InsnIndex,
        operands: &[Operand],
        allocs: &[Allocation],
        clobbers: PRegSet,
        may_write: PRegSet,
        insts: &mut impl ReloadInsts,
    ) {
        let mut read = PRegSet::default();
        let mut written = clobbers;
        for (operand, alloc) in operands.iter().zip(allocs) {
            match (operand.kind(), alloc.as_reg(), alloc.as_stack()) {
                (OperandKind::Use, Some(reg), _) => read.add(reg),
                (OperandKind::Def, Some(reg), _) => {
                    written.add(reg);
                    // The reloaded value stays visible through a def that
                    // keeps some of its bits, so the reload stays.
                    if let Some(i) = self.reloads.iter().position(|r| r.reg == reg) {
                        if insts.def_keeps_bits(inst, reg) {
                            self.reloads.swap_remove(i);
                        }
                    }
                }
                (OperandKind::Def, _, Some(slot)) => self.slot_written(slot),
                _ => {}
            }
        }

        let mut folded_here = false;
        let mut i = 0;
        while i < self.reloads.len() {
            let Reload {
                reg, slot, reader, ..
            } = self.reloads[i];
            if read.contains(reg) {
                // One memory operand per instruction, and one reader per
                // reload.
                if reader.is_none() && !folded_here && insts.can_fold(inst, reg, slot) {
                    folded_here = true;
                    self.reloads[i].reader = Some(inst);
                } else {
                    self.reloads.swap_remove(i);
                    continue;
                }
            }
            i += 1;
        }
        if written != PRegSet::default() {
            for reg in written {
                self.reg_written(reg, insts);
            }
            self.held.retain(|(reg, _)| !written.contains(*reg));
        }
        // A register that may have been written no longer holds a known
        // value, and it is not known to be dead either, so a reload into it
        // stays.
        for reg in may_write {
            if !written.contains(reg) {
                self.reloads.retain(|r| r.reg != reg);
                self.held.retain(|(held, _)| *held != reg);
            }
        }
    }

    /// `slot` is written: a reload of it that has not found its reader
    /// would read the new value, and no register holds that value.
    fn slot_written(&mut self, slot: SpillSlot) {
        self.reloads
            .retain(|r| r.slot != slot || r.reader.is_some());
        self.held.retain(|(_, s)| *s != slot);
    }

    /// `reg` is written: the value a pending reload left in it is dead.
    /// Commit the reload's fold if its one read found a reader that takes
    /// the slot.
    fn reg_written(&mut self, reg: PReg, insts: &mut impl ReloadInsts) {
        let Some(i) = self.reloads.iter().position(|r| r.reg == reg) else {
            return;
        };
        let reload = self.reloads.swap_remove(i);
        if let Some(inst) = reload.reader {
            let folded = insts.fold(inst, reg, reload.slot);
            debug_assert!(folded);
            if folded {
                self.edits.insert(reload.edit, EditFold::Skip);
                self.folded.push((inst, reg, reload.slot));
            }
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
enum Symbol {
    Initial(Allocation),
    Def(InsnIndex, usize),
    /// A def that may keep bits of the symbol it replaces.
    PartialDef(InsnIndex, usize, u32),
    Clobber(InsnIndex, PReg),
}

/// Checks the decisions of [`ReloadFolds`] for one block by running the
/// block twice over symbolic values, once with the allocator's edits and
/// once with the rewritten ones, and comparing what the instructions read.
/// Panics on a difference.
///
/// A location starts with a symbol of its own; a def or a clobber gives its
/// location a new one, which for a def that may keep bits depends on the old
/// one. Every operand read must see the same symbol in both runs, where a
/// folded operand reads its slot. Every location must end with the same
/// symbol, except in a block that returns, where nothing is live after the
/// return's operands.
///
/// This covers the decisions: which reloads go, which register a copy
/// takes, and the order of reads and writes within the block. It takes
/// operands, clobbers and `def_keeps_bits` as declared, so it does not find
/// an instruction that writes a register without declaring it, and it does
/// not decode the rewritten instruction: that the memory operand addresses
/// the right bytes is covered by tests.
#[derive(Default)]
pub(crate) struct ReloadCheck {
    symbols: FxHashMap<Symbol, u32>,
    /// Locations under the allocator's edits and the rewritten ones.
    locations: [FxHashMap<Allocation, u32>; 2],
    returns: bool,
}

const REWRITTEN: usize = 1;

impl ReloadCheck {
    fn symbol(&mut self, symbol: Symbol) -> u32 {
        let next = self.symbols.len() as u32;
        *self.symbols.entry(symbol).or_insert(next)
    }

    fn get(&mut self, run: usize, alloc: Allocation) -> u32 {
        if let Some(&symbol) = self.locations[run].get(&alloc) {
            return symbol;
        }
        let symbol = self.symbol(Symbol::Initial(alloc));
        self.locations[run].insert(alloc, symbol);
        symbol
    }

    /// An edit moves `from` to `to`; the rewrites decided `decision` for it.
    pub(crate) fn edit(&mut self, from: Allocation, to: Allocation, decision: Option<EditFold>) {
        let value = self.get(0, from);
        self.locations[0].insert(to, value);
        match decision {
            Some(EditFold::Skip) => {}
            Some(EditFold::MoveFrom(reg)) => {
                let value = self.get(REWRITTEN, Allocation::reg(reg));
                self.locations[REWRITTEN].insert(to, value);
            }
            None => {
                let value = self.get(REWRITTEN, from);
                self.locations[REWRITTEN].insert(to, value);
            }
        }
    }

    /// The return `inst` reads `allocs`.
    pub(crate) fn ret(&mut self, inst: InsnIndex, allocs: &[Allocation]) {
        for alloc in allocs {
            assert_eq!(
                self.get(0, *alloc),
                self.get(REWRITTEN, *alloc),
                "reload folding changed {alloc} read by the return {inst:?}",
            );
        }
        self.returns = true;
    }

    /// Instruction `inst` runs. `folded` is the register it no longer reads
    /// and the slot it reads instead; `keeps` is `def_keeps_bits` for it.
    /// A register of `may_write` that is not a def or a clobber must hold
    /// the same value in both runs before the instruction, since either may
    /// be what it holds afterwards; it then counts as clobbered.
    pub(crate) fn inst(
        &mut self,
        inst: InsnIndex,
        operands: &[Operand],
        allocs: &[Allocation],
        mut clobbers: PRegSet,
        may_write: PRegSet,
        folded: Option<(PReg, SpillSlot)>,
        keeps: impl Fn(PReg) -> bool,
    ) {
        let mut folded_reads = 0;
        for (operand, alloc) in operands.iter().zip(allocs) {
            if operand.kind() != OperandKind::Use {
                continue;
            }
            let original = self.get(0, *alloc);
            let rewritten = match folded {
                Some((reg, slot)) if alloc.as_reg() == Some(reg) => {
                    folded_reads += 1;
                    self.get(REWRITTEN, Allocation::stack(slot))
                }
                _ => self.get(REWRITTEN, *alloc),
            };
            assert_eq!(
                original, rewritten,
                "reload folding changed {alloc} read by {inst:?}",
            );
        }
        if let Some((reg, _)) = folded {
            assert_eq!(folded_reads, 1, "{inst:?} read {reg} more than once");
        }

        for reg in may_write {
            let alloc = Allocation::reg(reg);
            let defined = operands
                .iter()
                .zip(allocs)
                .any(|(operand, a)| operand.kind() == OperandKind::Def && *a == alloc);
            if clobbers.contains(reg) || defined {
                continue;
            }
            assert_eq!(
                self.get(0, alloc),
                self.get(REWRITTEN, alloc),
                "reload folding changed {alloc}, which {inst:?} may keep",
            );
            clobbers.add(reg);
        }
        for reg in clobbers {
            let symbol = self.symbol(Symbol::Clobber(inst, reg));
            for locations in &mut self.locations {
                locations.insert(Allocation::reg(reg), symbol);
            }
        }
        for (i, (operand, alloc)) in operands.iter().zip(allocs).enumerate() {
            if operand.kind() != OperandKind::Def {
                continue;
            }
            let keeps = alloc.as_reg().is_some_and(&keeps);
            for which in 0..2 {
                let symbol = if keeps {
                    let old = self.get(which, *alloc);
                    self.symbol(Symbol::PartialDef(inst, i, old))
                } else {
                    self.symbol(Symbol::Def(inst, i))
                };
                self.locations[which].insert(*alloc, symbol);
            }
        }
    }

    /// The block ends.
    pub(crate) fn finish(mut self) {
        if self.returns {
            return;
        }
        let allocs: Vec<Allocation> = self
            .locations
            .iter()
            .flat_map(|locations| locations.keys().copied())
            .collect();
        for alloc in allocs {
            assert_eq!(
                self.get(0, alloc),
                self.get(REWRITTEN, alloc),
                "reload folding changed {alloc} at the end of the block",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regalloc2::{RegClass, VReg};

    /// Instructions for which the test says which (instruction, register)
    /// reads can take a slot and which defs keep bits.
    #[derive(Default)]
    struct Insts {
        foldable: Vec<(usize, PReg)>,
        keeps: Vec<(usize, PReg)>,
    }

    impl ReloadInsts for Insts {
        fn can_fold(&mut self, inst: InsnIndex, reg: PReg, _slot: SpillSlot) -> bool {
            self.foldable.contains(&(inst.index(), reg))
        }
        fn fold(&mut self, inst: InsnIndex, reg: PReg, slot: SpillSlot) -> bool {
            self.can_fold(inst, reg, slot)
        }
        fn def_keeps_bits(&self, inst: InsnIndex, reg: PReg) -> bool {
            self.keeps.contains(&(inst.index(), reg))
        }
    }

    #[derive(Clone)]
    enum Step {
        Edit(Allocation, Allocation),
        /// Operands with their allocations, clobbers, and registers that
        /// may be written.
        Inst(Vec<(OperandKind, Allocation)>, PRegSet, PRegSet),
        Ret(Vec<Allocation>),
    }

    fn xmm(n: usize) -> PReg {
        PReg::new(n, RegClass::Float)
    }
    fn gpr(n: usize) -> PReg {
        PReg::new(n, RegClass::Int)
    }
    fn r(reg: PReg) -> Allocation {
        Allocation::reg(reg)
    }
    fn s(n: usize) -> Allocation {
        Allocation::stack(SpillSlot::new(n))
    }
    fn mv(from: Allocation, to: Allocation) -> Step {
        Step::Edit(from, to)
    }
    fn inst(operands: &[(OperandKind, Allocation)]) -> Step {
        Step::Inst(operands.to_vec(), PRegSet::default(), PRegSet::default())
    }
    fn call(clobbers: PRegSet) -> Step {
        Step::Inst(Vec::new(), clobbers, PRegSet::default())
    }
    fn call_with_temp(clobbers: PRegSet, temp: PReg) -> Step {
        Step::Inst(Vec::new(), clobbers, PRegSet::empty().with(temp))
    }
    const USE: OperandKind = OperandKind::Use;
    const DEF: OperandKind = OperandKind::Def;

    fn operands(list: &[(OperandKind, Allocation)]) -> (Vec<Operand>, Vec<Allocation>) {
        let class = |alloc: &Allocation| alloc.as_reg().map_or(RegClass::Float, |reg| reg.class());
        list.iter()
            .enumerate()
            .map(|(i, (kind, alloc))| {
                let vreg = VReg::new(i, class(alloc));
                let operand = match kind {
                    OperandKind::Use => Operand::any_use(vreg),
                    OperandKind::Def => Operand::any_def(vreg),
                };
                (operand, *alloc)
            })
            .unzip()
    }

    /// Run the rewrites over `steps`; an edit's index and an instruction's
    /// index are its position in `steps`.
    fn decide(steps: &[Step], insts: &mut Insts) -> ReloadFolds {
        let mut folds = ReloadFolds {
            fold: true,
            forward: true,
            ..ReloadFolds::default()
        };
        folds.start_block();
        for (i, step) in steps.iter().enumerate() {
            match step {
                Step::Edit(from, to) => folds.edit(i, *from, *to, insts),
                Step::Inst(list, clobbers, may_write) => {
                    let (operands, allocs) = operands(list);
                    let iix = InsnIndex::new(i);
                    folds.inst(iix, &operands, &allocs, *clobbers, *may_write, insts);
                }
                Step::Ret(read) => folds.ret(read, insts),
            }
        }
        folds
    }

    /// Check `folds` against `steps`; panics if a decision is wrong.
    fn check(steps: &[Step], insts: &Insts, folds: &ReloadFolds) {
        let mut check = ReloadCheck::default();
        for (i, step) in steps.iter().enumerate() {
            let iix = InsnIndex::new(i);
            match step {
                Step::Edit(from, to) => check.edit(*from, *to, folds.edits.get(&i).copied()),
                Step::Inst(list, clobbers, may_write) => {
                    let (operands, allocs) = operands(list);
                    let folded = folds
                        .folded
                        .iter()
                        .find(|(reader, ..)| *reader == iix)
                        .map(|&(_, reg, slot)| (reg, slot));
                    let keeps = |reg| insts.def_keeps_bits(iix, reg);
                    check.inst(
                        iix, &operands, &allocs, *clobbers, *may_write, folded, keeps,
                    );
                }
                Step::Ret(read) => check.ret(iix, read),
            }
        }
        check.finish();
    }

    /// The decisions for `steps`, checked.
    fn run(steps: &[Step], insts: &mut Insts) -> ReloadFolds {
        let folds = decide(steps, insts);
        check(steps, insts, &folds);
        folds
    }

    /// `steps` with these decisions seeded in place of the pass's own.
    fn seeded(
        steps: &[Step],
        insts: &Insts,
        edits: &[(usize, EditFold)],
        folded: &[(usize, PReg, usize)],
    ) {
        let folds = ReloadFolds {
            edits: edits.iter().copied().collect(),
            folded: folded
                .iter()
                .map(|&(inst, reg, slot)| (InsnIndex::new(inst), reg, SpillSlot::new(slot)))
                .collect(),
            ..ReloadFolds::default()
        };
        check(steps, insts, &folds);
    }

    /// A reload, its foldable reader and the next write of the register.
    fn folding_block() -> (Vec<Step>, Insts) {
        let steps = vec![
            mv(s(0), r(xmm(1))),
            inst(&[(USE, r(xmm(1))), (DEF, r(xmm(2)))]),
            inst(&[(DEF, r(xmm(1)))]),
        ];
        let insts = Insts {
            foldable: vec![(1, xmm(1))],
            ..Insts::default()
        };
        (steps, insts)
    }

    #[test]
    fn folds_a_reload_read_once() {
        let (steps, mut insts) = folding_block();
        let folds = run(&steps, &mut insts);
        assert_eq!(folds.edits.get(&0), Some(&EditFold::Skip));
        assert_eq!(folds.folded.len(), 1);
        // The same decisions, seeded, pass the checker.
        seeded(&steps, &insts, &[(0, EditFold::Skip)], &[(1, xmm(1), 0)]);
    }

    #[test]
    fn keeps_a_reload_whose_slot_is_rewritten_before_the_reader() {
        // An edit stores to the slot between the reload and its reader.
        let (mut steps, mut insts) = folding_block();
        steps.insert(1, mv(r(xmm(5)), s(0)));
        insts.foldable = vec![(2, xmm(1))];
        let folds = run(&steps, &mut insts);
        assert!(folds.edits.is_empty() && folds.folded.is_empty());

        // The same with an instruction that defines the slot.
        let (mut steps, mut insts) = folding_block();
        steps.insert(1, inst(&[(DEF, s(0))]));
        insts.foldable = vec![(2, xmm(1))];
        let folds = run(&steps, &mut insts);
        assert!(folds.edits.is_empty() && folds.folded.is_empty());
    }

    #[test]
    #[should_panic(expected = "reload folding changed")]
    fn checker_rejects_a_fold_across_a_slot_write() {
        let (mut steps, mut insts) = folding_block();
        steps.insert(1, mv(r(xmm(5)), s(0)));
        insts.foldable = vec![(2, xmm(1))];
        seeded(&steps, &insts, &[(0, EditFold::Skip)], &[(2, xmm(1), 0)]);
    }

    #[test]
    fn folds_when_the_slot_is_rewritten_after_the_reader() {
        let (mut steps, mut insts) = folding_block();
        steps.insert(2, mv(r(xmm(5)), s(0)));
        let folds = run(&steps, &mut insts);
        assert_eq!(folds.edits.get(&0), Some(&EditFold::Skip));
    }

    #[test]
    fn keeps_a_reload_an_edit_reads_after_the_reader() {
        let (mut steps, mut insts) = folding_block();
        steps.insert(2, mv(r(xmm(1)), s(3)));
        let folds = run(&steps, &mut insts);
        assert!(folds.edits.is_empty() && folds.folded.is_empty());

        // A copy to another register reads it too.
        let (mut steps, mut insts) = folding_block();
        steps.insert(2, mv(r(xmm(1)), r(xmm(7))));
        let folds = run(&steps, &mut insts);
        assert!(folds.edits.is_empty() && folds.folded.is_empty());
    }

    #[test]
    #[should_panic(expected = "reload folding changed")]
    fn checker_rejects_a_fold_before_an_edit_that_reads_the_register() {
        let (mut steps, insts) = folding_block();
        steps.insert(2, mv(r(xmm(1)), s(3)));
        seeded(&steps, &insts, &[(0, EditFold::Skip)], &[(1, xmm(1), 0)]);
    }

    #[test]
    fn keeps_a_reload_read_twice_or_live_out() {
        // A second instruction reads the register.
        let (mut steps, mut insts) = folding_block();
        steps.insert(2, inst(&[(USE, r(xmm(1)))]));
        let folds = run(&steps, &mut insts);
        assert!(folds.edits.is_empty());

        // The register is not written again in the block.
        let (mut steps, mut insts) = folding_block();
        steps.pop();
        let folds = run(&steps, &mut insts);
        assert!(folds.edits.is_empty());

        // The next def keeps some of the register's bits.
        let (steps, mut insts) = folding_block();
        insts.keeps = vec![(2, xmm(1))];
        let folds = run(&steps, &mut insts);
        assert!(folds.edits.is_empty());
    }

    #[test]
    #[should_panic(expected = "at the end of the block")]
    fn checker_rejects_a_fold_of_a_live_out_register() {
        let (mut steps, insts) = folding_block();
        steps.pop();
        seeded(&steps, &insts, &[(0, EditFold::Skip)], &[(1, xmm(1), 0)]);
    }

    #[test]
    #[should_panic(expected = "at the end of the block")]
    fn checker_rejects_a_fold_before_a_def_that_keeps_bits() {
        let (steps, mut insts) = folding_block();
        insts.keeps = vec![(2, xmm(1))];
        seeded(&steps, &insts, &[(0, EditFold::Skip)], &[(1, xmm(1), 0)]);
    }

    #[test]
    fn a_return_ends_the_life_of_registers_it_does_not_read() {
        let (mut steps, mut insts) = folding_block();
        steps.pop();
        steps.push(Step::Ret(vec![r(xmm(0))]));
        let folds = run(&steps, &mut insts);
        assert_eq!(folds.edits.get(&0), Some(&EditFold::Skip));

        let (mut steps, mut insts) = folding_block();
        steps.pop();
        steps.push(Step::Ret(vec![r(xmm(1))]));
        let folds = run(&steps, &mut insts);
        assert!(folds.edits.is_empty());
    }

    #[test]
    #[should_panic(expected = "read by the return")]
    fn checker_rejects_a_fold_of_a_returned_register() {
        let (mut steps, insts) = folding_block();
        steps.pop();
        steps.push(Step::Ret(vec![r(xmm(1))]));
        seeded(&steps, &insts, &[(0, EditFold::Skip)], &[(1, xmm(1), 0)]);
    }

    /// A store of `from` to slot 0, `between`, then a reload into `to`.
    fn forwarding_block(from: PReg, between: &[Step], to: PReg) -> Vec<Step> {
        let mut steps = vec![mv(r(from), s(0))];
        steps.extend_from_slice(between);
        steps.push(mv(s(0), r(to)));
        steps.push(inst(&[(USE, r(to))]));
        steps
    }

    #[test]
    fn forwards_a_stored_register() {
        let steps = forwarding_block(gpr(3), &[], gpr(6));
        let folds = run(&steps, &mut Insts::default());
        assert_eq!(folds.edits.get(&1), Some(&EditFold::MoveFrom(gpr(3))));

        let steps = forwarding_block(gpr(3), &[], gpr(3));
        let folds = run(&steps, &mut Insts::default());
        assert_eq!(folds.edits.get(&1), Some(&EditFold::Skip));

        // Not from a register of another class.
        let steps = forwarding_block(gpr(3), &[], xmm(3));
        let folds = run(&steps, &mut Insts::default());
        assert!(folds.edits.is_empty());
    }

    #[test]
    fn a_def_on_the_stack_ends_forwarding_from_its_slot() {
        for to in [gpr(3), gpr(6)] {
            let steps = forwarding_block(gpr(3), &[inst(&[(DEF, s(0))])], to);
            let folds = run(&steps, &mut Insts::default());
            assert!(folds.edits.is_empty(), "reload into {to}");
        }
        // A def of another slot does not.
        let steps = forwarding_block(gpr(3), &[inst(&[(DEF, s(1))])], gpr(6));
        let folds = run(&steps, &mut Insts::default());
        assert_eq!(folds.edits.get(&2), Some(&EditFold::MoveFrom(gpr(3))));
    }

    #[test]
    #[should_panic(expected = "reload folding changed")]
    fn checker_rejects_forwarding_across_a_def_on_the_stack() {
        let steps = forwarding_block(gpr(3), &[inst(&[(DEF, s(0))])], gpr(6));
        seeded(
            &steps,
            &Insts::default(),
            &[(2, EditFold::MoveFrom(gpr(3)))],
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "reload folding changed")]
    fn checker_rejects_skipping_a_reload_across_a_def_on_the_stack() {
        let steps = forwarding_block(gpr(3), &[inst(&[(DEF, s(0))])], gpr(3));
        seeded(&steps, &Insts::default(), &[(2, EditFold::Skip)], &[]);
    }

    #[test]
    fn a_written_register_ends_forwarding_from_it() {
        // By a def, by a clobber and by an edit.
        let writes = [
            inst(&[(DEF, r(gpr(3)))]),
            call(PRegSet::empty().with(gpr(3))),
            mv(r(gpr(9)), r(gpr(3))),
        ];
        for write in writes {
            for to in [gpr(3), gpr(6)] {
                let steps = forwarding_block(gpr(3), &[write.clone()], to);
                let folds = run(&steps, &mut Insts::default());
                assert!(folds.edits.is_empty(), "reload into {to}");
            }
        }
        // A clobber of other registers does not.
        let steps = forwarding_block(gpr(3), &[call(PRegSet::empty().with(gpr(4)))], gpr(6));
        let folds = run(&steps, &mut Insts::default());
        assert_eq!(folds.edits.get(&2), Some(&EditFold::MoveFrom(gpr(3))));
    }

    #[test]
    fn a_call_temporary_ends_forwarding_and_keeps_a_reload() {
        // The call declares no clobbers but may write its temporary.
        let none = PRegSet::empty();
        for to in [gpr(11), gpr(6)] {
            let steps = forwarding_block(gpr(11), &[call_with_temp(none, gpr(11))], to);
            let folds = run(&steps, &mut Insts::default());
            assert!(folds.edits.is_empty(), "reload into {to}");
        }
        let steps = forwarding_block(gpr(3), &[call_with_temp(none, gpr(11))], gpr(6));
        let folds = run(&steps, &mut Insts::default());
        assert_eq!(folds.edits.get(&2), Some(&EditFold::MoveFrom(gpr(3))));

        // A reload into the temporary is not dead at the call, so a later
        // def does not commit the fold.
        let (mut steps, mut insts) = folding_block();
        steps[2] = call_with_temp(none, xmm(1));
        steps.push(inst(&[(DEF, r(xmm(1)))]));
        let folds = run(&steps, &mut insts);
        assert!(folds.edits.is_empty() && folds.folded.is_empty());
    }

    #[test]
    #[should_panic(expected = "may keep")]
    fn checker_rejects_a_fold_that_ends_at_a_call_temporary() {
        let (mut steps, insts) = folding_block();
        steps[2] = call_with_temp(PRegSet::empty(), xmm(1));
        steps.push(inst(&[(DEF, r(xmm(1)))]));
        seeded(&steps, &insts, &[(0, EditFold::Skip)], &[(1, xmm(1), 0)]);
    }

    #[test]
    #[should_panic(expected = "reload folding changed")]
    fn checker_rejects_forwarding_from_a_call_temporary() {
        let steps = forwarding_block(
            gpr(11),
            &[call_with_temp(PRegSet::empty(), gpr(11))],
            gpr(6),
        );
        seeded(
            &steps,
            &Insts::default(),
            &[(2, EditFold::MoveFrom(gpr(11)))],
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "reload folding changed")]
    fn checker_rejects_forwarding_from_a_clobbered_register() {
        let steps = forwarding_block(gpr(3), &[call(PRegSet::empty().with(gpr(3)))], gpr(6));
        seeded(
            &steps,
            &Insts::default(),
            &[(2, EditFold::MoveFrom(gpr(3)))],
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "reload folding changed")]
    fn checker_rejects_a_copy_from_the_wrong_register() {
        let steps = forwarding_block(gpr(3), &[], gpr(6));
        seeded(
            &steps,
            &Insts::default(),
            &[(1, EditFold::MoveFrom(gpr(4)))],
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "reload folding changed")]
    fn checker_rejects_a_dropped_reload() {
        let steps = forwarding_block(gpr(3), &[], gpr(6));
        seeded(&steps, &Insts::default(), &[(1, EditFold::Skip)], &[]);
    }

    #[test]
    #[should_panic(expected = "reload folding changed")]
    fn checker_rejects_a_fold_that_reads_the_wrong_slot() {
        let (steps, insts) = folding_block();
        seeded(&steps, &insts, &[(0, EditFold::Skip)], &[(1, xmm(1), 4)]);
    }
}
