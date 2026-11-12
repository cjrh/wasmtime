//! A randomized differential test of the x64 reload rewrites
//! (`regalloc_fold_reloads`, `regalloc_forward_spills`).
//!
//! This is not a coverage-guided fuzzer: it feeds pseudo-random bytes to
//! `cranelift-fuzzgen`, the generator behind the `cranelift-fuzzgen` fuzz
//! target, mutating inputs the generator accepted and preferring larger
//! functions. It needs neither `cargo fuzz` nor a nightly toolchain. Each
//! generated test case is compiled for the host with the rewrites off and
//! with them on, with the post-rewrite check enabled, and every input is
//! run in the interpreter and in both compiled versions. All three results
//! must agree bit for bit.
//!
//! Cases rotate through both register allocators, three ISA flag sets
//! (baseline SSE2, SSE4.1, AVX2) and three rewrite choices (both, operand
//! folding only, forwarding only).
//!
//! ```text
//! cargo run --release -p cranelift-filetests --example reload_fold_fuzz -- \
//!     SECONDS [THREADS] [SEED]
//! ```

use arbitrary::Unstructured;
use cranelift_codegen::Context;
use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::{Function, LibCall, Signature, UserExternalName, UserFuncName};
use cranelift_codegen::isa::{self, OwnedTargetIsa};
use cranelift_codegen::settings::{self, Configurable, Flags};
use cranelift_control::ControlPlane;
use cranelift_filetests::function_runner::TestFileCompiler;
use cranelift_fuzzgen::{FuzzGen, PrintableTestCase, TestCaseInput};
use cranelift_interpreter::environment::{FuncIndex, FunctionStore};
use cranelift_interpreter::interpreter::{
    Interpreter, InterpreterError, InterpreterState, LibCallValues,
};
use cranelift_interpreter::step::ControlFlow;
use smallvec::smallvec;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

const INTERPRETER_FUEL: u64 = 4096;

const ALLOWED_LIBCALLS: &[LibCall] = &[
    LibCall::CeilF32,
    LibCall::CeilF64,
    LibCall::FloorF32,
    LibCall::FloorF64,
    LibCall::TruncF32,
    LibCall::TruncF64,
];

const ISA_FLAG_SETS: [(&str, &[&str]); 3] = [
    ("sse2", &[]),
    ("sse41", &["has_sse3", "has_ssse3", "has_sse41"]),
    (
        "avx2",
        &[
            "has_sse3",
            "has_ssse3",
            "has_sse41",
            "has_sse42",
            "has_popcnt",
            "has_avx",
            "has_avx2",
            "has_fma",
            "has_bmi1",
            "has_bmi2",
            "has_lzcnt",
        ],
    ),
];

/// The rewrites enabled in the build compared against the plain one.
const REWRITES: [(&str, bool, bool); 3] = [
    ("both", true, true),
    ("fold", true, false),
    ("forward", false, true),
];

const ALGORITHMS: [&str; 2] = ["backtracking", "single_pass"];

#[derive(Default)]
struct Counters {
    cases: AtomicU64,
    invalid: AtomicU64,
    compiled: AtomicU64,
    /// Compiled cases whose machine code differs with the rewrites on.
    changed: AtomicU64,
    timeouts: AtomicU64,
    traps: AtomicU64,
    runs: AtomicU64,
    failures: AtomicU64,
    /// Compiled cases per (algorithm, ISA flag set).
    per_config: [AtomicU64; 6],
    /// Compiled cases per rewrite choice.
    per_rewrite: [AtomicU64; 3],
    /// IR instructions in the compiled cases.
    insts: AtomicU64,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn build_isa(
    base: &Flags,
    algorithm: &str,
    isa_flags: &[&str],
    fold: bool,
    forward: bool,
) -> OwnedTargetIsa {
    let mut builder = settings::builder();
    for value in base.iter() {
        builder.set(value.name, &value.value_string()).unwrap();
    }
    builder.set("regalloc_algorithm", algorithm).unwrap();
    builder.enable("regalloc_checker").unwrap();
    builder
        .set("regalloc_fold_reloads", if fold { "true" } else { "false" })
        .unwrap();
    builder
        .set(
            "regalloc_forward_spills",
            if forward { "true" } else { "false" },
        )
        .unwrap();
    let mut isa_builder = isa::lookup(target_lexicon::Triple::host()).unwrap();
    for flag in isa_flags {
        isa_builder.enable(flag).unwrap();
    }
    isa_builder.finish(Flags::new(builder)).unwrap()
}

struct Case {
    functions: Vec<Function>,
    inputs: Vec<TestCaseInput>,
    plain: OwnedTargetIsa,
    rewritten: OwnedTargetIsa,
}

fn generate(bytes: &[u8], config: u64) -> anyhow::Result<Case> {
    let mut u = Unstructured::new(bytes);
    let mut generator = FuzzGen::new(&mut u);
    let algorithm = ALGORITHMS[(config % 2) as usize];
    let (_, isa_flags) = ISA_FLAG_SETS[(config / 2 % 3) as usize];
    let (_, fold, forward) = REWRITES[(config / 6 % 3) as usize];

    let base = generator.generate_flags(target_lexicon::Triple::host().architecture)?;
    let plain = build_isa(&base, algorithm, isa_flags, false, false);
    let rewritten = build_isa(&base, algorithm, isa_flags, fold, forward);

    let func_count = generator
        .u
        .int_in_range(generator.config.testcase_funcs.clone())?;
    let mut functions: Vec<Function> = Vec::with_capacity(func_count);
    for i in (0..func_count).rev() {
        let name = UserFuncName::user(1, i as u32);
        let usercalls: Vec<(UserExternalName, Signature)> = functions
            .iter()
            .map(|f| {
                (
                    f.name.get_user().unwrap().clone(),
                    f.stencil.signature.clone(),
                )
            })
            .collect();
        functions.push(generator.generate_func(
            name,
            rewritten.clone(),
            usercalls,
            ALLOWED_LIBCALLS.to_vec(),
        )?);
    }
    functions.reverse();
    let inputs = generator.generate_test_inputs(&functions[0].signature)?;
    Ok(Case {
        functions,
        inputs,
        plain,
        rewritten,
    })
}

fn interpret(functions: &[Function], args: &[DataValue]) -> Result<Option<Vec<DataValue>>, bool> {
    let mut env = FunctionStore::default();
    for func in functions {
        env.add(func.name.to_string(), func);
    }
    let state = InterpreterState::default()
        .with_function_store(env)
        .with_libcall_handler(|libcall: LibCall, args: LibCallValues| {
            use LibCall::*;
            Ok(smallvec![match (libcall, &args[..]) {
                (CeilF32, [DataValue::F32(a)]) => DataValue::F32(a.ceil()),
                (CeilF64, [DataValue::F64(a)]) => DataValue::F64(a.ceil()),
                (FloorF32, [DataValue::F32(a)]) => DataValue::F32(a.floor()),
                (FloorF64, [DataValue::F64(a)]) => DataValue::F64(a.floor()),
                (TruncF32, [DataValue::F32(a)]) => DataValue::F32(a.trunc()),
                (TruncF64, [DataValue::F64(a)]) => DataValue::F64(a.trunc()),
                _ => unreachable!(),
            }])
        });
    let mut interpreter = Interpreter::new(state).with_fuel(Some(INTERPRETER_FUEL));
    match interpreter.call_by_index(FuncIndex::from_u32(0), args) {
        Ok(ControlFlow::Return(results)) => Ok(Some(results.to_vec())),
        // A trap: skip this input.
        Ok(ControlFlow::Trap(_)) => Ok(None),
        // Probably an infinite loop: drop the case.
        Err(InterpreterError::FuelExhausted) => Err(true),
        other => panic!("interpreter failed: {other:?}"),
    }
}

fn same(a: &[DataValue], b: &[DataValue]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.bitwise_eq(b))
}

fn code_differs(case: &Case) -> bool {
    case.functions.iter().any(|func| {
        let compile = |isa: &OwnedTargetIsa| {
            let mut ctx = Context::for_function(func.clone());
            ctx.compile(isa.as_ref(), &mut ControlPlane::default())
                .map(|code| code.code_buffer().to_vec())
                .ok()
        };
        compile(&case.plain) != compile(&case.rewritten)
    })
}

fn check(case: &Case, counters: &Counters) {
    let planes = || vec![ControlPlane::default(); case.functions.len()];
    let mut plain = TestFileCompiler::new(case.plain.clone());
    plain.add_functions(&case.functions, planes()).unwrap();
    let plain = plain.compile().unwrap();
    let mut rewritten = TestFileCompiler::new(case.rewritten.clone());
    rewritten.add_functions(&case.functions, planes()).unwrap();
    let rewritten = rewritten.compile().unwrap();
    counters.compiled.fetch_add(1, Relaxed);
    if code_differs(case) {
        counters.changed.fetch_add(1, Relaxed);
    }

    let main = &case.functions[0];
    let plain_entry = plain.get_trampoline(main).unwrap();
    let rewritten_entry = rewritten.get_trampoline(main).unwrap();
    for args in &case.inputs {
        let expected = match interpret(&case.functions, args) {
            Ok(Some(results)) => results,
            Ok(None) => {
                counters.traps.fetch_add(1, Relaxed);
                continue;
            }
            Err(_) => {
                counters.timeouts.fetch_add(1, Relaxed);
                return;
            }
        };
        let without = plain_entry.call(&plain, args);
        let with = rewritten_entry.call(&rewritten, args);
        counters.runs.fetch_add(1, Relaxed);
        assert!(
            same(&with, &without),
            "rewrites changed the result for {args:?}: {with:?}, without {without:?}"
        );
        assert!(
            same(&with, &expected),
            "native result for {args:?} is {with:?}, interpreter gives {expected:?}"
        );
    }
}

/// Total instruction count of a case; the search below favours larger cases,
/// which spill more.
fn size(case: &Case) -> usize {
    case.functions
        .iter()
        .map(|f| {
            f.layout
                .blocks()
                .map(|b| f.layout.block_insts(b).count())
                .sum::<usize>()
        })
        .sum()
}

const POOL: usize = 256;
const MAX_SIZE: usize = 4000;

/// Derives the next input. Uniformly random bytes rarely get past the
/// generator (any opcode unsupported by the target rejects the case) and the
/// survivors are tiny, so most inputs mutate an earlier accepted input.
fn next_input(rng: &mut Rng, pool: &[(Vec<u8>, usize)], bytes: &mut Vec<u8>) {
    bytes.clear();
    let choice = rng.next() % 16;
    if pool.is_empty() || choice == 0 {
        let len = 256 + (rng.next() % 8192) as usize;
        while bytes.len() < len {
            bytes.extend_from_slice(&rng.next().to_le_bytes());
        }
        return;
    }
    let parent = &pool[(rng.next() % pool.len() as u64) as usize].0;
    bytes.extend_from_slice(parent);
    if choice < 8 {
        // Change a few bytes.
        for _ in 0..1 + rng.next() % 4 {
            let at = (rng.next() % bytes.len() as u64) as usize;
            bytes[at] = rng.next() as u8;
        }
    } else if choice < 13 {
        // Keep a prefix and replace the rest.
        let keep = (rng.next() % bytes.len() as u64) as usize;
        let len = (keep + 64 + (rng.next() % 4096) as usize).min(16384);
        bytes.truncate(keep);
        while bytes.len() < len {
            bytes.extend_from_slice(&rng.next().to_le_bytes());
        }
    } else {
        // Splice in a piece of another accepted input.
        let other = &pool[(rng.next() % pool.len() as u64) as usize].0;
        let at = (rng.next() % bytes.len() as u64) as usize;
        let from = (rng.next() % other.len() as u64) as usize;
        bytes.truncate(at);
        bytes.extend_from_slice(&other[from..]);
        bytes.truncate(16384);
    }
}

fn worker(seed: u64, thread: u64, deadline: Instant, counters: &Counters) {
    let mut rng = Rng(seed ^ (thread + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut bytes = Vec::new();
    let mut pool: Vec<(Vec<u8>, usize)> = Vec::new();
    while Instant::now() < deadline {
        let case_number = counters.cases.fetch_add(1, Relaxed);
        next_input(&mut rng, &pool, &mut bytes);
        let config = rng.next() % 18;
        let Ok(case) = generate(&bytes, config) else {
            counters.invalid.fetch_add(1, Relaxed);
            continue;
        };
        let size = size(&case);
        if size > MAX_SIZE {
            counters.invalid.fetch_add(1, Relaxed);
            continue;
        }
        counters.insts.fetch_add(size as u64, Relaxed);
        if pool.len() < POOL {
            pool.push((bytes.clone(), size));
        } else {
            let slot = (rng.next() % POOL as u64) as usize;
            if size * 5 >= pool[slot].1 * 4 {
                pool[slot] = (bytes.clone(), size);
            }
        }
        counters.per_config[(config % 6) as usize].fetch_add(1, Relaxed);
        counters.per_rewrite[(config / 6 % 3) as usize].fetch_add(1, Relaxed);
        let result = panic::catch_unwind(AssertUnwindSafe(|| check(&case, counters)));
        if let Err(payload) = result {
            let failures = counters.failures.fetch_add(1, Relaxed);
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            eprintln!(
                "FAILURE in case {case_number} (thread {thread}, allocator {}, isa {}, rewrites {}): {message}",
                ALGORITHMS[(config % 2) as usize],
                ISA_FLAG_SETS[(config / 2 % 3) as usize].0,
                REWRITES[(config / 6 % 3) as usize].0,
            );
            if failures < 5 {
                let path = format!("reload-fold-fuzz-failure-{case_number}.clif");
                let text = format!(
                    "{:?}",
                    PrintableTestCase::run(&case.rewritten, &case.functions, &case.inputs)
                );
                let _ = std::fs::write(&path, text);
                eprintln!("wrote {path}");
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let seconds: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(60);
    let threads: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(930);

    let counters = Counters::default();
    let deadline = Instant::now() + Duration::from_secs(seconds);
    std::thread::scope(|scope| {
        for thread in 0..threads {
            let counters = &counters;
            scope.spawn(move || worker(seed, thread, deadline, counters));
        }
    });

    let get = |counter: &AtomicU64| counter.load(Relaxed);
    println!("seed {seed}, {threads} threads, {seconds} s");
    println!("cases generated:        {}", get(&counters.cases));
    println!("  rejected by fuzzgen:  {}", get(&counters.invalid));
    println!("  compiled twice:       {}", get(&counters.compiled));
    println!("  code changed:         {}", get(&counters.changed));
    println!(
        "  mean IR instructions: {}",
        get(&counters.insts) / get(&counters.compiled).max(1)
    );
    println!("  dropped on timeout:   {}", get(&counters.timeouts));
    println!("inputs trapping:        {}", get(&counters.traps));
    println!("inputs compared:        {}", get(&counters.runs));
    for (i, counter) in counters.per_config.iter().enumerate() {
        println!(
            "  {} {}: {} cases",
            ALGORITHMS[i % 2],
            ISA_FLAG_SETS[i / 2].0,
            get(counter)
        );
    }
    for (i, counter) in counters.per_rewrite.iter().enumerate() {
        println!("  rewrites {}: {} cases", REWRITES[i].0, get(counter));
    }
    println!("failures:               {}", get(&counters.failures));
    if get(&counters.failures) != 0 {
        std::process::exit(1);
    }
}
