#!/usr/bin/env python3
"""Generate the x86-reload-fold-generated-*.clif run tests.

Each function is a random straight-line program over more f64, i64 and
vector values than there are registers, all of which stay live to the end,
with calls, `try_call`, `br_table`, trapping instructions and scalar
instructions that keep the upper bits of their destination mixed in. The
expected results come from evaluating the same program here.

    python3 x86-reload-fold-gen.py

rewrites the files beside this script.
"""

import math
import os
import random
import struct

MASK = (1 << 64) - 1


def signed(x):
    x &= MASK
    return x - (1 << 64) if x >> 63 else x


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def fhex(x):
    if x == 0.0:
        return "-0x0.0" if math.copysign(1.0, x) < 0 else "0x0.0"
    return x.hex().replace("p+", "p")


def nearest(x):
    r = float(round(x))
    return math.copysign(r, x)


def ok(x):
    return math.isfinite(x) and abs(x) < 1e100


class Gen:
    def __init__(self, rng, name, runs, conv="", sse41=False, wide=False):
        self.rng = rng
        self.name = name
        self.conv = conv
        self.sse41 = sse41
        self.wide = wide
        self.runs = runs
        self.lines = []
        self.next_value = 4
        self.next_block = 1
        # name -> one Python value per run
        self.f = {"v0": [r[0] for r in runs], "v1": [r[1] for r in runs]}
        self.i = {"v2": [r[2] for r in runs], "v3": [r[3] for r in runs]}
        self.x2 = {}
        self.x4 = {}

    def value(self):
        self.next_value += 1
        return f"v{self.next_value - 1}"

    def block(self):
        self.next_block += 1
        return f"block{self.next_block - 1}"

    def emit(self, text):
        self.lines.append("    " + text)

    def pick(self, pool):
        names = list(pool)
        # Favor old values to keep live ranges long.
        return self.rng.choice(names[: max(2, len(names) // 2)] + names)

    def float_op(self):
        a, b = self.pick(self.f), self.pick(self.f)
        ops = ["fadd", "fsub", "fmul", "fadd", "fsub", "sqrt", "fcvt", "fpromote"]
        if self.sse41:
            ops += ["ceil", "floor", "trunc", "nearest"]
        op = self.rng.choice(ops)
        va, vb = self.f[a], self.f[b]
        if op in ("fadd", "fsub", "fmul"):
            fn = {"fadd": lambda x, y: x + y, "fsub": lambda x, y: x - y, "fmul": lambda x, y: x * y}[op]
            res = [fn(x, y) for x, y in zip(va, vb)]
            if not all(ok(x) for x in res):
                return
            v = self.value()
            self.emit(f"{v} = {op} {a}, {b}")
        elif op == "sqrt":
            res = [math.sqrt(abs(x)) for x in va]
            t, v = self.value(), self.value()
            self.emit(f"{t} = fabs {a}")
            self.emit(f"{v} = sqrt {t}")
        elif op == "fcvt":
            i = self.pick(self.i)
            res = [float(signed(x)) for x in self.i[i]]
            v = self.value()
            self.emit(f"{v} = fcvt_from_sint.f64 {i}")
        elif op == "fpromote":
            res = [f32(x) for x in va]
            if not all(ok(x) for x in res):
                return
            t, v = self.value(), self.value()
            self.emit(f"{t} = fdemote.f32 {a}")
            self.emit(f"{v} = fpromote.f64 {t}")
        else:
            fn = {"ceil": math.ceil, "floor": math.floor, "trunc": math.trunc}.get(op)
            if fn:
                res = [math.copysign(float(fn(x)), x) for x in va]
            else:
                res = [nearest(x) for x in va]
            v = self.value()
            self.emit(f"{v} = {op} {a}")
        self.f[v] = res

    def int_op(self):
        a, b = self.pick(self.i), self.pick(self.i)
        op = self.rng.choice(["iadd", "isub", "imul", "bxor", "udiv", "trapz"])
        va, vb = self.i[a], self.i[b]
        if op == "udiv":
            c, t, v = self.value(), self.value(), self.value()
            self.emit(f"{c} = iconst.i64 1")
            self.emit(f"{t} = bor {b}, {c}")
            self.emit(f"{v} = udiv {a}, {t}")
            res = [(x & MASK) // ((y | 1) & MASK) for x, y in zip(va, vb)]
        elif op == "trapz":
            c, t, v = self.value(), self.value(), self.value()
            self.emit(f"{c} = iconst.i64 1")
            self.emit(f"{t} = bor {b}, {c}")
            self.emit(f"trapz {t}, user1")
            self.emit(f"{v} = iadd {a}, {t}")
            res = [(x + (y | 1)) & MASK for x, y in zip(va, vb)]
        else:
            fn = {
                "iadd": lambda x, y: x + y,
                "isub": lambda x, y: x - y,
                "imul": lambda x, y: x * y,
                "bxor": lambda x, y: x ^ y,
            }[op]
            v = self.value()
            self.emit(f"{v} = {op} {a}, {b}")
            res = [fn(x, y) & MASK for x, y in zip(va, vb)]
        self.i[v] = res

    def vector_op(self):
        lanes = self.rng.choice([2, 4])
        pool = self.x2 if lanes == 2 else self.x4
        if len(pool) < 2 or self.rng.random() < 0.3:
            srcs = [self.pick(self.f) for _ in range(lanes)]
            v = self.value()
            self.emit(f"{v} = splat.f64x{lanes} {srcs[0]}")
            for lane in range(1, lanes):
                n = self.value()
                self.emit(f"{n} = insertlane {v}, {srcs[lane]}, {lane}")
                v = n
            pool[v] = [[self.f[s][r] for s in srcs] for r in range(len(self.runs))]
            return
        a, b = self.pick(pool), self.pick(pool)
        op = self.rng.choice(["fadd", "fsub", "fmul"])
        fn = {"fadd": lambda x, y: x + y, "fsub": lambda x, y: x - y, "fmul": lambda x, y: x * y}[op]
        res = [[fn(x, y) for x, y in zip(la, lb)] for la, lb in zip(pool[a], pool[b])]
        if not all(ok(x) for lanes_ in res for x in lanes_):
            return
        v = self.value()
        self.emit(f"{v} = {op} {a}, {b}")
        pool[v] = res

    def call_op(self):
        kind = self.rng.choice(["f", "i", "try"])
        if kind == "f":
            a, v = self.pick(self.f), self.value()
            self.emit(f"{v} = call fn0({a})")
            self.f[v] = self.f[a]
        elif kind == "i":
            a, v = self.pick(self.i), self.value()
            self.emit(f"{v} = call fn1({a})")
            self.i[v] = self.i[a]
        else:
            a, v, e = self.pick(self.f), self.value(), self.value()
            normal, handler = self.block(), self.block()
            self.emit(f"try_call fn2({a}), sig0, {normal}({v}), [default: {handler}(exn0)]")
            self.lines[-1] = self.lines[-1].replace(f"{normal}({v})", f"{normal}(ret0)")
            self.lines.append("")
            self.lines.append(f"{handler}({e}: i64):")
            self.emit(f"trap user2")
            self.lines.append("")
            self.lines.append(f"{normal}({v}: f64):")
            self.f[v] = self.f[a]

    def table_op(self):
        i, a, b = self.pick(self.i), self.pick(self.f), self.pick(self.f)
        c, t, idx, v = self.value(), self.value(), self.value(), self.value()
        blocks = [self.block() for _ in range(3)]
        merge = self.block()
        self.emit(f"{c} = iconst.i64 3")
        self.emit(f"{t} = band {i}, {c}")
        self.emit(f"{idx} = ireduce.i32 {t}")
        self.emit(f"br_table {idx}, {blocks[2]}, [{blocks[0]}, {blocks[1]}]")
        fns = [
            ("fadd", lambda x, y: x + y),
            ("fsub", lambda x, y: x - y),
            ("fadd", lambda x, y: y + y),
        ]
        operands = [(a, b), (a, b), (b, b)]
        for blk, (op, _), (x, y) in zip(blocks, fns, operands):
            n = self.value()
            self.lines.append("")
            self.lines.append(f"{blk}:")
            self.emit(f"{n} = {op} {x}, {y}")
            self.emit(f"jump {merge}({n})")
        self.lines.append("")
        self.lines.append(f"{merge}({v}: f64):")
        res = []
        for r in range(len(self.runs)):
            which = min(self.i[i][r] & 3, 2)
            res.append(fns[which][1](self.f[a][r], self.f[b][r]))
        self.f[v] = res

    def finish(self, tail):
        acc = "v0"
        total = list(self.f["v0"])
        for name, vals in list(self.f.items())[1:]:
            n = self.value()
            self.emit(f"{n} = fadd {acc}, {name}")
            total = [x + y for x, y in zip(total, vals)]
            acc = n
        for pool in (self.x2, self.x4):
            for name, vals in pool.items():
                for lane in range(len(vals[0])):
                    e, n = self.value(), self.value()
                    self.emit(f"{e} = extractlane {name}, {lane}")
                    self.emit(f"{n} = fadd {acc}, {e}")
                    total = [x + v[lane] for x, v in zip(total, vals)]
                    acc = n
        iacc = "v2"
        itotal = list(self.i["v2"])
        for name, vals in list(self.i.items())[1:]:
            n = self.value()
            self.emit(f"{n} = bxor {iacc}, {name}")
            itotal = [x ^ y for x, y in zip(itotal, vals)]
            iacc = n
        c, n = self.value(), self.value()
        self.emit(f"{c} = fcvt_from_sint.f64 {iacc}")
        self.emit(f"{n} = fadd {acc}, {c}")
        total = [x + float(signed(y)) for x, y in zip(total, itotal)]
        if tail:
            self.emit(f"return_call fn2({n})")
        else:
            self.emit(f"return {n}")
        return total

    def generate(self, ops, tail=False):
        for _ in range(ops):
            r = self.rng.random()
            if r < 0.45:
                self.float_op()
            elif r < 0.75:
                self.int_op()
            elif r < 0.87:
                self.call_op()
            elif r < 0.92:
                self.table_op()
            elif self.wide:
                self.vector_op()
            else:
                self.float_op()
        total = self.finish(tail)
        assert all(math.isfinite(x) for x in total)
        conv = f" {self.conv}" if self.conv else ""
        idconv = conv if self.conv == "windows_fastcall" else ""
        out = [
            f"function %{self.name}(f64, f64, i64, i64) -> f64{conv} {{",
            "    sig0 = (f64) -> f64 tail",
            f"    fn0 = %id(f64) -> f64{idconv}" if not idconv else f"    fn0 = %id_fastcall(f64) -> f64{idconv}",
            "    fn1 = %id_int(i64) -> i64",
            "    fn2 = %id_tail(f64) -> f64 tail",
            "",
            "block0(v0: f64, v1: f64, v2: i64, v3: i64):",
        ]
        out += self.lines
        out.append("}")
        name = self.name
        if self.conv:
            # The run directive calls with the default convention.
            name = self.name + "_entry"
            out += [
                "",
                f"function %{name}(f64, f64, i64, i64) -> f64 {{",
                f"    fn0 = %{self.name}(f64, f64, i64, i64) -> f64{conv}",
                "",
                "block0(v0: f64, v1: f64, v2: i64, v3: i64):",
                "    v4 = call fn0(v0, v1, v2, v3)",
                "    return v4",
                "}",
            ]
        for run, result in zip(self.runs, total):
            args = f"{fhex(run[0])}, {fhex(run[1])}, {signed(run[2])}, {signed(run[3])}"
            out.append(f"; run: %{name}({args}) == {fhex(result)}")
        out.append("")
        return "\n".join(out)


HELPERS = """function %id(f64) -> f64 {
block0(v0: f64):
    return v0
}

function %id_fastcall(f64) -> f64 windows_fastcall {
block0(v0: f64):
    return v0
}

function %id_int(i64) -> i64 {
block0(v0: i64):
    return v0
}

function %id_tail(f64) -> f64 tail {
block0(v0: f64):
    return v0
}
"""

RUNS = [
    (1.5, -2.25, 7, 1000003),
    (-0.375, 12.0, MASK - 4, 3),
    (1024.0, 0.001953125, 0x123456789, MASK - 0xFFFF),
]


def header(targets):
    out = [
        "test run",
        "set preserve_frame_pointers=true",
        "set regalloc_fold_reloads=true",
        "set regalloc_forward_spills=true",
        "set regalloc_checker=true",
    ]
    out += targets
    out.append("set regalloc_algorithm=single_pass")
    out += targets
    out += [
        "",
        ";; Generated by x86-reload-fold-gen.py; edit that script, not this file.",
        ";;",
        ";; Each function holds more values than there are registers and keeps",
        ";; them live to its end, so the allocator spills, reloads and evicts",
        ";; throughout. Expected results were computed by the generator.",
        "",
        HELPERS,
    ]
    return "\n".join(out)


def write(path, targets, functions):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, path), "w") as f:
        f.write(header(targets))
        f.write("\n".join(functions))


def main():
    rng = random.Random(930)
    sse = ["target x86_64", "target x86_64 has_sse41", "target x86_64 has_sse41 has_avx"]
    funcs = []
    for n in range(8):
        funcs.append(Gen(rng, f"mixed{n}", RUNS).generate(60 + 10 * n))
    for n in range(3):
        funcs.append(Gen(rng, f"fastcall{n}", RUNS, conv="windows_fastcall").generate(70))
    for n in range(3):
        funcs.append(Gen(rng, f"tail{n}", RUNS, conv="tail").generate(70, tail=True))
    write("x86-reload-fold-generated.clif", sse, funcs)

    funcs = []
    for n in range(6):
        funcs.append(Gen(rng, f"rounding{n}", RUNS, sse41=True).generate(80))
    write(
        "x86-reload-fold-generated-sse41.clif",
        ["target x86_64 has_sse41", "target x86_64 has_sse41 has_avx"],
        funcs,
    )

    funcs = []
    for n in range(6):
        funcs.append(Gen(rng, f"wide{n}", RUNS, sse41=True, wide=True).generate(90))
    write(
        "x86-reload-fold-generated-ymm.clif",
        ["target x86_64 has_sse41 has_avx has_avx2"],
        funcs,
    )


if __name__ == "__main__":
    main()
