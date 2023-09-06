import argparse
import collections
import dataclasses
import enum
import pathlib
import shlex
import string
import subprocess
import sys


class TokenType(enum.Enum):
    OPEN = enum.auto()
    SYMBOL = enum.auto()
    INT = enum.auto()
    CLOSE = enum.auto()


SYMBOL_FIRST = string.ascii_letters + "-_$@+-*/=<>:"
SYMBOL_CONT = SYMBOL_FIRST + string.digits


class Location(collections.namedtuple("Location", ["line", "column"])):
    def __str__(self) -> str:
        return f"{self.line}:{self.column}"


def resolve_offsets_to_locations(
    source: str, offsets: list[int]
) -> list[tuple[int, int]]:
    offsets = sorted(offsets)
    locations = []
    offset, column, line = 0, 1, 1
    while offset < len(source) and offsets:
        if offset == offsets[0]:
            locations.append(Location(line, column))
            offsets = offsets[1:]
        if source[offset] == "\n":
            line += 1
            column = 1
        else:
            column += 1
        offset += 1
    return locations


def print_table(table: list[list[str]]):
    if not len(table):
        return
    max_column_lengths = [0] * len(table[0])
    for row in table:
        for column, entry in enumerate(row):
            max_column_lengths[column] = max(len(entry), max_column_lengths[column])

    for row in table:
        print(
            "".join(
                ("{:%d}" % (max_column_lengths[column] + 1,)).format(entry)
                for column, entry in enumerate(row)
            )
        )


@dataclasses.dataclass
class Tokens:
    source: str
    types: list[TokenType]
    sources: list[str]
    offsets: list[int]

    def dump(self):
        locations = resolve_offsets_to_locations(self.source, self.offsets)
        table = [
            [str(ty), str(src), str(loc)]
            for ty, src, loc in zip(self.types, self.sources, locations)
        ]
        print_table(table)


def tokenize(source: str) -> Tokens:
    """Tokenize source into a list of tokens"""
    original_source = source
    types, sources, offsets = [], [], []
    offset = 0

    def append(ty: TokenType, token_source: str):
        nonlocal offset, source
        types.append(ty)
        sources.append(token_source)
        offsets.append(offset)
        offset += len(token_source)
        source = source[len(token_source) :]

    while source:
        while source[:1].isspace():
            offset += 1
            source = source[1:]
        if not source:
            break

        if source.startswith("("):
            append(TokenType.OPEN, "(")
            continue

        if source.startswith(")"):
            append(TokenType.CLOSE, ")")
            continue

        if (first := source[:1]) and first in SYMBOL_FIRST:
            n = 0
            while n < len(source) and source[n] in SYMBOL_CONT:
                n += 1
            append(TokenType.SYMBOL, source[:n])
            continue

        if source[:1].isdigit():
            n = 0
            while n < len(source) and source[n].isdigit():
                n += 1
            append(TokenType.INT, source[:n])
            continue

        print("Failed at source:" + source)
        raise NotImplementedError()

    return Tokens(original_source, types, sources, offsets)


class ParseState(enum.Enum):
    WAITING_FOR_CALLEE = enum.auto()
    INSIDE_EXPRESSION = enum.auto()


class OpType(enum.Enum):
    # params[0] - literal (token_id)
    SET_LITERAL = enum.auto()

    # params[0]  - syscall number
    # params[1:] - arguments
    SYSCALL = enum.auto()

    # params - values to add
    PLUS = enum.auto()


@dataclasses.dataclass
class Instruction:
    op: OpType
    target: int  # Target register
    params: list[int]


@dataclasses.dataclass
class IR:
    tokens: Tokens
    instructions: list[Instruction]

    def dump(self):
        for i in self.instructions:
            match i.op:
                case OpType.SET_LITERAL:
                    # TODO: Pretty print token
                    argument = self.tokens.sources[i.params[0]]
                    print(f"${i.target} = {argument}")

                case OpType.SYSCALL:
                    arguments = ", ".join(f"${p}" for p in i.params)
                    print(f"${i.target} = syscall {arguments}")

                case OpType.PLUS:
                    arguments = ", ".join(f"${p}" for p in i.params)
                    print(f"${i.target} = + {arguments}")

                case op:
                    assert False, f"unkown operation: {op}"


def parse(tokens: Tokens) -> IR:
    offset = 0
    expressions_before = []
    state = ParseState.INSIDE_EXPRESSION

    next_free_variable_index = 0

    def next_register() -> int:
        nonlocal next_free_variable_index
        t, next_free_variable_index = (
            next_free_variable_index,
            next_free_variable_index + 1,
        )
        return t

    expr_result_registers = [next_register()]
    expr_stack = [[]]
    ir = []

    while offset < len(tokens.types):
        match (state, tokens.types[offset]):
            case (ParseState.WAITING_FOR_CALLEE, TokenType.SYMBOL):
                expr_stack.append([offset])
                state = ParseState.INSIDE_EXPRESSION

            case (ParseState.INSIDE_EXPRESSION, TokenType.OPEN):
                register = next_register()
                expr_result_registers.append(register)
                expr_stack[-1].append(register)
                state = ParseState.WAITING_FOR_CALLEE

            case (ParseState.INSIDE_EXPRESSION, TokenType.INT):
                register = next_register()
                expr_stack[-1].append(register)
                ir.append(Instruction(OpType.SET_LITERAL, register, [offset]))

            case (ParseState.INSIDE_EXPRESSION, TokenType.CLOSE):
                register = expr_result_registers.pop()
                expr = expr_stack.pop()
                assert (
                    tokens.types[expr[0]] == TokenType.SYMBOL
                ), "other callee are not implemented yet"

                match tokens.sources[expr[0]]:
                    case "syscall":
                        ir.append(Instruction(OpType.SYSCALL, register, expr[1:]))
                    case "+":
                        ir.append(Instruction(OpType.PLUS, register, expr[1:]))
                    case src:
                        assert False, f"unknown expression type: {src}"

            case x:
                assert False, f"Invalid state & token pair: {x}"

        offset += 1

    return IR(tokens=tokens, instructions=ir)


def emit_assembly(ir: IR, output_path: pathlib.Path):
    """
    As a first attempt we assume the dumb strategy.
    Since syscall requires specific registers, we reserve them for it's usage.
    All other x86-64 registers are free to use.

              num   arg0   arg1   arg2   arg3   arg4  arg5
    syscall: "rax", "rdi", "rsi", "rdx", "r10", "r8", "r9"

    remaining: rbx, rcx, r10..r15
    """

    syscall_registers = ["rax", "rdi", "rsi", "rdx", "r10", "r8", "r9"]
    free_registers = ["rbx", "rcx"] + list(f"r{n}" for n in range(10, 16))

    with open(output_path, "w") as out:
        print("format ELF64 executable 3", file=out)
        print("segment readable executable", file=out)

        register_to_machine_register = {}

        for i in ir.instructions:
            print(f"  ;; {i}", file=out)
            match i.op:
                case OpType.SET_LITERAL:
                    machine_register = free_registers.pop()
                    register_to_machine_register[i.target] = machine_register
                    # TODO: Some normalization of passed literal into a form friendly for FASM
                    argument = ir.tokens.sources[i.params[0]]
                    print(f"  mov {machine_register}, {argument}", file=out)

                case OpType.PLUS:
                    machine_register = free_registers.pop()
                    register_to_machine_register[i.target] = machine_register
                    if len(i.params) == 0:
                        print(f"  xor {machine_register}, {machine_register}", file=out)
                    else:
                        print(
                            f"  mov {machine_register}, {register_to_machine_register[i.params[0]]}",
                            file=out,
                        )
                        for register in i.params[1:]:
                            print(
                                f"  add {machine_register}, {register_to_machine_register[register]}",
                                file=out,
                            )

                case OpType.SYSCALL:
                    machine_register = free_registers.pop()
                    register_to_machine_register[i.target] = machine_register
                    for m_reg, reg in zip(syscall_registers, i.params):
                        print(
                            f"  mov {m_reg}, {register_to_machine_register[reg]}",
                            file=out,
                        )
                    print(f"  syscall", file=out)
                    print(f"  mov {machine_register}, rax", file=out)

                case op:
                    assert False, f"Unknown op: {op}"


def cmd(*args):
    print("[CMD] " + shlex.join(args), flush=True)
    subprocess.run(args, check=True)


def assemble(asm_path: pathlib.Path, output_path: pathlib.Path):
    cmd("fasm", "-m", "524288", asm_path, output_path)
    cmd("chmod", "+x", output_path)


def compile_listy(
    *,
    source_path: pathlib.Path,
    asm_path: pathlib.Path,
    output_path: pathlib.Path,
    print_tokens=False,
    print_ir=False,
):
    with open(source_path) as source_file:
        source = source_file.read()

    tokens = tokenize(source)
    if print_tokens:
        tokens.dump()

    ir = parse(tokens)
    if print_ir:
        ir.dump()

    emit_assembly(ir, asm_path)
    assemble(asm_path, output_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source_file", nargs=1, help="Source file with Listy code")
    p.add_argument(
        "--dump-tokens", action="store_true", help="Dump tokens of a program"
    )
    p.add_argument("--dump-ir", action="store_true", help="Dump IR")
    p.add_argument("-o", "--output", type=str, help="Path to output file")

    args = p.parse_args()
    source_path = pathlib.Path(args.source_file[0])
    asm_path = source_path.stem + ".fasm"  # TODO: Parameter
    output_path = source_path.stem if args.output is None else args.output

    compile_listy(
        source_path=source_path,
        asm_path=asm_path,
        output_path=output_path,
        print_tokens=args.dump_tokens,
        print_ir=args.dump_ir,
    )


if __name__ == "__main__":
    main()
