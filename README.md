# Listy

__THIS LANGUAGE IS WIP AND ALWAYS WILL BE__

A programming language that is meant to teach me how to make [self-compiling compiler](https://en.wikipedia.org/wiki/Bootstrapping_(compilers)) and how to handle [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form).

Python compiler is only used as an initial compiler, until the language is mature enough to compile itself.

## Features

- [x] Compiled (with FASM)
- [x] Using SSA
- [ ] Turing complete
- [ ] Statically typed
- [ ] Self-hosted (goal <3)

## Language syntax

Lisp-inspired, but without lisp semantics.
It is simply a choice that doesn't involve much thinking and let's me focus on semantics and compilation instead of writing yet another parser. I've done it enough in [other projects](https://github.com/RobertBendun/musique).

But parsing is a little bit interesting, since it is not a classical recursive-descent, but a technique inspired by [Carbon](https://www.youtube.com/watch?v=ZI198eFghJk).
I don't know if I follow it since the code of Carbon is not exactly a welcoming one (to be fair I only looked at it once for few minutes and decided that postponing its understanding is the best choice now).

## Expression types

### `(+ ...)`

Returns sum of all arguments or 0 if none were provided.

### `(syscall ...)`

Calls syscall with given arguments as parameters (first one is syscall id, rest is syscall parameters.
