# Contributing

Thank you for considering a change to nmem.

## Before you write code

1. Read [RULES.md](RULES.md). Those invariants are non-negotiable.
2. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) so new edges and
   scores land in the right layer.
3. Prefer a failing test that states the invariant, then the fix.

## Setup

```bash
cargo test
cargo build --release
```

No `npm` is required for the engine, MCP server, or dashboard.

## What we will merge

- Bug fixes with a regression test
- Retrieval / causal correctness
- Smaller binary, faster encode/recall, less disk
- Docs that match the code
- Optional, clearly gated extras that do not pull a GPU stack into default builds

## What we will not merge

- A vector database as the default store
- Embedding models or ONNX runtimes in the default crate
- Expanding MCP to the 63-tool Python surface
- UI frameworks or Node as a runtime dependency
- Changes that persist *after* the MCP client has already been told `ok`
- Bayes-on-recall or other silent belief drift

## Style

- Rust 2021, `cargo fmt` if you have it
- No `unwrap` on I/O or user input in library code
- Keep public API small: `Brain` is the facade
- Do not add comments that restate the next line

## Pull requests

- One concern per PR
- Update README / RULES / MCP docs when flags, env, or tools change
- Include `cargo test` output in the description if CI is not set up yet

## Security

nmem is a local process. The dashboard binds wherever you tell it to
(`0.0.0.0` is convenient and dangerous on a public host). Do not expose
it to the internet without a reverse proxy and access control. The JSON
brain may contain secrets the user typed — treat the file as private.
