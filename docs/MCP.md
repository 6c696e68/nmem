# MCP server

nmem speaks **JSON-RPC 2.0** as **newline-delimited JSON** on stdio.

This matches OpenClaw. It is not the HTTP `Content-Length` framing used by
some Claude Code transports. If you need that framing, put a thin adapter
in front — do not change the default server.

## Start

```bash
nmem mcp
# or
nmem-mcp
```

Brain path: `NMEM_BRAIN`, else `~/.neuralmemory/brains/$NEURALMEMORY_BRAIN.json`
(`default.json` when unset).

## Handshake

Client:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"openclaw","version":"0"}}}
```

Server result includes `protocolVersion`, `capabilities.tools`, `serverInfo`.

Then a notification (no `id`, no reply):

```json
{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
```

Also implemented: `ping`, `tools/list`, `tools/call`, `resources/list` (empty),
`prompts/list` (empty). Unknown methods return JSON-RPC error `-32601`.

## Persist-before-ack

For mutating tools the server writes the brain file, *then* prints the
response line. If persist fails, `isError` is true and the tool result
explains the I/O error.

Mutating: `nmem_remember`, `nmem_todo`, `nmem_forget`, `nmem_link`,
`nmem_auto`, `nmem_consolidate`, `nmem_recall`, `nmem_context`.

## Tools

All `tools/call` results are:

```json
{"content":[{"type":"text","text":"..."}],"isError":false}
```

### nmem_remember

Arguments: `content` (string, required), `type`, `tags`, `priority`,
`expires_days`.

### nmem_recall

Arguments: `query` (required), `limit`, `depth`.

Prints ranked summaries with score, hop, confidence, embed cosine.

### nmem_context

Arguments: `query`, `token_budget` (default 400).

Returns a prompt block that stays under the budget.

### nmem_todo

Arguments: `task`. Stored as type `todo` (30-day default expiry).

### nmem_forget

Arguments: `query` (id or text). Deletes one fiber.

### nmem_link

Arguments: `from`, `to`, `type` (`caused_by` default), `weight`.

Picks two distinct matching neurons.

### nmem_auto

Arguments: `text`, `action`. Splits into sentences / paragraphs and remembers
each substantial chunk. Used by OpenClaw session-end hooks.

### nmem_consolidate

No arguments. Decay, promote, merge, expire.

### nmem_causal

Arguments: `query`, `hops`, `direction` (`causes` | `effects`).

### nmem_show

Arguments: `memory_id`.

### nmem_stats / nmem_health

Counts, timezone, embed dim; health grade and issues.

## OpenClaw plugin shim

The official plugin runs `python -m neural_memory.mcp`. Ship
`compat/neural_memory/` on `PYTHONPATH` and set `NMEM_BIN` so that module
`exec`s `nmem mcp`. No Python engine remains in the path.

## Client ids

Numeric and string JSON-RPC ids are both accepted and echoed.

## What is intentionally missing

The Python package exposes dozens of extra tools (batch import, raw graph
surgery, experimental learners). Those blow an agent’s context window on
weak hardware. This server keeps the twelve tools Claw actually calls.
