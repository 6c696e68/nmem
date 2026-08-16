# Engineering rules

These are product invariants, not style preferences. A change that violates
them is a bug even if tests were updated to match.

## Memory model

1. The brain is a **graph**. Recall is spreading activation plus ranking.
   Hashed embed is a *hint*. It must not become the primary index.
2. Do not add a vector database, ANN index, or embedding-model runtime
   unless the user explicitly opts into a separate, optional crate.
3. Causal questions walk `CAUSED_BY` / `LEADS_TO` (and documented fallbacks).
   Do not answer “why” with cosine alone.
4. Typed overlap must not spray `CAUSED_BY` onto every similar fiber.
   Clause extract + inherit/push causal edges. Never invert cause and effect.
5. Hypothesis belief is updated **at encode time** from evidence language.
   Recall must not run Bayes again or write belief back. Two recalls in a
   row must leave `fiber.belief` unchanged.
6. Expired and superseded fibers are hidden from recall and context.
   Inspection tools (`show`, dashboard list) may still display them.
7. `forget` matches the best Jaccard fiber, not “first token hit”.
   `link` must pick two *distinct* neurons.

## Persistence and MCP

8. Mutating MCP tools persist **before** the JSON-RPC response is written.
   A client that kills the process after `ok` must still have a durable brain.
9. Do not persist on `initialize`, `tools/list`, or `ping`.
10. CLI and MCP share the same default path
    (`NMEM_BRAIN` → `~/.neuralmemory/brains/<name>.json`).
11. Saves are atomic: write a sibling tmp file, `rename`, delete tmp on failure.
12. Invalid UTF-8 on MCP stdin is skipped. It must not exit the process.

## Time and ranking

13. Relative time (“today”, “hôm nay”, weekdays) uses the **local** calendar.
    Offset order: `NMEM_TZ_HOURS` → `NMEM_TZ` → libc → UTC.
14. Query expansion must not map generic words (`web`, `json`, `line`) back
    to abbreviations such as `jwt`.
15. Refractory period must not block a second recall a few milliseconds later
    (OpenClaw auto-context then `nmem_recall`). Intra-spread cycles use
    the visited set, not leftover refractory.

## Encode and indexes

16. Exact neuron reuse goes through the **exact content index**
    `(type, lowercased content) → id`. Do not reintroduce a full-graph
    linear scan for `find_by_content_exact`.
17. Anchor overlap on encode uses the **inverted index** with per-token and
    result caps. Do not scan every neuron on each `remember`.
18. Simhash ALIAS compares a **bounded recent window** (fiber insertion order),
    not every fiber in the brain.
19. Graph walks (`neighbors` / activation) must not clone full Neuron/Synapse
    objects on the hot path; pass references or lightweight edge records.

## Hardware

20. No GPU. No model download. No required network at runtime.
21. New dependencies need a reason. Prefer `std`. The release binary should
    stay in the low-megabyte range.
22. Dashboard is served from the same binary. Do not require Node to operate
    the product.

## Tests

23. Engine, MCP stdio (real child process), and dashboard HTTP tests must
    stay green. Do not delete a regression to land a feature.
24. HashMap iteration is not a stable seed. Causal seed ranking must prefer
    neurons that already have causal outgoing edges.
25. Encode-scale stress (`stress_encode`) must not regress to quadratic
    full-graph scans on `remember`.

## Language and docs

26. Published documentation is **English only**.
27. Do not document sandbox ports, container paths, or preview internals
    as if they were part of the product.
28. README, RULES, and MCP docs must be updated in the same change that
    alters CLI flags, env vars, or tool names.
