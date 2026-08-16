# Architecture

nmem is a single-process cognitive store. The unit of meaning is a **fiber**
(one remembered episode) sitting on a directed graph of **neurons** and
**synapses**.

```
text ──encode──► neurons + typed synapses + fiber
                      │
query ──lexical / IDF / inverted index──► anchors
                      │
              spreading activation
                      │
         fiber score × RRF × hashed embed
                      │
                   ranked recall
```

## Layers

| Crate module | Role |
| --- | --- |
| `encoder` | Type detect, extract entities/places/intents, clause relations, conflicts |
| `store` | In-memory graph, inverted + exact indexes, fiber order, atomic JSON |
| `activation` | Hop-decaying spread, diminishing returns |
| `retrieval` | IDF anchors, fiber score, causal semantics, RRF, embed fusion |
| `causal` | BFS on cause/effect (seed prefers nodes with causal degree) |
| `hebbian` | `Δw = η_eff · pre · post · (w_max − w)` plus competitive normalize |
| `evidence` | Bayes on hypothesis/prediction **at encode** |
| `conflict` | Decision reversal → `SUPERSEDES` |
| `consolidation` | Decay, stage promotion, expiry, merge (skips conflicted / non-active) |
| `temporal` | Local-calendar windows |
| `embed` | 128-d hashed n-grams, not persisted |
| `mcp` | JSON-RPC 2.0 NDJSON |
| `dashboard` | `std` HTTP + static HTML |

`Brain` is the only facade most callers should use.

## Indexes

| Structure | Key | Purpose |
| --- | --- | --- |
| `inv` | token → neuron ids | Lexical recall + encode overlap candidates |
| `exact` | (neuron type, lowercased content) → id | O(1) reuse on encode |
| `adj` | neuron id → synapse ids | Graph walks without scanning all edges |
| `fiber_order` | insertion sequence | Bounded recent window for ALIAS / scans |

## Encoding

1. Reject empty input.
2. Suggest `MemoryType` (fact, decision, error, todo, hypothesis, …).
3. Create an **anchor** neuron (the raw sentence, truncated on UTF-8 char
   boundaries).
4. Extract time, entities, concepts, actions, places, intents. Reuse exact
   content matches.
5. Wire typed edges (`HAPPENED_AT`, `INVOLVES`, `AT_LOCATION`, `RELATED_TO`,
   `CO_OCCURS` capped at 6 cluster members).
6. Clause patterns (`X because Y`, `X caused Y`) create concept neurons and
   typed edges **on the anchor**, then inherit/push those edges onto
   overlapping events so encode order does not drop the chain.
7. Overlap with existing anchors is `RELATED_TO` only. It is not a license
   to stamp `CAUSED_BY` on every neighbour.
8. Conflicts: a reversed decision marks the old fiber `Superseded` and adds
   `SUPERSEDES`.
9. Near-duplicate fibers (simhash Hamming ≤ 12) get `ALIAS`.
10. Evidence language updates overlapping hypothesis `belief` once.

Each fiber gets default expiry from its type (todo/error 30 days, context 7,
facts none, …).

## Recall

1. Tokenize and expand the query (synonyms, EN↔VI, abbreviations — no reverse
   map from generic words).
2. IDF-weighted inverted-index lookup for anchors.
3. Spread from those anchors. Passive roles (`HAPPENED_AT`, `ALIAS`, …) have
   multiplier 0 and do not carry activation.
4. Score every **live** fiber that received activation:
   salience × recency × conductivity × coverage × type/trust/stage ×
   lexical × temporal window × causal/evidence semantics.
5. Reciprocal rank fusion of graph order and lexical order.
6. Multiply by `1 + 0.28 · max(cosine, 0)` from the hashed embed.
7. Truncate. Conduct the winners. Hebbian-update path edges. Normalize
   outgoing weights **only** on sources that were just reinforced.

Session warmth (`Brain.warm`) primes the next query in-process. It is not
written to disk.

## Causal walk

`causal` / `effects` BFS from a seed. The seed is the lexical match with the
highest count of outgoing causal synapses, then anchors over concepts. This
avoids HashMap-tie flakes landing on a bare event with no edges.

If the seed has no directed cause/effect edge, a documented fallback walks
any causal-family neighbour. Related-to is not in that family.

## Persistence

`BrainSnapshot` JSON:

```text
{ version, brain, neurons[], states[], synapses[], fibers[] }
```

Load rebuilds adjacency and the inverted index. Unknown future fields should
be added with `#[serde(default)]`.

## Concurrency

The engine is single-threaded. The dashboard wraps `Brain` in a `Mutex`.
MCP is one request at a time on stdin. There is no multi-writer protocol.

## What this is not

It is not SQLite, not HNSW, not an LLM, not a multi-tenant server. Those can
exist *around* nmem. They must not replace the graph.
