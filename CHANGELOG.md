# Changelog

## 0.1.1

Initial public release of the Rust neural-memory core (performance pass included).

### Core
- Spreading-activation graph (neurons, typed synapses, fibers)
- Encode: clause relations, inverse edges, conflict / supersede, expiry
- Recall: IDF, RRF, causal semantics, hashed 128-d embed (ranking only)
- Local-calendar time (`NMEM_TZ` / `NMEM_TZ_HOURS` / libc)

### Indexes and speed
- Exact content index for O(1) neuron reuse on encode
- Anchor overlap via inverted index with caps (no full-graph scan)
- Activation edge cache without Neuron/Synapse clones per hop
- Recall scores by id; materializes only top-ranked fiber bodies
- Bounded simhash ALIAS window via fiber insertion order
- Release profile: `opt-level = 3`, LTO, stripped

### Surfaces
- MCP stdio for OpenClaw — persist before ack, 12 tools
- Local HTTP dashboard in the same binary
- Atomic compact JSON persistence
- Python shim so `python -m neural_memory.mcp` execs `nmem mcp`
