//! nmem — spreading-activation cognitive memory.
//!
//! A single-process graph brain: neurons, typed synapses, fibers (memories),
//! Hebbian learning, local-calendar time, and a 128-d hashed embed used only
//! as a ranking hint. No GPU. No vector database.
//!
//! ```ignore
//! let mut brain = nmem::Brain::new("demo");
//! brain.remember("JWT expiry caused the Tuesday outage")?;
//! let hits = brain.recall("why did the outage happen");
//! ```

pub mod activation;
pub mod brain;
pub mod causal;
pub mod conflict;
pub mod consolidation;
pub mod context;
pub mod dashboard;
pub mod embed;
pub mod encoder;
pub mod evidence;
pub mod extract;
pub mod health;
pub mod hebbian;
pub mod idf;
pub mod mcp;
pub mod query;
pub mod retrieval;
pub mod simhash;
pub mod stages;
pub mod store;
pub mod temporal;
pub mod types;

pub use brain::Brain;
pub use causal::{CausalDir, CausalResult};
pub use context::ContextPack;
pub use encoder::{EncodeError, EncodingResult};
pub use health::HealthReport;
pub use retrieval::{RecallOpts, RecallResult, RecalledMemory};
pub use store::{MemoryStore, Store};
pub use types::{
    ActivationResult, BrainConfig, Fiber, MemoryStage, MemoryStatus, MemoryTier, MemoryType,
    Neuron, NeuronState, NeuronType, Synapse, SynapseType, DEFAULT_CONFIG,
};
