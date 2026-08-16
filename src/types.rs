use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn new_id() -> String {
    Uuid::new_v4().to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronType {
    Time,
    Spatial,
    Entity,
    Action,
    State,
    Concept,
    Sensory,
    Intent,
    Hypothesis,
    Prediction,
    Schema,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Fact,
    Decision,
    Preference,
    Todo,
    Insight,
    Context,
    Instruction,
    Error,
    Workflow,
    Reference,
    Tool,
    Hypothesis,
    Prediction,
    Schema,
    Boundary,
}

impl MemoryType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Decision => "decision",
            Self::Preference => "preference",
            Self::Todo => "todo",
            Self::Insight => "insight",
            Self::Context => "context",
            Self::Instruction => "instruction",
            Self::Error => "error",
            Self::Workflow => "workflow",
            Self::Reference => "reference",
            Self::Tool => "tool",
            Self::Hypothesis => "hypothesis",
            Self::Prediction => "prediction",
            Self::Schema => "schema",
            Self::Boundary => "boundary",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "fact" => Self::Fact,
            "decision" => Self::Decision,
            "preference" => Self::Preference,
            "todo" => Self::Todo,
            "insight" => Self::Insight,
            "context" => Self::Context,
            "instruction" => Self::Instruction,
            "error" => Self::Error,
            "workflow" => Self::Workflow,
            "reference" => Self::Reference,
            "tool" => Self::Tool,
            "hypothesis" => Self::Hypothesis,
            "prediction" => Self::Prediction,
            "schema" => Self::Schema,
            "boundary" => Self::Boundary,
            _ => return None,
        })
    }

    pub fn default_expiry_days(self) -> Option<u32> {
        match self {
            Self::Fact | Self::Preference | Self::Instruction | Self::Reference | Self::Schema | Self::Boundary => None,
            Self::Decision | Self::Tool => Some(90),
            Self::Todo | Self::Error | Self::Prediction => Some(30),
            Self::Insight | Self::Hypothesis => Some(180),
            Self::Context => Some(7),
            Self::Workflow => Some(365),
        }
    }

    pub fn decay_rate(self) -> f64 {
        match self {
            Self::Fact => 0.02,
            Self::Decision | Self::Preference | Self::Hypothesis => 0.03,
            Self::Reference => 0.04,
            Self::Insight | Self::Instruction => 0.05,
            Self::Tool => 0.06,
            Self::Context | Self::Workflow => 0.08,
            Self::Prediction => 0.10,
            Self::Error => 0.12,
            Self::Todo => 0.15,
            Self::Schema | Self::Boundary => 0.01,
        }
    }

    pub fn is_high_signal(self) -> bool {
        matches!(
            self,
            Self::Decision | Self::Insight | Self::Preference | Self::Instruction | Self::Boundary
        )
    }

    pub fn default_tier(self) -> MemoryTier {
        match self {
            Self::Preference | Self::Instruction | Self::Boundary | Self::Schema => MemoryTier::Hot,
            Self::Todo | Self::Context => MemoryTier::Cold,
            _ => MemoryTier::Warm,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStage {
    #[default]
    ShortTerm,
    Working,
    Episodic,
    Semantic,
}

impl MemoryStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ShortTerm => "stm",
            Self::Working => "working",
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    Hot,
    #[default]
    Warm,
    Cold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    #[default]
    Active,
    Superseded,
    Expired,
}

impl MemoryStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Superseded => "superseded",
            Self::Expired => "expired",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynapseType {
    HappenedAt,
    Before,
    After,
    During,
    AtLocation,
    Contains,
    Near,
    CausedBy,
    LeadsTo,
    Enables,
    Prevents,
    CoOccurs,
    RelatedTo,
    SimilarTo,
    IsA,
    HasProperty,
    Involves,
    Felt,
    Evokes,
    Contradicts,
    ResolvedBy,
    EffectiveFor,
    UsedWith,
    Alias,
    EvidenceFor,
    EvidenceAgainst,
    Predicted,
    VerifiedBy,
    FalsifiedBy,
    Supersedes,
    DerivedFrom,
    EvolvesFrom,
    SubgoalOf,
    SourceOf,
    Imports,
    Calls,
    DependsOn,
    Inherits,
    Implements,
    DefinedIn,
    Raises,
    StoredBy,
    HasValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynapseRole {
    Supersession,
    Reinforcement,
    Weakening,
    Sequential,
    Structural,
    Lateral,
    Passive,
}

impl SynapseType {
    pub fn role(self) -> SynapseRole {
        use SynapseRole::*;
        use SynapseType::*;
        match self {
            ResolvedBy | Supersedes | EvolvesFrom | FalsifiedBy => Supersession,
            EvidenceFor | VerifiedBy | EffectiveFor => Reinforcement,
            EvidenceAgainst | Contradicts | Prevents => Weakening,
            Before | After | LeadsTo | Enables | CausedBy | Calls => Sequential,
            CoOccurs | RelatedTo | SimilarTo | UsedWith => Lateral,
            HappenedAt | During | Felt | Evokes | Alias => Passive,
            _ => Structural,
        }
    }

    pub fn role_multiplier(self) -> f64 {
        match self.role() {
            SynapseRole::Sequential => 1.3,
            SynapseRole::Reinforcement => 1.2,
            SynapseRole::Supersession => 1.1,
            SynapseRole::Structural => 1.0,
            SynapseRole::Weakening => 0.9,
            SynapseRole::Lateral => 0.85,
            SynapseRole::Passive => 0.0,
        }
    }

    pub fn is_bidirectional(self) -> bool {
        matches!(
            self,
            Self::CoOccurs | Self::RelatedTo | Self::SimilarTo | Self::Near | Self::UsedWith
        )
    }

    pub fn inverse(self) -> Option<Self> {
        match self {
            Self::Before => Some(Self::After),
            Self::After => Some(Self::Before),
            Self::CausedBy => Some(Self::LeadsTo),
            Self::LeadsTo => Some(Self::CausedBy),
            Self::Contains => Some(Self::AtLocation),
            Self::AtLocation => Some(Self::Contains),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::HappenedAt => "happened_at",
            Self::Before => "before",
            Self::After => "after",
            Self::During => "during",
            Self::AtLocation => "at_location",
            Self::Contains => "contains",
            Self::Near => "near",
            Self::CausedBy => "caused_by",
            Self::LeadsTo => "leads_to",
            Self::Enables => "enables",
            Self::Prevents => "prevents",
            Self::CoOccurs => "co_occurs",
            Self::RelatedTo => "related_to",
            Self::SimilarTo => "similar_to",
            Self::IsA => "is_a",
            Self::HasProperty => "has_property",
            Self::Involves => "involves",
            Self::Felt => "felt",
            Self::Evokes => "evokes",
            Self::Contradicts => "contradicts",
            Self::ResolvedBy => "resolved_by",
            Self::EffectiveFor => "effective_for",
            Self::UsedWith => "used_with",
            Self::Alias => "alias",
            Self::EvidenceFor => "evidence_for",
            Self::EvidenceAgainst => "evidence_against",
            Self::Predicted => "predicted",
            Self::VerifiedBy => "verified_by",
            Self::FalsifiedBy => "falsified_by",
            Self::Supersedes => "supersedes",
            Self::DerivedFrom => "derived_from",
            Self::EvolvesFrom => "evolves_from",
            Self::SubgoalOf => "subgoal_of",
            Self::SourceOf => "source_of",
            Self::Imports => "imports",
            Self::Calls => "calls",
            Self::DependsOn => "depends_on",
            Self::Inherits => "inherits",
            Self::Implements => "implements",
            Self::DefinedIn => "defined_in",
            Self::Raises => "raises",
            Self::StoredBy => "stored_by",
            Self::HasValue => "has_value",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: String,
    pub type_: NeuronType,
    pub content: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: u64,
    pub ephemeral: bool,
}

impl Neuron {
    pub fn create(type_: NeuronType, content: impl Into<String>) -> Self {
        Self {
            id: new_id(),
            type_,
            content: content.into(),
            metadata: HashMap::new(),
            created_at: now_ms(),
            ephemeral: false,
        }
    }

    pub fn with_meta(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    pub fn is_anchor(&self) -> bool {
        self.metadata
            .get("is_anchor")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    pub neuron_id: String,
    pub activation_level: f64,
    pub access_frequency: u32,
    pub last_activated: Option<u64>,
    pub decay_rate: f64,
    pub firing_threshold: f64,
    #[serde(default)]
    pub refractory_until: u64,
}

impl NeuronState {
    pub fn new(neuron_id: impl Into<String>, decay_rate: f64) -> Self {
        Self {
            neuron_id: neuron_id.into(),
            activation_level: 0.0,
            access_frequency: 0,
            last_activated: None,
            decay_rate,
            firing_threshold: 0.3,
            refractory_until: 0,
        }
    }

    /// Sigmoid-gated activation from the Python `NeuronState.activate`.
    pub fn activate(&mut self, level: f64, now: u64, steepness: f64) {
        let clamped = level.clamp(0.0, 1.0);
        let sigmoid = 1.0 / (1.0 + (-steepness * (clamped - 0.5)).exp());
        self.activation_level = sigmoid;
        self.access_frequency += 1;
        self.last_activated = Some(now);
    }

    pub fn fire(&mut self, level: f64, now: u64, steepness: f64, refractory_ms: u64) {
        self.activate(level, now, steepness);
        self.refractory_until = now.saturating_add(refractory_ms.max(1));
    }

    pub fn in_refractory(&self, now: u64) -> bool {
        now < self.refractory_until
    }

    /// Exponential decay: `level * e^(-rate * days)`.
    pub fn decay(&mut self, delta_seconds: f64) {
        if delta_seconds <= 0.0 {
            return;
        }
        let days = delta_seconds / 86_400.0;
        self.activation_level *= (-self.decay_rate * days).exp();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub type_: SynapseType,
    pub weight: f64,
    pub direction: Direction,
    pub reinforced_count: u32,
    pub last_activated: Option<u64>,
    pub created_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Uni,
    Bi,
}

impl Synapse {
    pub fn create(source: impl Into<String>, target: impl Into<String>, type_: SynapseType, weight: f64) -> Self {
        let direction = if type_.is_bidirectional() {
            Direction::Bi
        } else {
            Direction::Uni
        };
        Self {
            id: new_id(),
            source_id: source.into(),
            target_id: target.into(),
            type_,
            weight: weight.clamp(0.0, 1.0),
            direction,
            reinforced_count: 0,
            last_activated: None,
            created_at: now_ms(),
        }
    }

    pub fn reinforce(&mut self, delta: f64, now: u64) {
        self.weight = (self.weight + delta).min(1.0);
        self.reinforced_count += 1;
        self.last_activated = Some(now);
    }

    /// Time decay: sigmoid half-life 60d, longer when reinforced. Floor 0.3 + 0.05*count.
    pub fn time_decay(&mut self, now: u64) {
        let last = self.last_activated.unwrap_or(self.created_at);
        let hours = ((now.saturating_sub(last)) as f64) / 3_600_000.0;
        let half_life = 1440.0 * (1.0 + self.reinforced_count as f64 * 0.5);
        let spread = half_life / 2.0;
        let exponent = ((hours - half_life) / spread).clamp(-100.0, 100.0);
        let mut factor = 1.0 / (1.0 + exponent.exp());
        let floor = 0.3 + (self.reinforced_count as f64 * 0.05).min(0.5);
        factor = factor.max(floor);
        self.weight *= factor;
    }

    pub fn other_end(&self, neuron_id: &str) -> Option<&str> {
        if self.source_id == neuron_id {
            Some(&self.target_id)
        } else if self.target_id == neuron_id {
            Some(&self.source_id)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fiber {
    pub id: String,
    pub neuron_ids: Vec<String>,
    pub synapse_ids: Vec<String>,
    pub anchor_neuron_id: String,
    pub pathway: Vec<String>,
    pub conductivity: f64,
    pub last_conducted: Option<u64>,
    pub summary: String,
    pub salience: f64,
    pub frequency: u32,
    pub memory_type: MemoryType,
    pub priority: u8,
    pub tags: Vec<String>,
    pub created_at: u64,
    #[serde(default = "default_belief")]
    pub belief: f64,
    #[serde(default)]
    pub stage: MemoryStage,
    #[serde(default)]
    pub tier: MemoryTier,
    #[serde(default = "default_trust")]
    pub trust: f64,
    #[serde(default)]
    pub expires_at: Option<u64>,
    #[serde(default)]
    pub status: MemoryStatus,
}

fn default_belief() -> f64 {
    0.5
}

fn default_trust() -> f64 {
    0.8
}

fn default_refractory_ms() -> u64 {
    250
}

fn default_rrf_k() -> f64 {
    60.0
}

impl Fiber {
    pub fn create(
        neuron_ids: Vec<String>,
        synapse_ids: Vec<String>,
        anchor: impl Into<String>,
        summary: impl Into<String>,
        memory_type: MemoryType,
        tags: Vec<String>,
    ) -> Self {
        let anchor = anchor.into();
        let pathway = if neuron_ids.contains(&anchor) {
            let mut p = vec![anchor.clone()];
            for id in &neuron_ids {
                if id != &anchor {
                    p.push(id.clone());
                }
            }
            p
        } else {
            neuron_ids.clone()
        };
        Self {
            id: new_id(),
            neuron_ids,
            synapse_ids,
            anchor_neuron_id: anchor,
            pathway,
            conductivity: 1.0,
            last_conducted: None,
            summary: summary.into(),
            salience: 0.0,
            frequency: 0,
            memory_type,
            priority: 5,
            tags,
            created_at: now_ms(),
            belief: 0.5,
            stage: MemoryStage::ShortTerm,
            tier: memory_type.default_tier(),
            trust: 0.8,
            expires_at: memory_type.default_expiry_days().map(|d| now_ms() + d as u64 * 86_400_000),
            status: MemoryStatus::Active,
        }
    }

    pub fn is_live(&self, now: u64) -> bool {
        if self.status != MemoryStatus::Active {
            return false;
        }
        match self.expires_at {
            Some(e) if now >= e => false,
            _ => true,
        }
    }

    pub fn conduct(&mut self, now: u64) {
        self.conductivity = (self.conductivity + 0.02).min(1.0);
        self.last_conducted = Some(now);
        self.frequency += 1;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    pub decay_rate: f64,
    pub reinforcement_delta: f64,
    pub activation_threshold: f64,
    pub max_spread_hops: u32,
    pub default_synapse_weight: f64,
    pub sigmoid_steepness: f64,
    pub firing_threshold: f64,
    pub diminishing_returns_enabled: bool,
    pub diminishing_returns_threshold: f64,
    pub diminishing_returns_min_neurons: u32,
    pub diminishing_returns_grace_hops: u32,
    pub recency_halflife_hours: f64,
    pub tag_match_boost: f64,
    pub high_signal_boost: f64,
    pub merge_overlap_threshold: f64,
    pub consolidation_prune_threshold: f64,
    #[serde(default = "default_refractory_ms")]
    pub refractory_ms: u64,
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f64,
}

pub const DEFAULT_CONFIG: BrainConfig = BrainConfig {
    decay_rate: 0.1,
    reinforcement_delta: 0.05,
    activation_threshold: 0.3,
    max_spread_hops: 4,
    default_synapse_weight: 0.5,
    sigmoid_steepness: 6.0,
    firing_threshold: 0.3,
    diminishing_returns_enabled: true,
    diminishing_returns_threshold: 0.15,
    diminishing_returns_min_neurons: 2,
    diminishing_returns_grace_hops: 1,
    recency_halflife_hours: 168.0,
    tag_match_boost: 0.15,
    high_signal_boost: 1.15,
    merge_overlap_threshold: 0.5,
    consolidation_prune_threshold: 0.05,
    refractory_ms: 250,
    rrf_k: 60.0,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainMeta {
    pub id: String,
    pub name: String,
    pub config: BrainConfig,
    pub created_at: u64,
    pub updated_at: u64,
}

impl BrainMeta {
    pub fn create(name: impl Into<String>) -> Self {
        let t = now_ms();
        Self {
            id: new_id(),
            name: name.into(),
            config: DEFAULT_CONFIG,
            created_at: t,
            updated_at: t,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationResult {
    pub neuron_id: String,
    pub activation_level: f64,
    pub hop_distance: u32,
    pub path: Vec<String>,
    pub source_anchor: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActivationTrace {
    pub new_neurons_per_hop: HashMap<u32, u32>,
    pub activation_gain_per_hop: HashMap<u32, f64>,
    pub max_hop_used: u32,
    pub max_hop_allowed: u32,
    pub stopped_early: bool,
    pub stop_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainSnapshot {
    pub version: String,
    pub brain: BrainMeta,
    pub neurons: Vec<Neuron>,
    pub states: Vec<NeuronState>,
    pub synapses: Vec<Synapse>,
    pub fibers: Vec<Fiber>,
}
