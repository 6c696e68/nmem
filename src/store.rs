//! In-memory graph + compact atomic JSON. Inverted index for O(tokens) lookup.
//! No SQLite — keeps the binary tiny on weak devices.

use crate::extract::tokenize;
use crate::types::{
    BrainMeta, BrainSnapshot, Fiber, Neuron, NeuronState, NeuronType, Synapse, now_ms,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub trait Store {
    fn meta(&self) -> &BrainMeta;
    fn meta_mut(&mut self) -> &mut BrainMeta;
    fn add_neuron(&mut self, n: Neuron);
    fn add_synapse(&mut self, s: Synapse) -> Synapse;
    fn get_synapse_mut(&mut self, id: &str) -> Option<&mut Synapse>;
    fn remove_synapse(&mut self, id: &str) -> bool;
    fn find_synapse(&self, source: &str, target: &str, ty: crate::types::SynapseType) -> Option<Synapse>;
    fn add_fiber(&mut self, f: Fiber);
    fn remove_fiber(&mut self, id: &str) -> bool;
    fn remove_neuron(&mut self, id: &str) -> bool;
    fn get_neuron(&self, id: &str) -> Option<&Neuron>;
    fn get_state(&self, id: &str) -> Option<&NeuronState>;
    fn get_state_mut(&mut self, id: &str) -> Option<&mut NeuronState>;
    fn get_fiber(&self, id: &str) -> Option<&Fiber>;
    fn get_fiber_mut(&mut self, id: &str) -> Option<&mut Fiber>;
    fn neurons(&self) -> Vec<&Neuron>;
    fn synapses(&self) -> Vec<&Synapse>;
    fn fibers(&self) -> Vec<&Fiber>;
    fn neuron_count(&self) -> usize;
    fn synapse_count(&self) -> usize;
    fn fiber_count(&self) -> usize;
    fn neighbors<'a>(&'a self, id: &str, min_weight: f64) -> Vec<(&'a Neuron, &'a Synapse)>;
    fn neighbors_out<'a>(&'a self, id: &str, min_weight: f64) -> Vec<(&'a Neuron, &'a Synapse)>;
    fn find_synapse_id(&self, source: &str, target: &str, ty: crate::types::SynapseType) -> Option<String>;
    fn find_by_content_exact(&self, content: &str, ty: NeuronType) -> Option<Neuron>;
    fn find_content_match(&self, tokens: &[String], limit: usize) -> Vec<Neuron>;
    fn find_anchor_overlap(&self, tokens: &[String], exclude: &str) -> Vec<(Neuron, f64)>;
    fn recent_fibers(&self, limit: usize) -> Vec<&Fiber>;
    fn token_df(&self, token: &str) -> u32;
    fn token_idf(&self, token: &str) -> f64 {
        crate::idf::idf(self.token_df(token), self.fiber_count() as u32)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStore {
    pub meta: BrainMeta,
    neurons: HashMap<String, Neuron>,
    states: HashMap<String, NeuronState>,
    synapses: HashMap<String, Synapse>,
    fibers: HashMap<String, Fiber>,
    adj: HashMap<String, Vec<String>>,
    /// token → neuron ids (rebuilt on load)
    inv: HashMap<String, HashSet<String>>,
    /// (neuron_type discriminant, lowercased content) → neuron id — O(1) exact reuse
    exact: HashMap<(u8, String), String>,
    /// Insertion order of fiber ids — O(1) recent-window scans on encode
    fiber_order: Vec<String>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::named("default")
    }

    pub fn named(name: impl Into<String>) -> Self {
        Self {
            meta: BrainMeta::create(name),
            neurons: HashMap::new(),
            states: HashMap::new(),
            synapses: HashMap::new(),
            fibers: HashMap::new(),
            adj: HashMap::new(),
            inv: HashMap::new(),
            exact: HashMap::new(),
            fiber_order: Vec::new(),
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let raw = std::fs::read(path)?;
        let snap: BrainSnapshot = serde_json::from_slice(&raw)?;
        Ok(Self::from_snapshot(snap))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), StoreError> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let snap = self.snapshot();
        let json = serde_json::to_vec(&snap)?;
        let path = path.as_ref();
        let fname = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("brain.json");
        let tmp = path.with_file_name(format!(".{}.{}.tmp", fname, std::process::id()));
        if let Err(e) = std::fs::write(&tmp, &json) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
        Ok(())
    }

    pub fn snapshot(&self) -> BrainSnapshot {
        BrainSnapshot {
            version: "nmem-rs/0.1".into(),
            brain: self.meta.clone(),
            neurons: self.neurons.values().cloned().collect(),
            states: self.states.values().cloned().collect(),
            synapses: self.synapses.values().cloned().collect(),
            fibers: self.fibers.values().cloned().collect(),
        }
    }

    pub fn from_snapshot(snap: BrainSnapshot) -> Self {
        let mut s = Self {
            meta: snap.brain,
            neurons: HashMap::new(),
            states: HashMap::new(),
            synapses: HashMap::new(),
            fibers: HashMap::new(),
            adj: HashMap::new(),
            inv: HashMap::new(),
            exact: HashMap::new(),
            fiber_order: Vec::new(),
        };
        for n in snap.neurons {
            s.add_neuron(n);
        }
        for st in snap.states {
            s.states.insert(st.neuron_id.clone(), st);
        }
        for syn in snap.synapses {
            s.insert_synapse(syn);
        }
        for f in snap.fibers {
            s.fibers.insert(f.id.clone(), f);
        }
        s
    }

    pub fn fibers_vec(&self) -> Vec<Fiber> {
        self.fibers.values().cloned().collect()
    }

    pub fn insert_synapse(&mut self, s: Synapse) {
        self.index_synapse(&s);
        self.synapses.insert(s.id.clone(), s);
        self.meta.updated_at = now_ms();
    }

    fn index_neuron(&mut self, n: &Neuron) {
        for t in tokenize(&n.content) {
            self.inv.entry(t).or_default().insert(n.id.clone());
        }
        let key = exact_key(n.type_, &n.content);
        self.exact.insert(key, n.id.clone());
    }

    fn deindex_neuron(&mut self, n: &Neuron) {
        for t in tokenize(&n.content) {
            if let Some(set) = self.inv.get_mut(&t) {
                set.remove(&n.id);
                if set.is_empty() {
                    self.inv.remove(&t);
                }
            }
        }
        let key = exact_key(n.type_, &n.content);
        if self.exact.get(&key).map(|id| id == &n.id).unwrap_or(false) {
            self.exact.remove(&key);
        }
    }

    fn index_synapse(&mut self, syn: &Synapse) {
        self.adj
            .entry(syn.source_id.clone())
            .or_default()
            .push(syn.id.clone());
        if syn.direction == crate::types::Direction::Bi || true {
            // Traversal is always bidirectional at the graph layer
            // (activation.py uses direction="both"); role/weight gate the signal.
            self.adj
                .entry(syn.target_id.clone())
                .or_default()
                .push(syn.id.clone());
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Store for MemoryStore {
    fn meta(&self) -> &BrainMeta {
        &self.meta
    }
    fn meta_mut(&mut self) -> &mut BrainMeta {
        &mut self.meta
    }

    fn add_neuron(&mut self, n: Neuron) {
        let decay = self.meta.config.decay_rate;
        if !self.states.contains_key(&n.id) {
            self.states
                .insert(n.id.clone(), NeuronState::new(&n.id, decay));
        }
        if let Some(old) = self.neurons.get(&n.id) {
            let old = old.clone();
            self.deindex_neuron(&old);
        }
        self.index_neuron(&n);
        self.neurons.insert(n.id.clone(), n);
        self.meta.updated_at = now_ms();
    }

    fn add_synapse(&mut self, s: Synapse) -> Synapse {
        if let Some(ex) = self.find_synapse(&s.source_id, &s.target_id, s.type_) {
            if let Some(existing) = self.synapses.get_mut(&ex.id) {
                existing.reinforce(0.05, now_ms());
                return existing.clone();
            }
        }
        self.index_synapse(&s);
        let out = s.clone();
        self.synapses.insert(s.id.clone(), s);
        self.meta.updated_at = now_ms();
        out
    }

    fn get_synapse_mut(&mut self, id: &str) -> Option<&mut Synapse> {
        self.synapses.get_mut(id)
    }

    fn remove_synapse(&mut self, id: &str) -> bool {
        let Some(s) = self.synapses.remove(id) else {
            return false;
        };
        if let Some(list) = self.adj.get_mut(&s.source_id) {
            list.retain(|x| x != id);
        }
        if let Some(list) = self.adj.get_mut(&s.target_id) {
            list.retain(|x| x != id);
        }
        true
    }

    fn find_synapse(&self, source: &str, target: &str, ty: crate::types::SynapseType) -> Option<Synapse> {
        self.find_synapse_id(source, target, ty)
            .and_then(|id| self.synapses.get(&id).cloned())
    }

    fn find_synapse_id(&self, source: &str, target: &str, ty: crate::types::SynapseType) -> Option<String> {
        for endpoint in [source, target] {
            if let Some(ids) = self.adj.get(endpoint) {
                for sid in ids {
                    if let Some(sy) = self.synapses.get(sid) {
                        if sy.type_ != ty { continue; }
                        if (sy.source_id == source && sy.target_id == target)
                            || (sy.direction == crate::types::Direction::Bi
                                && sy.source_id == target
                                && sy.target_id == source)
                        {
                            return Some(sy.id.clone());
                        }
                    }
                }
            }
        }
        None
    }

    fn add_fiber(&mut self, f: Fiber) {
        if !self.fibers.contains_key(&f.id) {
            self.fiber_order.push(f.id.clone());
        }
        self.fibers.insert(f.id.clone(), f);
        self.meta.updated_at = now_ms();
    }

    fn remove_fiber(&mut self, id: &str) -> bool {
        let gone = self.fibers.remove(id).is_some();
        if gone {
            self.fiber_order.retain(|x| x != id);
        }
        gone
    }

    fn remove_neuron(&mut self, id: &str) -> bool {
        let Some(n) = self.neurons.remove(id) else {
            return false;
        };
        self.deindex_neuron(&n);
        self.states.remove(id);
        let syn_ids: Vec<String> = self
            .adj
            .get(id)
            .cloned()
            .unwrap_or_default();
        for sid in syn_ids {
            self.remove_synapse(&sid);
        }
        self.adj.remove(id);
        true
    }

    fn get_neuron(&self, id: &str) -> Option<&Neuron> {
        self.neurons.get(id)
    }
    fn get_state(&self, id: &str) -> Option<&NeuronState> {
        self.states.get(id)
    }
    fn get_state_mut(&mut self, id: &str) -> Option<&mut NeuronState> {
        self.states.get_mut(id)
    }
    fn get_fiber(&self, id: &str) -> Option<&Fiber> {
        self.fibers.get(id)
    }
    fn get_fiber_mut(&mut self, id: &str) -> Option<&mut Fiber> {
        self.fibers.get_mut(id)
    }

    fn neurons(&self) -> Vec<&Neuron> {
        self.neurons.values().collect()
    }
    fn synapses(&self) -> Vec<&Synapse> {
        self.synapses.values().collect()
    }
    fn fibers(&self) -> Vec<&Fiber> {
        self.fibers.values().collect()
    }
    fn neuron_count(&self) -> usize {
        self.neurons.len()
    }
    fn synapse_count(&self) -> usize {
        self.synapses.len()
    }
    fn fiber_count(&self) -> usize {
        self.fibers.len()
    }

    fn neighbors<'a>(&'a self, id: &str, min_weight: f64) -> Vec<(&'a Neuron, &'a Synapse)> {
        let Some(ids) = self.adj.get(id) else { return vec![] };
        let mut out = Vec::with_capacity(ids.len());
        let mut seen = HashSet::new();
        for sid in ids {
            if !seen.insert(sid.as_str()) { continue; }
            let Some(syn) = self.synapses.get(sid) else { continue };
            if syn.weight < min_weight { continue; }
            if let Some(other) = syn.other_end(id) {
                if let Some(n) = self.neurons.get(other) {
                    out.push((n, syn));
                }
            }
        }
        out
    }

    fn neighbors_out<'a>(&'a self, id: &str, min_weight: f64) -> Vec<(&'a Neuron, &'a Synapse)> {
        let Some(ids) = self.adj.get(id) else { return vec![] };
        let mut out = Vec::with_capacity(ids.len());
        let mut seen = HashSet::new();
        for sid in ids {
            if !seen.insert(sid.as_str()) { continue; }
            let Some(syn) = self.synapses.get(sid) else { continue };
            if syn.weight < min_weight { continue; }
            if syn.source_id != id && syn.direction != crate::types::Direction::Bi { continue; }
            let other = if syn.source_id == id { syn.target_id.as_str() } else { syn.source_id.as_str() };
            if let Some(n) = self.neurons.get(other) {
                out.push((n, syn));
            }
        }
        out
    }

    fn find_by_content_exact(&self, content: &str, ty: NeuronType) -> Option<Neuron> {
        let key = exact_key(ty, content);
        let id = self.exact.get(&key)?;
        self.neurons.get(id).cloned()
    }


    fn find_content_match(&self, tokens: &[String], limit: usize) -> Vec<Neuron> {
        if tokens.is_empty() || limit == 0 {
            return vec![];
        }
        let n_docs = self.fibers.len().max(1) as u32;
        let mut cand: HashSet<&String> = HashSet::new();
        for t in tokens {
            if let Some(ids) = self.inv.get(t) {
                cand.extend(ids.iter());
            }
        }
        let mut scored: Vec<(f64, &str)> = Vec::new();
        if !cand.is_empty() {
            scored.reserve(cand.len());
            for id in cand {
                let Some(n) = self.neurons.get(id) else { continue };
                let c = n.content.to_lowercase();
                let mut hits = 0.0;
                for t in tokens {
                    if c.contains(t.as_str()) {
                        let df = self.inv.get(t).map(|s| s.len() as u32).unwrap_or(1);
                        hits += crate::idf::idf(df, n_docs);
                        if n.is_anchor() { hits += 0.35; }
                    }
                }
                if hits > 0.0 {
                    scored.push((hits / tokens.len() as f64, id.as_str()));
                }
            }
        }
        if scored.is_empty() {
            for (id, n) in &self.neurons {
                let lower = n.content.to_lowercase();
                let mut hits = 0.0;
                for t in tokens {
                    if t.chars().count() < 3 { continue; }
                    if lower.contains(t.as_str()) {
                        hits += 0.6;
                        continue;
                    }
                    for w in lower.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() >= 3) {
                        if w.starts_with(t.as_str()) || t.starts_with(w) || edit1(w, t) {
                            hits += 0.6;
                            break;
                        }
                    }
                }
                if hits > 0.0 {
                    scored.push((hits / tokens.len() as f64, id.as_str()));
                }
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).filter_map(|(_, id)| self.neurons.get(id).cloned()).collect()
    }

    fn find_anchor_overlap(&self, tokens: &[String], exclude: &str) -> Vec<(Neuron, f64)> {
        if tokens.is_empty() {
            return vec![];
        }
        // Candidate anchors from inverted index only — never full-graph scan.
        let mut cand: HashSet<&String> = HashSet::new();
        for t in tokens {
            if let Some(ids) = self.inv.get(t) {
                // Cap per-token fanout so common words cannot explode work.
                for id in ids.iter().take(96) {
                    cand.insert(id);
                }
            }
        }
        let mut out: Vec<(Neuron, f64)> = Vec::new();
        for id in cand {
            if id.as_str() == exclude {
                continue;
            }
            let Some(n) = self.neurons.get(id) else { continue };
            if !n.is_anchor() {
                continue;
            }
            let c = crate::extract::tokenize(&n.content);
            let score = crate::extract::jaccard(tokens, &c);
            if score >= 0.04 {
                out.push((n.clone(), score));
            }
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(24);
        out
    }


    fn recent_fibers(&self, limit: usize) -> Vec<&Fiber> {
        if limit == 0 || self.fiber_order.is_empty() {
            return vec![];
        }
        let start = self.fiber_order.len().saturating_sub(limit);
        self.fiber_order[start..]
            .iter()
            .rev()
            .filter_map(|id| self.fibers.get(id))
            .collect()
    }

    fn token_df(&self, token: &str) -> u32 {
        self.inv.get(token).map(|s| s.len() as u32).unwrap_or(0)
    }
}

fn exact_key(ty: NeuronType, content: &str) -> (u8, String) {
    (ty as u8, content.to_lowercase())
}

fn edit1(a: &str, b: &str) -> bool {
    let aa: Vec<char> = a.chars().collect();
    let bb: Vec<char> = b.chars().collect();
    let (la, lb) = (aa.len(), bb.len());
    if la.abs_diff(lb) > 1 {
        return false;
    }
    if la == lb {
        return aa.iter().zip(bb.iter()).filter(|(x, y)| x != y).count() == 1;
    }
    let (shorter, longer) = if la < lb { (&aa, &bb) } else { (&bb, &aa) };
    let mut i = 0;
    let mut j = 0;
    let mut skipped = false;
    while i < shorter.len() && j < longer.len() {
        if shorter[i] == longer[j] {
            i += 1;
            j += 1;
        } else if !skipped {
            skipped = true;
            j += 1;
        } else {
            return false;
        }
    }
    true
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
