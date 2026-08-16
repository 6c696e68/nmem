//! Public Brain facade: remember / recall / causal / health / persist / forget / link.

use crate::causal::{self, CausalDir, CausalResult};
use crate::consolidation::{self, ConsolidationReport};
use crate::encoder::{self, EncodeError, EncodingResult, RememberOpts};
use crate::health::{self, HealthReport};
use crate::retrieval::{self, RecallOpts, RecallResult};
use crate::store::{MemoryStore, Store, StoreError};
use crate::types::{BrainSnapshot, MemoryType, Synapse, SynapseType, now_ms};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Brain {
    store: MemoryStore,
    path: Option<PathBuf>,
    warm: HashMap<String, f64>,
}

impl Brain {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            store: MemoryStore::named(name),
            path: None,
            warm: HashMap::new(),
        }
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            Ok(Self {
                store: MemoryStore::load(&path)?,
                path: Some(path),
                warm: HashMap::new(),
            })
        } else {
            let mut b = Self::new("default");
            b.path = Some(path);
            Ok(b)
        }
    }

    pub fn save(&self) -> Result<(), StoreError> {
        if let Some(p) = &self.path {
            self.store.save(p)?;
        }
        Ok(())
    }

    pub fn save_as(&mut self, path: impl AsRef<Path>) -> Result<(), StoreError> {
        let p = path.as_ref().to_path_buf();
        self.store.save(&p)?;
        self.path = Some(p);
        Ok(())
    }

    pub fn remember(&mut self, content: &str) -> Result<EncodingResult, EncodeError> {
        self.remember_typed(content, None, vec![], 5)
    }

    pub fn remember_typed(
        &mut self,
        content: &str,
        memory_type: Option<MemoryType>,
        tags: Vec<String>,
        priority: u8,
    ) -> Result<EncodingResult, EncodeError> {
        encoder::encode(
            &mut self.store,
            content,
            RememberOpts {
                memory_type,
                tags,
                priority,
            },
        )
    }

    pub fn recall(&mut self, query: &str) -> RecallResult {
        self.recall_opts(query, RecallOpts::default())
    }

    pub fn recall_opts(&mut self, query: &str, mut opts: RecallOpts) -> RecallResult {
        if opts.warm.is_empty() {
            opts.warm = self.warm.clone();
        }
        let r = retrieval::recall(&mut self.store, query, opts);
        for v in self.warm.values_mut() {
            *v *= 0.6;
        }
        for (id, a) in &r.activations {
            let e = self.warm.entry(id.clone()).or_insert(0.0);
            *e = e.max(a.activation_level * 0.75);
        }
        self.warm.retain(|_, v| *v > 0.05);
        r
    }

    pub fn session_size(&self) -> usize {
        self.warm.len()
    }

    /// Pack top memories into a token-budgeted prompt block.
    pub fn context(&mut self, query: &str, token_budget: usize) -> crate::context::ContextPack {
        crate::context::pack(&mut self.store, query, token_budget)
    }

    pub fn causal(&self, query: &str, hops: u32) -> CausalResult {
        causal::causal(&self.store, query, hops)
    }

    pub fn causes(&self, query: &str, hops: u32) -> CausalResult {
        causal::causal_dir(&self.store, query, hops, CausalDir::Causes)
    }

    pub fn effects(&self, query: &str, hops: u32) -> CausalResult {
        causal::causal_dir(&self.store, query, hops, CausalDir::Effects)
    }

    pub fn health(&self) -> HealthReport {
        health::health(&self.store)
    }

    pub fn consolidate(&mut self) -> ConsolidationReport {
        consolidation::consolidate(&mut self.store)
    }

    pub fn forget(&mut self, query_or_id: &str) -> Option<String> {
        let q = query_or_id.trim();
        if q.is_empty() {
            return None;
        }
        if self.store.get_fiber(q).is_some() {
            self.store.remove_fiber(q);
            return Some(q.to_string());
        }
        let hits = self.store.find_content_match(&crate::extract::tokenize(q), 8);
        if hits.is_empty() {
            return None;
        }
        let qtoks = crate::extract::tokenize(q);
        let mut best: Option<(f64, String)> = None;
        for n in hits {
            for f in self.store.fibers() {
                if f.anchor_neuron_id != n.id && !f.neuron_ids.contains(&n.id) {
                    continue;
                }
                let sim = crate::extract::jaccard(&qtoks, &crate::extract::tokenize(&f.summary));
                if best.as_ref().map(|(s, _)| sim > *s).unwrap_or(true) {
                    best = Some((sim, f.id.clone()));
                }
            }
        }
        let (score, id) = best?;
        if score < 0.08 && qtoks.len() > 1 {
            return None;
        }
        self.store.remove_fiber(&id);
        Some(id)
    }

    /// Resolve one side of a link: fiber id, neuron id, or content candidates.
    /// Exact ids yield a single hit; free text yields ranked candidates so short
    /// queries that appear in many anchors can still pair with a distinct peer.
    fn resolve_link_candidates(&self, query_or_id: &str) -> Vec<String> {
        let q = query_or_id.trim();
        if q.is_empty() {
            return vec![];
        }
        if let Some(f) = self.store.get_fiber(q) {
            return vec![f.anchor_neuron_id.clone()];
        }
        if self.store.get_neuron(q).is_some() {
            return vec![q.to_string()];
        }
        let hits = self.store.find_content_match(&crate::extract::tokenize(q), 12);
        let mut ids = Vec::new();
        // Anchors first, then other neurons — stable preference without dropping alternatives.
        for n in hits.iter().filter(|n| n.is_anchor()) {
            ids.push(n.id.clone());
        }
        for n in hits.iter().filter(|n| !n.is_anchor()) {
            if !ids.iter().any(|id| id == &n.id) {
                ids.push(n.id.clone());
            }
        }
        ids
    }

    pub fn link(
        &mut self,
        a_query: &str,
        b_query: &str,
        ty: SynapseType,
        weight: f64,
    ) -> Option<Synapse> {
        let hits_a = self.resolve_link_candidates(a_query);
        let hits_b = self.resolve_link_candidates(b_query);
        let mut pair = None;
        for a in &hits_a {
            for b in &hits_b {
                if a != b {
                    pair = Some((a.clone(), b.clone()));
                    break;
                }
            }
            if pair.is_some() {
                break;
            }
        }
        let (aid, bid) = pair?;
        let s = self
            .store
            .add_synapse(Synapse::create(&aid, &bid, ty, weight.clamp(0.1, 1.0)));
        if let Some(inv) = ty.inverse() {
            let _ = self
                .store
                .add_synapse(Synapse::create(&bid, &aid, inv, weight * 0.95));
        }
        Some(s)
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut MemoryStore {
        &mut self.store
    }

    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.store.snapshot())
    }

    pub fn import_snapshot(&mut self, snap: BrainSnapshot) {
        self.store = MemoryStore::from_snapshot(snap);
    }

    pub fn decay_synapses(&mut self) -> u32 {
        let now = now_ms();
        let ids: Vec<String> = self.store.synapses().into_iter().map(|s| s.id.clone()).collect();
        let mut n = 0u32;
        for id in ids {
            if let Some(s) = self.store.get_synapse_mut(&id) {
                s.time_decay(now);
                n += 1;
            }
        }
        n
    }
}
