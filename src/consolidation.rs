//! Decay + prune + real fiber merge. Port of free-tier consolidation.

use crate::extract::{jaccard, tokenize};
use crate::store::Store;
use crate::types::{Fiber, MemoryStatus, SynapseType, now_ms};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const PROTECTED: &[SynapseType] = &[
    SynapseType::CausedBy,
    SynapseType::LeadsTo,
    SynapseType::Enables,
    SynapseType::Prevents,
    SynapseType::ResolvedBy,
    SynapseType::Supersedes,
];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationReport {
    pub synapses_decayed: u32,
    pub synapses_pruned: u32,
    pub fibers_merged: u32,
    pub fibers_removed: u32,
    pub neurons_pruned: u32,
    pub neurons_touched: u32,
    pub stages_promoted: u32,
    pub conductivity_decayed: u32,
    pub expired: u32,
}

pub fn consolidate<S: Store>(store: &mut S) -> ConsolidationReport {
    let now = now_ms();
    let prune_th = store.meta().config.consolidation_prune_threshold;
    let merge_th = store.meta().config.merge_overlap_threshold;

    let mut report = ConsolidationReport::default();

    let ids: Vec<String> = store.neurons().into_iter().map(|n| n.id.clone()).collect();
    for id in &ids {
        if let Some(st) = store.get_state_mut(id) {
            if let Some(last) = st.last_activated {
                let secs = (now.saturating_sub(last) as f64) / 1000.0;
                st.decay(secs);
                report.neurons_touched += 1;
            }
        }
    }

    let syn_ids: Vec<String> = store.synapses().into_iter().map(|s| s.id.clone()).collect();
    let mut prune = Vec::new();
    for id in syn_ids {
        let should_prune = if let Some(s) = store.get_synapse_mut(&id) {
            let protected = PROTECTED.contains(&s.type_);
            s.time_decay(now);
            report.synapses_decayed += 1;
            !protected && s.weight < prune_th && s.reinforced_count == 0
        } else {
            false
        };
        if should_prune {
            prune.push(id);
        }
    }
    for id in &prune {
        if store.remove_synapse(id) {
            report.synapses_pruned += 1;
        }
    }

    let merged = merge_fibers(store, merge_th);
    report.fibers_merged = merged.0;
    report.fibers_removed = merged.1;

    report.neurons_pruned = prune_orphans(store);

    let fids: Vec<String> = store.fibers().into_iter().map(|f| f.id.clone()).collect();
    for id in fids {
        if let Some(f) = store.get_fiber_mut(&id) {
            if crate::stages::maybe_promote(f, now) {
                report.stages_promoted += 1;
            }
            let before = f.conductivity;
            crate::stages::decay_conductivity(f, now);
            if f.conductivity < before {
                report.conductivity_decayed += 1;
            }
            if f.status == MemoryStatus::Active {
                if let Some(e) = f.expires_at {
                    if now >= e {
                        f.status = MemoryStatus::Expired;
                        report.expired += 1;
                    }
                }
            }
        }
    }

    report
}

fn conflicted<S: Store>(store: &S, a: &str, b: &str) -> bool {
    store.neighbors(a, 0.0).into_iter().any(|(n, s)| {
        n.id == b
            && matches!(
                s.type_,
                SynapseType::Contradicts
                    | SynapseType::Supersedes
                    | SynapseType::FalsifiedBy
                    | SynapseType::EvidenceAgainst
            )
    })
}

fn merge_fibers<S: Store>(store: &mut S, threshold: f64) -> (u32, u32) {
    let fibers: Vec<Fiber> = store.fibers().into_iter().cloned().collect();
    let mut used: HashSet<String> = HashSet::new();
    let mut merged = 0u32;
    let mut removed = 0u32;
    for i in 0..fibers.len() {
        if used.contains(&fibers[i].id) {
            continue;
        }
        let ai = tokenize(&fibers[i].summary);
        let mut group = vec![i];
        for j in (i + 1)..fibers.len() {
            if used.contains(&fibers[j].id) {
                continue;
            }
            if fibers[i].status != MemoryStatus::Active || fibers[j].status != MemoryStatus::Active {
                continue;
            }
            if fibers[i].memory_type != fibers[j].memory_type {
                continue;
            }
            if conflicted(store, &fibers[i].anchor_neuron_id, &fibers[j].anchor_neuron_id) {
                continue;
            }
            let aj = tokenize(&fibers[j].summary);
            if jaccard(&ai, &aj) >= threshold {
                group.push(j);
            }
        }
        if group.len() < 2 {
            continue;
        }
        group.sort_by(|&a, &b| {
            fibers[b]
                .salience
                .partial_cmp(&fibers[a].salience)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(fibers[b].created_at.cmp(&fibers[a].created_at))
        });
        let winner = group[0];
        used.insert(fibers[winner].id.clone());
        let mut neuron_ids = fibers[winner].neuron_ids.clone();
        let mut synapse_ids = fibers[winner].synapse_ids.clone();
        let mut tags = fibers[winner].tags.clone();
        for &loser in &group[1..] {
            used.insert(fibers[loser].id.clone());
            for id in &fibers[loser].neuron_ids {
                if !neuron_ids.contains(id) {
                    neuron_ids.push(id.clone());
                }
            }
            for id in &fibers[loser].synapse_ids {
                if !synapse_ids.contains(id) {
                    synapse_ids.push(id.clone());
                }
            }
            for t in &fibers[loser].tags {
                if !tags.contains(t) {
                    tags.push(t.clone());
                }
            }
            store.remove_fiber(&fibers[loser].id);
            removed += 1;
        }
        if let Some(w) = store.get_fiber_mut(&fibers[winner].id) {
            w.neuron_ids = neuron_ids;
            w.synapse_ids = synapse_ids;
            w.tags = tags;
            w.salience = (w.salience + 0.1).min(1.0);
            w.frequency += group.len() as u32 - 1;
        }
        merged += 1;
    }
    (merged, removed)
}

fn prune_orphans<S: Store>(store: &mut S) -> u32 {
    let live: HashSet<String> = store
        .fibers()
        .into_iter()
        .flat_map(|f| f.neuron_ids.iter().cloned())
        .collect();
    let ids: Vec<String> = store
        .neurons()
        .into_iter()
        .filter(|n| !n.is_anchor() && !live.contains(&n.id))
        .map(|n| n.id.clone())
        .collect();
    let mut n = 0u32;
    for id in ids {
        if store.remove_neuron(&id) {
            n += 1;
        }
    }
    n
}

pub fn merge_candidates<S: Store>(store: &S, threshold: f64) -> Vec<(String, String, f64)> {
    let fibers: Vec<_> = store.fibers().into_iter().cloned().collect();
    let mut pairs = Vec::new();
    for i in 0..fibers.len() {
        let ai = tokenize(&fibers[i].summary);
        for j in (i + 1)..fibers.len() {
            if fibers[i].memory_type != fibers[j].memory_type {
                continue;
            }
            let aj = tokenize(&fibers[j].summary);
            let s = jaccard(&ai, &aj);
            if s >= threshold {
                pairs.push((fibers[i].id.clone(), fibers[j].id.clone(), s));
            }
        }
    }
    pairs
}
