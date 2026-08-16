//! Causal / temporal traversal — port of `engine/causal_traversal.py`.
//! BFS (not greedy single-path) with cycle guard + geometric-mean confidence.

use crate::extract::tokenize;
use crate::store::Store;
use crate::types::{SynapseType, now_ms};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

const CAUSE_TYPES: &[SynapseType] = &[SynapseType::CausedBy];
const EFFECT_TYPES: &[SynapseType] = &[SynapseType::LeadsTo, SynapseType::Enables];
const ANY_CAUSAL: &[SynapseType] = &[
    SynapseType::CausedBy,
    SynapseType::LeadsTo,
    SynapseType::ResolvedBy,
    SynapseType::Enables,
    SynapseType::Prevents,
    SynapseType::EvidenceFor,
    SynapseType::EvidenceAgainst,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalDir {
    Causes,
    Effects,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalHop {
    pub from_id: String,
    pub from_content: String,
    pub synapse: String,
    pub to_id: String,
    pub to_content: String,
    pub weight: f64,
    pub depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalResult {
    pub query: String,
    pub seed: Option<String>,
    pub seed_id: Option<String>,
    pub direction: CausalDir,
    pub chain: Vec<CausalHop>,
    pub confidence: f64,
}

pub fn causal<S: Store>(store: &S, query: &str, max_hops: u32) -> CausalResult {
    causal_dir(store, query, max_hops, CausalDir::Causes)
}

pub fn causal_dir<S: Store>(store: &S, query: &str, max_hops: u32, direction: CausalDir) -> CausalResult {
    let tokens = tokenize(query);
    let mut matches = store.find_content_match(&tokens, 8);
    matches.sort_by(|a, b| {
        let ca = causal_degree(store, &a.id);
        let cb = causal_degree(store, &b.id);
        cb.cmp(&ca)
            .then_with(|| b.is_anchor().cmp(&a.is_anchor()))
    });
    let Some(seed) = matches.into_iter().next() else {
        return CausalResult {
            query: query.into(),
            seed: None,
            seed_id: None,
            direction,
            chain: vec![],
            confidence: 0.0,
        };
    };

    let types: &[SynapseType] = match direction {
        CausalDir::Causes => CAUSE_TYPES,
        CausalDir::Effects => EFFECT_TYPES,
    };

    let mut chain = Vec::new();
    let mut visited = HashSet::new();
    visited.insert(seed.id.clone());
    let mut q = VecDeque::new();
    q.push_back((seed.id.clone(), 0u32));

    while let Some((current, depth)) = q.pop_front() {
        if depth >= max_hops {
            continue;
        }
        let mut neigh: Vec<_> = store
            .neighbors_out(&current, 0.1)
            .into_iter()
            .filter(|(_, s)| types.contains(&s.type_))
            .collect();
        // Fallback: if no directed cause edge, allow inverse walk of LeadsTo / CausedBy.
        if neigh.is_empty() {
            neigh = store
                .neighbors(&current, 0.1)
                .into_iter()
                .filter(|(_, s)| ANY_CAUSAL.contains(&s.type_))
                .collect();
        }
        neigh.sort_by(|a, b| {
            let wa = a.1.weight * a.1.type_.role_multiplier();
            let wb = b.1.weight * b.1.type_.role_multiplier();
            wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (neuron, syn) in neigh {
            if !visited.insert(neuron.id.clone()) {
                continue;
            }
            let from = store.get_neuron(&current);
            chain.push(CausalHop {
                from_id: current.clone(),
                from_content: from.map(|n| n.content.clone()).unwrap_or_default(),
                synapse: syn.type_.as_str().to_string(),
                to_id: neuron.id.clone(),
                to_content: neuron.content.clone(),
                weight: syn.weight,
                depth,
            });
            q.push_back((neuron.id.clone(), depth + 1));
        }
    }

    let confidence = if chain.is_empty() {
        0.0
    } else {
        let prod: f64 = chain.iter().map(|h| h.weight.max(1e-9)).product();
        prod.powf(1.0 / chain.len() as f64)
    };

    CausalResult {
        query: query.into(),
        seed: Some(seed.content),
        seed_id: Some(seed.id),
        direction,
        chain,
        confidence,
    }
}

fn causal_degree<S: Store>(store: &S, id: &str) -> usize {
    store
        .neighbors_out(id, 0.1)
        .into_iter()
        .filter(|(_, s)| ANY_CAUSAL.contains(&s.type_))
        .count()
}

/// Temporal sequence via BEFORE / AFTER.
pub fn sequence<S: Store>(store: &S, query: &str, max_hops: u32, forward: bool) -> CausalResult {
    let tokens = tokenize(query);
    let matches = store.find_content_match(&tokens, 8);
    let Some(seed) = matches.into_iter().next() else {
        return CausalResult {
            query: query.into(),
            seed: None,
            seed_id: None,
            direction: if forward {
                CausalDir::Effects
            } else {
                CausalDir::Causes
            },
            chain: vec![],
            confidence: 0.0,
        };
    };
    let want = if forward {
        SynapseType::Before
    } else {
        SynapseType::After
    };
    let mut chain = Vec::new();
    let mut visited = HashSet::new();
    visited.insert(seed.id.clone());
    let mut current = seed.id.clone();
    for depth in 0..max_hops {
        let best = store
            .neighbors_out(&current, 0.1)
            .into_iter()
            .filter(|(_, s)| s.type_ == want)
            .max_by(|a, b| a.1.weight.partial_cmp(&b.1.weight).unwrap_or(std::cmp::Ordering::Equal));
        let Some((neuron, syn)) = best else {
            break;
        };
        if !visited.insert(neuron.id.clone()) {
            break;
        }
        let from = store.get_neuron(&current);
        chain.push(CausalHop {
            from_id: current.clone(),
            from_content: from.map(|n| n.content.clone()).unwrap_or_default(),
            synapse: syn.type_.as_str().to_string(),
            to_id: neuron.id.clone(),
            to_content: neuron.content.clone(),
            weight: syn.weight,
            depth,
        });
        current = neuron.id.clone();
    }
    CausalResult {
        query: query.into(),
        seed: Some(seed.content),
        seed_id: Some(seed.id),
        direction: if forward {
            CausalDir::Effects
        } else {
            CausalDir::Causes
        },
        chain,
        confidence: 0.0,
    }
}

pub fn now() -> u64 {
    now_ms()
}
