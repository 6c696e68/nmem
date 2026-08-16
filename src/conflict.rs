//! Encode-time conflict detection — port of `engine/conflict_detection.py`.

use crate::extract::{jaccard, keywords, tokenize};
use crate::store::Store;
use crate::types::{MemoryStatus, MemoryType, Neuron, Synapse, SynapseType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConflictKind {
    FactualContradiction,
    DecisionReversal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub kind: ConflictKind,
    pub existing_id: String,
    pub existing_content: String,
    pub new_content: String,
    pub confidence: f64,
}

/// Detect contradictions / reversed decisions against existing anchors.
pub fn detect(store: &dyn Store, content: &str, memory_type: MemoryType) -> Vec<Conflict> {
    let tokens = tokenize(content);
    if tokens.len() < 3 {
        return vec![];
    }
    let kws = keywords(content, 8);
    let overlaps = store.find_anchor_overlap(&tokens, "");
    let mut out = Vec::new();
    for (other, score) in overlaps.into_iter().take(12) {
        if score < 0.18 {
            continue;
        }
        if memory_type == MemoryType::Decision {
            let other_type = other
                .metadata
                .get("memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if other_type == "decision" {
                let ok = keywords(&other.content, 8);
                let shared = jaccard(&kws, &ok);
                if shared >= 0.25 && different_conclusion(&kws, &ok) {
                    out.push(Conflict {
                        kind: ConflictKind::DecisionReversal,
                        existing_id: other.id.clone(),
                        existing_content: other.content.clone(),
                        new_content: content.to_string(),
                        confidence: (shared + score) / 2.0,
                    });
                }
            }
        }
        if negation_flip(content, &other.content) {
            out.push(Conflict {
                kind: ConflictKind::FactualContradiction,
                existing_id: other.id.clone(),
                existing_content: other.content.clone(),
                new_content: content.to_string(),
                confidence: score.max(0.6),
            });
        }
    }
    out
}

pub fn apply(store: &mut dyn Store, conflicts: &[Conflict], new_anchor: &Neuron) -> Vec<Synapse> {
    let mut created = Vec::new();
    for c in conflicts {
        let ty = match c.kind {
            ConflictKind::DecisionReversal => SynapseType::Supersedes,
            ConflictKind::FactualContradiction => SynapseType::Contradicts,
        };
        let s = store.add_synapse(Synapse::create(&new_anchor.id, &c.existing_id, ty, c.confidence.min(0.9)));
        created.push(s);
        // Anti-Hebbian on existing edges into the disputed memory.
        let weaken: Vec<String> = store
            .neighbors(&c.existing_id, 0.0)
            .into_iter()
            .map(|(_, syn)| syn.id.clone())
            .collect();
        for id in weaken {
            if let Some(syn) = store.get_synapse_mut(&id) {
                let u = crate::hebbian::anti_hebbian_update(
                    syn.weight,
                    c.confidence,
                    crate::hebbian::LearningConfig::default(),
                );
                syn.weight = u.new_weight;
            }
        }
        if let Some(inv) = ty.inverse() {
            let _ = store.add_synapse(Synapse::create(&c.existing_id, &new_anchor.id, inv, c.confidence.min(0.85)));
        }
        if let Some(n) = store.get_neuron(&c.existing_id).cloned() {
            let mut n = n;
            n.metadata
                .insert("_disputed".into(), serde_json::json!(true));
            store.add_neuron(n);
        }
        if c.kind == ConflictKind::DecisionReversal {
            let fids: Vec<String> = store
                .fibers()
                .into_iter()
                .filter(|f| f.anchor_neuron_id == c.existing_id)
                .map(|f| f.id.clone())
                .collect();
            for fid in fids {
                if let Some(f) = store.get_fiber_mut(&fid) {
                    f.status = MemoryStatus::Superseded;
                }
            }
        }
    }
    created
}

fn different_conclusion(a: &[String], b: &[String]) -> bool {
    let aa: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let bb: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let only_a = aa.difference(&bb).count();
    let only_b = bb.difference(&aa).count();
    only_a >= 1 && only_b >= 1
}

fn negation_flip(a: &str, b: &str) -> bool {
    let al = a.to_lowercase();
    let bl = b.to_lowercase();
    let a_neg = al.contains(" not ") || al.contains("n't") || al.contains("không");
    let b_neg = bl.contains(" not ") || bl.contains("n't") || bl.contains("không");
    a_neg != b_neg && jaccard(&tokenize(a), &tokenize(b)) >= 0.35
}
