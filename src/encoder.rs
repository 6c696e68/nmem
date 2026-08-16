//! Encode an experience into neurons + synapses + a fiber.
//! Port of MemoryEncoder / pipeline_steps (lean cognitive path).

use crate::conflict;
use crate::extract::{
    detect_relations, extract_actions, extract_entities, extract_intents, extract_places,
    extract_relations, keywords, suggest_memory_type, tokenize,
};
use crate::store::Store;
use crate::types::{Fiber, MemoryType, Neuron, NeuronType, Synapse, SynapseType, now_ms};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingResult {
    pub fiber: Fiber,
    pub neurons_created: Vec<Neuron>,
    pub neurons_linked: Vec<String>,
    pub synapses_created: Vec<Synapse>,
    pub conflicts: Vec<conflict::Conflict>,
}

#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("empty memory")]
    Empty,
}

pub struct RememberOpts {
    pub memory_type: Option<MemoryType>,
    pub tags: Vec<String>,
    pub priority: u8,
}

impl Default for RememberOpts {
    fn default() -> Self {
        Self {
            memory_type: None,
            tags: vec![],
            priority: 5,
        }
    }
}

pub fn encode(
    store: &mut dyn Store,
    content: &str,
    opts: RememberOpts,
) -> Result<EncodingResult, EncodeError> {
    let content = content.trim();
    if content.is_empty() {
        return Err(EncodeError::Empty);
    }
    let memory_type = opts
        .memory_type
        .unwrap_or_else(|| suggest_memory_type(content));
    let mut created: Vec<Neuron> = Vec::new();
    let mut linked: Vec<String> = Vec::new();
    let mut synapses: Vec<Synapse> = Vec::new();

    let raw_kws = keywords(content, 10);
    let anchor = Neuron::create(NeuronType::Concept, truncate_anchor(content, 500))
        .with_meta("is_anchor", serde_json::json!(true))
        .with_meta("memory_type", serde_json::json!(memory_type.as_str()))
        .with_meta("_raw_keywords", serde_json::json!(raw_kws));
    let anchor_id = anchor.id.clone();
    store.add_neuron(anchor.clone());
    created.push(anchor.clone());

    let mut time_ns = Vec::new();
    let now = now_ms();
    for hint in crate::temporal::extract_windows(content, now) {
        if let Some(ex) = store.find_by_content_exact(&hint.label, NeuronType::Time) {
            linked.push(ex.id.clone());
            time_ns.push(ex);
        } else {
            let n = Neuron::create(NeuronType::Time, hint.label.clone())
                .with_meta("t_start", serde_json::json!(hint.start))
                .with_meta("t_end", serde_json::json!(hint.end));
            store.add_neuron(n.clone());
            created.push(n.clone());
            time_ns.push(n);
        }
    }

    let mut entity_ns = Vec::new();
    for e in extract_entities(content) {
        if e.eq_ignore_ascii_case(content) {
            continue;
        }
        if let Some(ex) = store.find_by_content_exact(&e, NeuronType::Entity) {
            linked.push(ex.id.clone());
            entity_ns.push(ex);
        } else {
            let n = Neuron::create(NeuronType::Entity, e);
            store.add_neuron(n.clone());
            created.push(n.clone());
            entity_ns.push(n);
        }
    }

    let entity_lower: Vec<String> = entity_ns.iter().map(|n| n.content.to_lowercase()).collect();
    let concept_limit = (content.chars().count() / 100).clamp(3, 12);
    let mut concept_ns = Vec::new();
    for kw in keywords(content, concept_limit) {
        if kw.chars().count() < 4 {
            continue;
        }
        if entity_lower.iter().any(|e| e == &kw.to_lowercase()) {
            continue;
        }
        if let Some(ex) = store.find_by_content_exact(&kw, NeuronType::Concept) {
            if ex.id != anchor_id {
                linked.push(ex.id.clone());
                concept_ns.push(ex);
            }
        } else {
            let n = Neuron::create(NeuronType::Concept, kw);
            store.add_neuron(n.clone());
            created.push(n.clone());
            concept_ns.push(n);
        }
    }

    let mut action_ns = Vec::new();
    for a in extract_actions(content) {
        if let Some(ex) = store.find_by_content_exact(&a, NeuronType::Action) {
            linked.push(ex.id.clone());
            action_ns.push(ex);
        } else {
            let n = Neuron::create(NeuronType::Action, a);
            store.add_neuron(n.clone());
            created.push(n.clone());
            action_ns.push(n);
        }
    }

    let mut place_ns = Vec::new();
    for p in extract_places(content) {
        if let Some(ex) = store.find_by_content_exact(&p, NeuronType::Spatial) {
            linked.push(ex.id.clone());
            place_ns.push(ex);
        } else {
            let n = Neuron::create(NeuronType::Spatial, p);
            store.add_neuron(n.clone());
            created.push(n.clone());
            place_ns.push(n);
        }
    }

    let mut intent_ns = Vec::new();
    for intent in extract_intents(content) {
        if let Some(ex) = store.find_by_content_exact(&intent, NeuronType::Intent) {
            linked.push(ex.id.clone());
            intent_ns.push(ex);
        } else {
            let n = Neuron::create(NeuronType::Intent, intent);
            store.add_neuron(n.clone());
            created.push(n.clone());
            intent_ns.push(n);
        }
    }

    let tokens = tokenize(content);
    let detected = detect_relations(content);
    let overlaps = store.find_anchor_overlap(&tokens, &anchor_id);

    for t in &time_ns {
        link(store, &mut synapses, &anchor_id, &t.id, SynapseType::HappenedAt, 0.9);
    }
    let hay = content.to_lowercase();
    for e in &entity_ns {
        let mentions = hay.matches(&e.content.to_lowercase()).count();
        let w = (0.7 + 0.05 * mentions as f64).min(0.95);
        link(store, &mut synapses, &anchor_id, &e.id, SynapseType::Involves, w);
    }
    for c in &concept_ns {
        link(store, &mut synapses, &anchor_id, &c.id, SynapseType::RelatedTo, 0.55);
    }
    for a in &action_ns {
        link(store, &mut synapses, &anchor_id, &a.id, SynapseType::Involves, 0.6);
    }
    for p in &place_ns {
        link(store, &mut synapses, &anchor_id, &p.id, SynapseType::AtLocation, 0.8);
    }
    for i in &intent_ns {
        link(store, &mut synapses, &anchor_id, &i.id, SynapseType::RelatedTo, 0.55);
    }

    let cluster: Vec<&Neuron> = entity_ns.iter().chain(concept_ns.iter()).collect();
    let cluster_n = cluster.len().min(6);
    for i in 0..cluster_n {
        for j in (i + 1)..cluster_n {
            link(store, &mut synapses, &cluster[i].id, &cluster[j].id, SynapseType::CoOccurs, 0.45);
        }
    }

    // Overlap = related, never spray CausedBy onto every neighbour.
    for (other, score) in overlaps.iter().take(8) {
        if *score < 0.12 {
            continue;
        }
        let w = (0.35 + score * 0.5).min(0.85);
        link(store, &mut synapses, &anchor_id, &other.id, SynapseType::RelatedTo, w);
        linked.push(other.id.clone());
    }

    // Clause-level wiring: "X because Y" → concept neurons + typed edge.
    let clause_rels = extract_relations(content);
    for rel in &clause_rels {
        let src_n = ensure_concept(store, &rel.source_span, &mut created, &mut linked);
        let tgt_n = ensure_concept(store, &rel.target_span, &mut created, &mut linked);
        link(store, &mut synapses, &src_n, &tgt_n, rel.type_, rel.confidence);
        link(store, &mut synapses, &anchor_id, &src_n, SynapseType::RelatedTo, 0.5);
        link(store, &mut synapses, &anchor_id, &tgt_n, SynapseType::RelatedTo, 0.5);
        if matches!(
            rel.type_,
            SynapseType::CausedBy | SynapseType::LeadsTo | SynapseType::Enables | SynapseType::Prevents
        ) {
            // Fiber anchor must carry the typed edge — walk starts at anchors.
            link(store, &mut synapses, &anchor_id, &tgt_n, rel.type_, rel.confidence);
        }
        // Attach overlapping events that *are* the effect/cause, not every neighbour.
        if matches!(
            rel.type_,
            SynapseType::CausedBy | SynapseType::LeadsTo | SynapseType::Enables | SynapseType::Prevents
        ) {
            let src_toks = tokenize(&rel.source_span);
            let tgt_toks = tokenize(&rel.target_span);
            for (other, score) in overlaps.iter().take(8) {
                if *score < 0.08 {
                    continue;
                }
                let ot = tokenize(&other.content);
                if crate::extract::jaccard(&ot, &src_toks) >= 0.10 {
                    link(store, &mut synapses, &other.id, &tgt_n, rel.type_, rel.confidence * 0.9);
                } else if crate::extract::jaccard(&ot, &tgt_toks) >= 0.10 {
                    link(store, &mut synapses, &src_n, &other.id, rel.type_, rel.confidence * 0.9);
                }
            }
        }
    }

    // Inherit causal outgoing from overlapping memories onto this event.
    for (other, score) in overlaps.iter().take(8) {
        if *score < 0.10 {
            continue;
        }
        let inherited: Vec<(String, SynapseType, f64)> = store
            .neighbors_out(&other.id, 0.15)
            .into_iter()
            .filter(|(_, s)| {
                matches!(
                    s.type_,
                    SynapseType::CausedBy
                        | SynapseType::LeadsTo
                        | SynapseType::Enables
                        | SynapseType::Prevents
                )
            })
            .filter_map(|(n, s)| {
                if n.id == anchor_id {
                    None
                } else {
                    Some((n.id.clone(), s.type_, s.weight))
                }
            })
            .collect();
        for (tid, ty, w) in inherited {
            link(store, &mut synapses, &anchor_id, &tid, ty, (w * 0.85).max(0.3));
        }
        // Push this memory's causal outgoing onto overlapping events (encode-order).
        let outgoing: Vec<(String, SynapseType, f64)> = store
            .neighbors_out(&anchor_id, 0.15)
            .into_iter()
            .filter(|(_, s)| {
                matches!(
                    s.type_,
                    SynapseType::CausedBy
                        | SynapseType::LeadsTo
                        | SynapseType::Enables
                        | SynapseType::Prevents
                )
            })
            .map(|(n, s)| (n.id.clone(), s.type_, s.weight))
            .collect();
        for (tid, ty, w) in outgoing {
            if tid == other.id {
                continue;
            }
            link(store, &mut synapses, &other.id, &tid, ty, (w * 0.85).max(0.3));
        }
    }

    let conflicts = conflict::detect(store, content, memory_type);
    let conflict_syns = conflict::apply(store, &conflicts, &anchor);
    synapses.extend(conflict_syns);

    let mut neuron_ids: Vec<String> = created.iter().map(|n| n.id.clone()).collect();
    for id in &linked {
        if !neuron_ids.contains(id) {
            neuron_ids.push(id.clone());
        }
    }
    let synapse_ids: Vec<String> = synapses.iter().map(|s| s.id.clone()).collect();
    let possible = (neuron_ids.len() * neuron_ids.len().saturating_sub(1)) as f64 / 2.0;
    let coherence = synapse_ids.len() as f64 / possible.max(1.0);
    let salience = (coherence + 0.3).min(1.0);

    let mut tags = opts.tags;
    for e in &entity_ns {
        let t = e.content.to_lowercase();
        if !tags.contains(&t) {
            tags.push(t);
        }
    }
    for kw in raw_kws.iter().take(4) {
        if !tags.contains(kw) {
            tags.push(kw.clone());
        }
    }

    let mut fiber = Fiber::create(
        neuron_ids,
        synapse_ids,
        anchor_id.clone(),
        content.to_string(),
        memory_type,
        tags,
    );
    fiber.salience = salience;
    fiber.priority = opts.priority.clamp(0, 10);
    fiber.tier = memory_type.default_tier();
    fiber.stage = crate::types::MemoryStage::ShortTerm;
    fiber.trust = if matches!(
        memory_type,
        MemoryType::Boundary | MemoryType::Instruction | MemoryType::Preference
    ) {
        0.9
    } else {
        0.8
    };
    if memory_type == MemoryType::Hypothesis || memory_type == MemoryType::Prediction {
        fiber.belief = 0.5;
    }
    store.add_fiber(fiber.clone());

    // Simhash ALIAS — O(1) recent window + overlap hits only.
    let my_toks = tokenize(content);
    let my_hash = crate::simhash::simhash(&my_toks);
    let overlap_ids: std::collections::HashSet<&str> =
        overlaps.iter().take(12).map(|(n, _)| n.id.as_str()).collect();
    let mut alias_cands: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    // Recent fibers (newest first) — constant window.
    for f in store.recent_fibers(64) {
        if f.id == fiber.id || f.memory_type != memory_type {
            continue;
        }
        if seen.insert(f.id.clone()) {
            alias_cands.push((f.id.clone(), f.summary.clone()));
        }
    }
    // Prefer also any overlap-anchor fiber if not already included (via recent is enough
    // for near-duplicates; overlap list covers semantic neighbours of this encode).
    let _ = overlap_ids;
    for (oid, summary) in alias_cands.into_iter().take(48) {
        let oh = crate::simhash::simhash(&tokenize(&summary));
        if crate::simhash::hamming(my_hash, oh) <= 12 {
            let target = store
                .get_fiber(&oid)
                .map(|f| f.anchor_neuron_id.clone())
                .unwrap_or(oid);
            let s = store.add_synapse(Synapse::create(
                &anchor_id,
                &target,
                SynapseType::Alias,
                0.8,
            ));
            synapses.push(s);
        }
    }

    // Bayesian update: this memory as evidence for/against overlapping hypotheses.
    let ev_for = detected.iter().any(|d| d.type_ == SynapseType::EvidenceFor)
        || extract_relations(content)
            .iter()
            .any(|r| r.type_ == SynapseType::EvidenceFor);
    let ev_against = detected.iter().any(|d| d.type_ == SynapseType::EvidenceAgainst)
        || extract_relations(content)
            .iter()
            .any(|r| r.type_ == SynapseType::EvidenceAgainst);
    if ev_for || ev_against {
        let hyp_meta: Vec<(String, String)> = store
            .fibers()
            .into_iter()
            .filter(|f| {
                f.id != fiber.id
                    && matches!(
                        f.memory_type,
                        MemoryType::Hypothesis | MemoryType::Prediction
                    )
                    && crate::extract::jaccard(&my_toks, &tokenize(&f.summary)) >= 0.12
            })
            .map(|f| (f.id.clone(), f.anchor_neuron_id.clone()))
            .collect();
        for (hid, ha) in hyp_meta {
            if let Some(h) = store.get_fiber_mut(&hid) {
                h.belief = if ev_against {
                    crate::evidence::evidence_against(h.belief, 0.7)
                } else {
                    crate::evidence::evidence_for(h.belief, 0.6)
                };
            }
            let ty = if ev_against {
                SynapseType::EvidenceAgainst
            } else {
                SynapseType::EvidenceFor
            };
            let s = store.add_synapse(Synapse::create(&anchor_id, &ha, ty, 0.7));
            synapses.push(s);
        }
    }

    let steep = store.meta().config.sigmoid_steepness;
    if let Some(st) = store.get_state_mut(&anchor.id) {
        st.activate(1.0, now_ms(), steep);
    }

    Ok(EncodingResult {
        fiber,
        neurons_created: created,
        neurons_linked: linked,
        synapses_created: synapses,
        conflicts,
    })
}

fn link(
    store: &mut dyn Store,
    synapses: &mut Vec<Synapse>,
    source: &str,
    target: &str,
    ty: SynapseType,
    w: f64,
) {
    let s = store.add_synapse(Synapse::create(source, target, ty, w));
    if let Some(inv) = ty.inverse() {
        let _ = store.add_synapse(Synapse::create(target, source, inv, w * 0.95));
    }
    synapses.push(s);
}

fn ensure_concept(
    store: &mut dyn Store,
    span: &str,
    created: &mut Vec<Neuron>,
    linked: &mut Vec<String>,
) -> String {
    let span = truncate_anchor(span.trim(), 80);
    if let Some(ex) = store.find_by_content_exact(&span, NeuronType::Concept) {
        if !linked.contains(&ex.id) {
            linked.push(ex.id.clone());
        }
        return ex.id;
    }
    let n = Neuron::create(NeuronType::Concept, span);
    let id = n.id.clone();
    store.add_neuron(n.clone());
    created.push(n);
    id
}

/// Truncate on a char boundary (Python `len` is code points; never panic on UTF-8).
pub fn truncate_anchor(content: &str, max_chars: usize) -> String {
    let count = content.chars().count();
    if count <= max_chars {
        return content.to_string();
    }
    let mut out: String = content.chars().take(max_chars).collect();
    let cut = out.rfind('.').max(out.rfind('\n')).unwrap_or(0);
    if cut > max_chars / 2 {
        out.truncate(cut + 1);
        return out.trim().to_string();
    }
    out.trim().to_string()
}
