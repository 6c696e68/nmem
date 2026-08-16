//! Recall: lexical anchors → spreading activation → fiber scoring.
//! Scoring matches `engine/retrieval.py` `_fiber_score` (salience × recency ×
//! conductivity × activation coverage).

use crate::activation::{spread, SpreadOpts};
use crate::extract::{expand_query, tokenize};
use crate::store::Store;
use crate::types::{
    ActivationResult, ActivationTrace, Fiber, MemoryType, SynapseType, now_ms,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalledMemory {
    pub fiber: Fiber,
    pub score: f64,
    pub activation: f64,
    pub hop: u32,
    pub path: Vec<String>,
    pub reason: String,
    pub confidence: f64,
    #[serde(default)]
    pub embed: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub query: String,
    pub memories: Vec<RecalledMemory>,
    pub activations: HashMap<String, ActivationResult>,
    pub anchors: Vec<String>,
    pub trace: ActivationTrace,
    pub elapsed_ms: u128,
}

pub struct RecallOpts {
    pub depth: Option<u32>,
    pub limit: usize,
    pub memory_type: Option<MemoryType>,
    pub warm: HashMap<String, f64>,
}

impl Default for RecallOpts {
    fn default() -> Self {
        Self {
            depth: None,
            limit: 8,
            memory_type: None,
            warm: HashMap::new(),
        }
    }
}

pub fn recall<S: Store>(store: &mut S, query: &str, opts: RecallOpts) -> RecallResult {
    let t0 = std::time::Instant::now();
    let tokens = tokenize(query);
    let kws = expand_query(query);
    let search: Vec<String> = if kws.is_empty() { tokens.clone() } else { kws };

    // IDF-weighted per-term slots (Python idf_anchor)
    let mut matches = Vec::new();
    let mut seen_n = std::collections::HashSet::new();
    let mut idf_rank: Vec<(f64, crate::types::Neuron)> = Vec::new();
    for kw in &search {
        let score = store.token_idf(kw);
        let slots = crate::idf::slots_for_idf(score);
        for n in store.find_content_match(std::slice::from_ref(kw), slots) {
            if seen_n.insert(n.id.clone()) {
                idf_rank.push((score, n));
            }
        }
    }
    idf_rank.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, n) in idf_rank.into_iter().take(24) {
        matches.push(n);
    }
    if matches.is_empty() {
        matches = store.find_content_match(&search, 24);
    }
    let mut anchors: Vec<String> = Vec::new();
    let mut anchor_act: HashMap<String, f64> = HashMap::new();
    for (i, n) in matches.iter().enumerate() {
        anchors.push(n.id.clone());
        let recency = recency_sigmoid(n.created_at, store.meta().config.recency_halflife_hours);
        let rank = 1.0 / (1.0 + i as f64 * 0.08);
        let boost = if n.is_anchor() { 1.0 } else { 0.75 };
        anchor_act.insert(n.id.clone(), (rank * boost + recency * 0.15).min(1.0));
    }

    let mut spread_opts = SpreadOpts {
        max_hops: opts.depth,
        decay_factor: 0.5,
        min_activation: None,
        anchor_activations: anchor_act,
        warm_activations: opts.warm,
    };
    if spread_opts.max_hops.is_none() {
        spread_opts.max_hops = Some(store.meta().config.max_spread_hops);
    }

    let (activations, trace) = if anchors.is_empty() {
        (HashMap::new(), ActivationTrace::default())
    } else {
        let cfg = store.meta().config.clone();
        spread(store, &anchors, &cfg, spread_opts)
    };

    let now = now_ms();
    let window = crate::temporal::parse_query_window(query, now);
    let delta = store.meta().config.reinforcement_delta;
    let steep = store.meta().config.sigmoid_steepness;
    let refractory_ms = store.meta().config.refractory_ms;
    for act in activations.values() {
        if let Some(st) = store.get_state_mut(&act.neuron_id) {
            st.fire(act.activation_level, now, steep, refractory_ms);
        }
    }

    let qtokens = &tokens;
    let cfg = &store.meta().config;
    let hl = cfg.recency_halflife_hours;
    let tag_boost_cfg = cfg.tag_match_boost;
    let high_signal_boost = cfg.high_signal_boost;
    let rrf_k = cfg.rrf_k;

    struct Cand {
        fid: String,
        score: f64,
        activation: f64,
        hop: u32,
        path: Vec<String>,
        reason: String,
        confidence: f64,
    }
    let mut cands: Vec<Cand> = Vec::new();

    for fiber in store.fibers() {
        if let Some(filter) = opts.memory_type {
            if fiber.memory_type != filter { continue; }
        }
        if !fiber.is_live(now) { continue; }
        let mut max_act = 0.0_f64;
        let mut mean_act = 0.0_f64;
        let mut act_count = 0usize;
        let mut best_act_ref: Option<&ActivationResult> = None;
        for nid in &fiber.neuron_ids {
            if let Some(a) = activations.get(nid) {
                act_count += 1;
                mean_act += a.activation_level;
                if a.activation_level > max_act {
                    max_act = a.activation_level;
                    best_act_ref = Some(a);
                }
            }
        }
        if act_count == 0 { continue; }
        mean_act /= act_count as f64;
        let coverage = act_count as f64 / fiber.neuron_ids.len().max(1) as f64;
        let activation_signal = if activations
            .get(&fiber.anchor_neuron_id)
            .is_some_and(|a| a.hop_distance == 0)
        {
            (max_act * 0.7 + coverage * 0.15 + mean_act * 0.15).max(0.8)
        } else {
            (max_act * 0.5 + coverage * 0.3 + mean_act * 0.2).max(0.05)
        };

        let (best_act, best_hop, best_path) =
            if let Some(a) = activations.get(&fiber.anchor_neuron_id) {
                (a.activation_level, a.hop_distance, a.path.clone())
            } else if let Some(a) = best_act_ref {
                (a.activation_level, a.hop_distance, a.path.clone())
            } else {
                continue;
            };

        let recency = recency_sigmoid(fiber.last_conducted.unwrap_or(fiber.created_at), hl);
        let tag_boost = tag_overlap(&fiber.tags, qtokens) * tag_boost_cfg;
        let base = fiber.salience.max(0.05) * recency * fiber.conductivity.max(0.05);
        let mut score = base * activation_signal;
        if fiber.memory_type.is_high_signal() { score *= high_signal_boost; }
        score += tag_boost;
        let lex = lexical_hits(&fiber.summary, qtokens);
        score *= 1.0 + 0.55 * lex;
        let age_h = (now.saturating_sub(fiber.created_at) as f64) / 3_600_000.0;
        score *= 1.0 + 0.15 * (-age_h / 24.0).exp();
        if (fiber.belief - 0.5).abs() > f64::EPSILON {
            score *= 0.85 + 0.3 * fiber.belief;
        }
        if fiber.stage == crate::types::MemoryStage::Semantic { score *= 1.1; }
        score *= 0.85 + 0.2 * fiber.trust;
        if let Some(w) = &window {
            if fiber_in_window(store, fiber, w) { score *= 1.4; } else { score *= 0.88; }
        }

        let reason = if best_hop == 0 {
            "direct lexical match".into()
        } else {
            format!("spreading activation hop {best_hop}")
        };
        let confidence = compute_confidence(best_act, best_hop, fiber.salience, recency);
        cands.push(Cand {
            fid: fiber.id.clone(),
            score, activation: best_act, hop: best_hop, path: best_path, reason, confidence,
        });
    }

    cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let keep = opts.limit.saturating_mul(4).max(16).min(cands.len());
    cands.truncate(keep);

    let mut scored: Vec<RecalledMemory> = Vec::with_capacity(cands.len());
    for c in cands {
        let Some(fiber) = store.get_fiber(&c.fid).cloned() else { continue };
        scored.push(RecalledMemory {
            fiber, score: c.score, activation: c.activation, hop: c.hop,
            path: c.path, reason: c.reason, confidence: c.confidence, embed: 0.0,
        });
    }

    apply_causal_semantics(store, &mut scored);
    apply_rrf(&mut scored, qtokens, rrf_k);
    apply_embed_fusion(&mut scored, query);
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(opts.limit);

    let mut touched: Vec<String> = Vec::new();
    for m in &scored {
        if let Some(f) = store.get_fiber_mut(&m.fiber.id) {
            f.conduct(now);
        }
        for w in m.path.windows(2) {
            reinforce_edge(store, &w[0], &w[1], delta, now);
            touched.push(w[0].clone());
        }
    }
    normalize_outgoing(store, 5.0, &touched);

    RecallResult {
        query: query.to_string(),
        memories: scored,
        activations,
        anchors,
        trace,
        elapsed_ms: t0.elapsed().as_millis(),
    }
}

fn recency_sigmoid(ts: u64, halflife_hours: f64) -> f64 {
    let hours = (now_ms().saturating_sub(ts) as f64) / 3_600_000.0;
    let hl = halflife_hours.max(1.0);
    (1.0 / (1.0 + ((hours - hl) / (hl / 2.0)).exp())).max(0.1)
}

fn lexical_hits(summary: &str, tokens: &[String]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let hay = summary.to_lowercase();
    let hits = tokens
        .iter()
        .filter(|t| t.chars().count() >= 3 && hay.contains(t.as_str()))
        .count();
    hits as f64 / tokens.len() as f64
}

fn fiber_in_window<S: Store>(store: &S, fiber: &Fiber, w: &crate::temporal::TimeWindow) -> bool {
    if crate::temporal::overlap(fiber.created_at, fiber.created_at, w.start, w.end) {
        return true;
    }
    for nid in &fiber.neuron_ids {
        let Some(n) = store.get_neuron(nid) else {
            continue;
        };
        if n.type_ != crate::types::NeuronType::Time {
            continue;
        }
        let s = n
            .metadata
            .get("t_start")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let e = n
            .metadata
            .get("t_end")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if s > 0 && crate::temporal::overlap(s, e, w.start, w.end) {
            return true;
        }
        if n.content.to_lowercase().contains(&w.label) {
            return true;
        }
    }
    fiber.summary.to_lowercase().contains(&w.label)
}

fn apply_embed_fusion(scored: &mut [RecalledMemory], query: &str) {
    if scored.is_empty() {
        return;
    }
    let qv = crate::embed::embed(query);
    for m in scored.iter_mut() {
        let ev = crate::embed::embed(&m.fiber.summary);
        let c = crate::embed::cosine(&qv, &ev);
        m.embed = c;
        if c > 0.0 {
            m.score *= 1.0 + 0.28 * c;
        }
        if c >= 0.55 && m.hop > 0 {
            m.reason = format!("{}, embed {c:.2}", m.reason);
        }
    }
}

fn apply_rrf(scored: &mut [RecalledMemory], qtokens: &[String], k: f64) {
    let n = scored.len();
    if n < 2 {
        return;
    }
    let k = k.max(1.0);
    let mut g_order: Vec<usize> = (0..n).collect();
    g_order.sort_by(|&a, &b| {
        scored[b]
            .score
            .partial_cmp(&scored[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut g_rank = vec![0usize; n];
    for (r, i) in g_order.iter().enumerate() {
        g_rank[*i] = r;
    }
    let mut lex: Vec<(usize, f64)> = scored
        .iter()
        .enumerate()
        .map(|(i, m)| (i, lexical_hits(&m.fiber.summary, qtokens)))
        .collect();
    lex.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut l_rank = vec![0usize; n];
    for (r, (i, _)) in lex.iter().enumerate() {
        l_rank[*i] = r;
    }
    for i in 0..n {
        scored[i].score += 1.0 / (k + g_rank[i] as f64) + 1.0 / (k + l_rank[i] as f64);
    }
}

fn tag_overlap(tags: &[String], tokens: &[String]) -> f64 {
    if tags.is_empty() || tokens.is_empty() {
        return 0.0;
    }
    let mut hits = 0;
    for t in tags {
        if tokens
            .iter()
            .any(|q| t.contains(q.as_str()) || q.contains(t.as_str()))
        {
            hits += 1;
        }
    }
    hits as f64 / tags.len() as f64
}

fn apply_causal_semantics<S: Store>(store: &S, scored: &mut [RecalledMemory]) {
    if scored.is_empty() { return; }
    let mut by_anchor: HashMap<String, usize> = HashMap::with_capacity(scored.len());
    for (i, m) in scored.iter().enumerate() {
        by_anchor.insert(m.fiber.anchor_neuron_id.clone(), i);
    }
    let mut boosts: HashMap<usize, f64> = HashMap::new();
    let mut seen_syn_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let anchor_ids: Vec<String> = by_anchor.keys().cloned().collect();
    for aid in &anchor_ids {
        for (other, syn) in store.neighbors(aid, 0.0) {
            if !by_anchor.contains_key(&other.id) { continue; }
            if !seen_syn_ids.insert(syn.id.clone()) { continue; }
            let Some(&si) = by_anchor.get(&syn.source_id) else { continue };
            let Some(&ti) = by_anchor.get(&syn.target_id) else { continue };
            match syn.type_ {
                SynapseType::ResolvedBy | SynapseType::FalsifiedBy => {
                    *boosts.entry(si).or_default() -= 0.25 * syn.weight;
                    *boosts.entry(ti).or_default() += 0.20 * syn.weight;
                }
                SynapseType::Supersedes | SynapseType::EvolvesFrom => {
                    *boosts.entry(si).or_default() += 0.20 * syn.weight;
                    *boosts.entry(ti).or_default() -= 0.25 * syn.weight;
                }
                SynapseType::EvidenceFor | SynapseType::VerifiedBy => {
                    *boosts.entry(ti).or_default() += 0.15 * syn.weight;
                }
                SynapseType::EvidenceAgainst | SynapseType::Contradicts => {
                    *boosts.entry(ti).or_default() -= 0.15 * syn.weight;
                }
                _ => {}
            }
        }
    }
    for (i, d) in boosts {
        if let Some(m) = scored.get_mut(i) {
            m.score = (m.score + d).max(0.0);
        }
    }
}

fn reinforce_edge<S: Store>(store: &mut S, a: &str, b: &str, _delta: f64, now: u64) {
    let hit = store
        .neighbors(a, 0.0)
        .into_iter()
        .find(|(n, _)| n.id == b)
        .map(|(_, s)| s.id.clone());
    let Some(id) = hit else { return };
    let pre = store.get_state(a).map(|s| s.activation_level).unwrap_or(0.6).max(0.3);
    let post = store.get_state(b).map(|s| s.activation_level).unwrap_or(0.6).max(0.3);
    let (count, w) = match store.get_synapse_mut(&id) {
        Some(s) => (s.reinforced_count, s.weight),
        None => return,
    };
    let upd = crate::hebbian::hebbian_update(w, pre, post, count, crate::hebbian::LearningConfig::default());
    if let Some(s) = store.get_synapse_mut(&id) {
        s.weight = upd.new_weight;
        if upd.delta > 0.0 {
            s.reinforced_count += 1;
            s.last_activated = Some(now);
        }
    }
}

fn normalize_outgoing<S: Store>(store: &mut S, budget: f64, sources: &[String]) {
    let mut seen = std::collections::HashSet::new();
    for src in sources {
        if !seen.insert(src.as_str()) { continue; }
        let ids: Vec<String> = store
            .neighbors_out(src, 0.0)
            .into_iter()
            .map(|(_, s)| s.id.clone())
            .collect();
        if ids.len() < 2 { continue; }
        let mut weights: Vec<f64> = Vec::with_capacity(ids.len());
        for id in &ids {
            match store.get_synapse_mut(id) {
                Some(s) => weights.push(s.weight),
                None => break,
            }
        }
        if weights.len() != ids.len() { continue; }
        crate::hebbian::scale_to_budget(&mut weights, budget);
        for (id, w) in ids.iter().zip(weights.iter()) {
            if let Some(s) = store.get_synapse_mut(id) {
                s.weight = *w;
            }
        }
    }
}

fn compute_confidence(act: f64, hop: u32, salience: f64, freshness: f64) -> f64 {
    let retrieval = act.clamp(0.0, 1.0);
    let hop_pen = 1.0 / (1.0 + hop as f64 * 0.2);
    (0.40 * retrieval
        + 0.25 * hop_pen
        + 0.20 * salience.clamp(0.0, 1.0)
        + 0.15 * freshness.clamp(0.0, 1.0))
    .clamp(0.0, 1.0)
}
