//! Spreading activation — port of `engine/activation.py`.
//!
//! `activation(hop) = initial * decay^hop * synapse.weight * freq_factor * role_mult`

use crate::store::Store;
use crate::types::{ActivationResult, ActivationTrace, BrainConfig};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

const MAX_QUEUE: usize = 50_000;

#[derive(Clone)]
struct ActState {
    neuron_id: String,
    level: f64,
    hops: u32,
    path: Vec<String>,
    source: String,
}

impl PartialEq for ActState {
    fn eq(&self, other: &Self) -> bool {
        self.level == other.level
    }
}
impl Eq for ActState {}
impl PartialOrd for ActState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ActState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.level.partial_cmp(&other.level).unwrap_or(Ordering::Equal)
    }
}

pub fn should_stop_spreading(
    trace: &ActivationTrace,
    current_hop: u32,
    threshold: f64,
    min_new: u32,
    grace: u32,
) -> (bool, String) {
    if current_hop <= grace {
        return (false, String::new());
    }
    let prev = current_hop - 1;
    let prev_new = *trace.new_neurons_per_hop.get(&prev).unwrap_or(&0);
    if prev_new < min_new {
        return (
            true,
            format!("hop {prev} added only {prev_new} neurons (min={min_new})"),
        );
    }
    if current_hop >= 2 {
        let prev_prev = *trace.new_neurons_per_hop.get(&(prev - 1)).unwrap_or(&0);
        if prev_prev > 0 {
            let ratio = prev_new as f64 / prev_prev as f64;
            if ratio < threshold {
                return (
                    true,
                    format!("gain ratio {ratio:.2} < {threshold}"),
                );
            }
        }
    }
    (false, String::new())
}

pub struct SpreadOpts {
    pub max_hops: Option<u32>,
    pub decay_factor: f64,
    pub min_activation: Option<f64>,
    pub anchor_activations: HashMap<String, f64>,
    pub warm_activations: HashMap<String, f64>,
}

impl Default for SpreadOpts {
    fn default() -> Self {
        Self {
            max_hops: None,
            decay_factor: 0.5,
            min_activation: None,
            anchor_activations: HashMap::new(),
            warm_activations: HashMap::new(),
        }
    }
}

pub fn spread(store: &dyn Store, anchors: &[String], config: &BrainConfig, opts: SpreadOpts) -> (HashMap<String, ActivationResult>, ActivationTrace) {
    let max_hops = opts.max_hops.unwrap_or(config.max_spread_hops);
    let min_act = opts.min_activation.unwrap_or(config.activation_threshold);
    let decay = opts.decay_factor;

    let mut trace = ActivationTrace {
        max_hop_allowed: max_hops,
        ..Default::default()
    };
    let mut results: HashMap<String, ActivationResult> = HashMap::new();
    let mut heap: BinaryHeap<ActState> = BinaryHeap::new();

    for id in anchors {
        if store.get_neuron(id).is_none() {
            continue;
        }
        let mut level = *opts.anchor_activations.get(id).unwrap_or(&1.0);
        if let Some(w) = opts.warm_activations.get(id) {
            if *w > level {
                level = (*w).min(1.0);
            }
        }
        heap.push(ActState {
            neuron_id: id.clone(),
            level,
            hops: 0,
            path: vec![id.clone()],
            source: id.clone(),
        });
        results.insert(
            id.clone(),
            ActivationResult {
                neuron_id: id.clone(),
                activation_level: level,
                hop_distance: 0,
                path: vec![id.clone()],
                source_anchor: id.clone(),
            },
        );
        *trace.new_neurons_per_hop.entry(0).or_default() += 1;
        *trace.activation_gain_per_hop.entry(0).or_default() += level;
    }

    #[derive(Clone)]
    struct CachedEdge {
        neuron_id: String,
        weight: f64,
        role_mult: f64,
    }

    let mut visited: HashSet<(String, String)> = HashSet::new();
    let mut neighbor_cache: HashMap<String, Vec<CachedEdge>> = HashMap::new();
    let mut dr_checked: HashSet<u32> = HashSet::new();

    while let Some(current) = heap.pop() {
        if heap.len() > MAX_QUEUE {
            break;
        }
        let vkey = (current.neuron_id.clone(), current.source.clone());
        if !visited.insert(vkey) {
            continue;
        }
        if current.hops >= max_hops {
            continue;
        }

        let next_hop = current.hops + 1;
        if config.diminishing_returns_enabled && !dr_checked.contains(&next_hop) && next_hop >= 2 {
            dr_checked.insert(next_hop);
            let (stop, reason) = should_stop_spreading(
                &trace,
                next_hop,
                config.diminishing_returns_threshold,
                config.diminishing_returns_min_neurons,
                config.diminishing_returns_grace_hops,
            );
            if stop {
                trace.stopped_early = true;
                trace.stop_reason = reason;
                break;
            }
        }

        if !neighbor_cache.contains_key(&current.neuron_id) {
            let edges: Vec<CachedEdge> = store
                .neighbors(&current.neuron_id, 0.1)
                .into_iter()
                .filter_map(|(n, s)| {
                    let role_mult = s.type_.role_multiplier();
                    if role_mult == 0.0 {
                        return None;
                    }
                    Some(CachedEdge {
                        neuron_id: n.id.clone(),
                        weight: s.weight,
                        role_mult,
                    })
                })
                .collect();
            neighbor_cache.insert(current.neuron_id.clone(), edges);
        }
        let edges = neighbor_cache.get(&current.neuron_id).unwrap();

        for edge in edges {
            let freq = store
                .get_state(&edge.neuron_id)
                .map(|s| s.access_frequency)
                .unwrap_or(0);
            let freq_factor = 1.0 + (0.05 * (1.0 + freq as f64).ln()).min(0.15);
            let new_level = current.level * decay * edge.weight * freq_factor * edge.role_mult;
            if new_level < min_act {
                continue;
            }
            let hop = current.hops + 1;
            let mut path = current.path.clone();
            path.push(edge.neuron_id.clone());

            let better = match results.get(&edge.neuron_id) {
                None => true,
                Some(ex) => new_level > ex.activation_level,
            };
            if better {
                if !results.contains_key(&edge.neuron_id) {
                    *trace.new_neurons_per_hop.entry(hop).or_default() += 1;
                }
                *trace.activation_gain_per_hop.entry(hop).or_default() += new_level;
                results.insert(
                    edge.neuron_id.clone(),
                    ActivationResult {
                        neuron_id: edge.neuron_id.clone(),
                        activation_level: new_level,
                        hop_distance: hop,
                        path: path.clone(),
                        source_anchor: current.source.clone(),
                    },
                );
                if hop > trace.max_hop_used {
                    trace.max_hop_used = hop;
                }
            }
            heap.push(ActState {
                neuron_id: edge.neuron_id.clone(),
                level: new_level,
                hops: hop,
                path,
                source: current.source.clone(),
            });
        }
    }

    (results, trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryStore;
    use crate::types::{Neuron, NeuronType, Synapse, SynapseType, DEFAULT_CONFIG};

    #[test]
    fn spreads_along_caused_by() {
        let mut s = MemoryStore::new();
        let a = Neuron::create(NeuronType::Concept, "outage");
        let b = Neuron::create(NeuronType::Concept, "jwt expiry");
        let aid = a.id.clone();
        let bid = b.id.clone();
        s.add_neuron(a);
        s.add_neuron(b);
        s.add_synapse(Synapse::create(&aid, &bid, SynapseType::CausedBy, 0.9));

        let (res, _) = spread(&s, &[aid.clone()], &DEFAULT_CONFIG, SpreadOpts::default());
        assert!(res.contains_key(&aid));
        assert!(res.contains_key(&bid), "should spread to cause via CAUSED_BY");
        assert!(res[&bid].activation_level > 0.3);
        assert_eq!(res[&bid].hop_distance, 1);
    }

    #[test]
    fn skips_passive_synapses() {
        let mut s = MemoryStore::new();
        let a = Neuron::create(NeuronType::Concept, "event");
        let b = Neuron::create(NeuronType::Time, "tuesday");
        let aid = a.id.clone();
        let bid = b.id.clone();
        s.add_neuron(a);
        s.add_neuron(b);
        s.add_synapse(Synapse::create(&aid, &bid, SynapseType::HappenedAt, 0.9));
        let (res, _) = spread(&s, &[aid.clone()], &DEFAULT_CONFIG, SpreadOpts::default());
        assert!(!res.contains_key(&bid), "HAPPENED_AT is passive — skip");
    }
}
