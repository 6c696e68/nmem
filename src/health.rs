//! Brain health — port of `brain_health` / `_compute_health_score`.

use crate::store::Store;
use crate::types::now_ms;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Freshness {
    pub total: u32,
    pub fresh: u32,
    pub recent: u32,
    pub aging: u32,
    pub stale: u32,
    pub ancient: u32,
    pub average_age_days: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub score: i32,
    pub grade: char,
    pub issues: Vec<String>,
    pub neurons: usize,
    pub synapses: usize,
    pub fibers: usize,
    pub orphans: usize,
    pub density: f64,
    pub freshness: Freshness,
    pub type_breakdown: HashMap<String, u32>,
}

pub fn health<S: Store>(store: &S) -> HealthReport {
    let now = now_ms();
    let fibers: Vec<_> = store.fibers().into_iter().cloned().collect();
    let mut f = Freshness {
        total: fibers.len() as u32,
        fresh: 0,
        recent: 0,
        aging: 0,
        stale: 0,
        ancient: 0,
        average_age_days: 0.0,
    };
    let mut ages = Vec::new();
    let mut types: HashMap<String, u32> = HashMap::new();
    for fib in &fibers {
        let days = (now.saturating_sub(fib.created_at) as f64) / 86_400_000.0;
        ages.push(days);
        match days {
            d if d < 7.0 => f.fresh += 1,
            d if d < 30.0 => f.recent += 1,
            d if d < 90.0 => f.aging += 1,
            d if d < 365.0 => f.stale += 1,
            _ => f.ancient += 1,
        }
        *types.entry(fib.memory_type.as_str().into()).or_default() += 1;
    }
    if !ages.is_empty() {
        f.average_age_days = ages.iter().sum::<f64>() / ages.len() as f64;
    }

    let n = store.neuron_count();
    let e = store.synapse_count();
    let density = if n > 1 {
        e as f64 / (n * (n - 1)) as f64
    } else {
        0.0
    };

    let fiber_neuron: std::collections::HashSet<String> = fibers
        .iter()
        .flat_map(|f| f.neuron_ids.iter().cloned())
        .collect();
    let orphans = store
        .neurons()
        .into_iter()
        .filter(|neu| !fiber_neuron.contains(&neu.id) && !neu.is_anchor())
        .count();

    let mut score: i32 = 100;
    let mut issues = Vec::new();

    let stale_ratio = if f.total == 0 {
        0.0
    } else {
        (f.stale + f.ancient) as f64 / f.total as f64
    };
    if stale_ratio > 0.5 {
        score -= 20;
        issues.push(format!("{:.0}% of memories are stale/ancient", stale_ratio * 100.0));
    } else if stale_ratio > 0.2 {
        score -= 10;
        issues.push(format!("{:.0}% of memories are stale/ancient", stale_ratio * 100.0));
    }

    if n > 0 && e == 0 {
        score -= 25;
        issues.push("no synapses — recall cannot spread".into());
    }
    if n > 20 && density < 0.005 {
        score -= 10;
        issues.push("graph is sparse — weak associations".into());
    }
    if orphans > n / 4 && n > 8 {
        score -= 8;
        issues.push(format!("{orphans} orphan neurons not in any fiber"));
    }
    if fibers.is_empty() {
        score -= 40;
        issues.push("brain is empty".into());
    }

    score = score.max(0);
    let grade = match score {
        90..=100 => 'A',
        80..=89 => 'B',
        65..=79 => 'C',
        50..=64 => 'D',
        _ => 'F',
    };

    HealthReport {
        score,
        grade,
        issues,
        neurons: n,
        synapses: e,
        fibers: fibers.len(),
        orphans,
        density,
        freshness: f,
        type_breakdown: types,
    }
}
