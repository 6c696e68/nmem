//! Formal Hebbian rule from `engine/learning_rule.py`.
//!
//! Δw = η_eff * pre * post * (w_max - w)
//! η_eff = η * (1 + novelty_boost * e^(-novelty_decay * freq))

#[derive(Debug, Clone, Copy)]
pub struct LearningConfig {
    pub learning_rate: f64,
    pub weight_max: f64,
    pub novelty_boost_max: f64,
    pub novelty_decay_rate: f64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            weight_max: 1.0,
            novelty_boost_max: 3.0,
            novelty_decay_rate: 0.06,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeightUpdate {
    pub new_weight: f64,
    pub delta: f64,
    pub effective_rate: f64,
    pub saturated: bool,
}

pub fn hebbian_update(
    current_weight: f64,
    pre: f64,
    post: f64,
    reinforced_count: u32,
    cfg: LearningConfig,
) -> WeightUpdate {
    let w = current_weight.clamp(0.0, cfg.weight_max);
    if pre <= 0.0 || post <= 0.0 {
        return WeightUpdate {
            new_weight: w,
            delta: 0.0,
            effective_rate: 0.0,
            saturated: false,
        };
    }
    let novelty = 1.0 + cfg.novelty_boost_max * (-cfg.novelty_decay_rate * reinforced_count as f64).exp();
    let eta = cfg.learning_rate * novelty;
    let delta = eta * pre * post * (cfg.weight_max - w);
    let new_w = (w + delta).clamp(0.0, cfg.weight_max);
    WeightUpdate {
        new_weight: new_w,
        delta: new_w - w,
        effective_rate: eta,
        saturated: w > cfg.weight_max * 0.95,
    }
}

pub fn anti_hebbian_update(current_weight: f64, strength: f64, cfg: LearningConfig) -> WeightUpdate {
    let delta = -cfg.learning_rate * strength.clamp(0.0, 1.0) * current_weight;
    let new_weight = (current_weight + delta).max(0.0);
    WeightUpdate {
        new_weight,
        delta: new_weight - current_weight,
        effective_rate: cfg.learning_rate,
        saturated: false,
    }
}

/// Scale outgoing weights if they exceed `budget` (Python competitive norm).
pub fn scale_to_budget(weights: &mut [f64], budget: f64) {
    let sum: f64 = weights.iter().sum();
    if sum <= budget || sum <= 0.0 {
        return;
    }
    let k = budget / sum;
    for w in weights.iter_mut() {
        *w *= k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn novel_learns_faster_than_familiar() {
        let cfg = LearningConfig::default();
        let novel = hebbian_update(0.5, 1.0, 1.0, 0, cfg);
        let old = hebbian_update(0.5, 1.0, 1.0, 40, cfg);
        assert!(novel.delta > old.delta);
    }

    #[test]
    fn saturates_near_ceiling() {
        let cfg = LearningConfig::default();
        let near = hebbian_update(0.99, 1.0, 1.0, 0, cfg);
        let mid = hebbian_update(0.4, 1.0, 1.0, 0, cfg);
        assert!(near.delta < mid.delta);
        assert!(near.new_weight <= 1.0);
    }

    #[test]
    fn zero_pre_no_learn() {
        let u = hebbian_update(0.5, 0.0, 1.0, 0, LearningConfig::default());
        assert_eq!(u.delta, 0.0);
    }
}
