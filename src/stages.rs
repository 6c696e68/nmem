//! Memory maturation — STM → working → episodic → semantic.
//! Port of `engine/memory_stages.py` + HOT/WARM/COLD tiers.

use crate::types::{Fiber, MemoryStage, MemoryTier, now_ms};

pub fn stage_decay_mult(stage: MemoryStage) -> f64 {
    match stage {
        MemoryStage::ShortTerm => 5.0,
        MemoryStage::Working => 2.0,
        MemoryStage::Episodic => 1.0,
        MemoryStage::Semantic => 0.3,
    }
}

pub fn tier_decay_mult(tier: MemoryTier) -> f64 {
    match tier {
        MemoryTier::Hot => 0.5,
        MemoryTier::Warm => 1.0,
        MemoryTier::Cold => 2.0,
    }
}

pub fn tier_floor(tier: MemoryTier) -> f64 {
    match tier {
        MemoryTier::Hot => 0.5,
        MemoryTier::Warm | MemoryTier::Cold => 0.0,
    }
}

/// Promote a fiber based on age + spacing (frequency across time).
pub fn maybe_promote(fiber: &mut Fiber, now: u64) -> bool {
    let age_h = (now.saturating_sub(fiber.created_at) as f64) / 3_600_000.0;
    let next = match fiber.stage {
        MemoryStage::ShortTerm if age_h >= 0.5 => Some(MemoryStage::Working),
        MemoryStage::Working if age_h >= 4.0 => Some(MemoryStage::Episodic),
        MemoryStage::Episodic if age_h >= 72.0 && fiber.frequency >= 2 => {
            Some(MemoryStage::Semantic)
        }
        _ => None,
    };
    if let Some(s) = next {
        fiber.stage = s;
        true
    } else {
        false
    }
}

/// Decay unused conductivity. HOT never drops below 0.5.
pub fn decay_conductivity(fiber: &mut Fiber, now: u64) {
    let last = fiber.last_conducted.unwrap_or(fiber.created_at);
    let days = (now.saturating_sub(last) as f64) / 86_400_000.0;
    if days <= 0.0 {
        return;
    }
    let k = stage_decay_mult(fiber.stage) * tier_decay_mult(fiber.tier);
    let factor = (-0.08 * days * k).exp();
    let floor = tier_floor(fiber.tier).max(0.12);
    fiber.conductivity = (fiber.conductivity * factor).max(floor);
}

pub fn age_hours(fiber: &Fiber) -> f64 {
    (now_ms().saturating_sub(fiber.created_at) as f64) / 3_600_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MemoryType;

    #[test]
    fn stm_promotes_after_30min() {
        let mut f = Fiber::create(vec!["a".into()], vec![], "a", "x", MemoryType::Fact, vec![]);
        f.created_at = now_ms() - 2 * 3_600_000;
        assert!(maybe_promote(&mut f, now_ms()));
        assert_eq!(f.stage, MemoryStage::Working);
    }

    #[test]
    fn hot_never_below_floor() {
        let mut f = Fiber::create(
            vec!["a".into()],
            vec![],
            "a",
            "prefer tabs",
            MemoryType::Preference,
            vec![],
        );
        f.tier = MemoryTier::Hot;
        f.conductivity = 0.55;
        f.last_conducted = Some(now_ms() - 40 * 86_400_000);
        decay_conductivity(&mut f, now_ms());
        assert!(f.conductivity >= 0.5, "{}", f.conductivity);
    }
}
