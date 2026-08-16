//! Bayesian belief update — cognitive layer (hypothesis / evidence).
//!
//! P' = P·L / (P·L + (1−P)·(1−L))

pub fn bayes(prior: f64, likelihood: f64) -> f64 {
    let p = prior.clamp(0.01, 0.99);
    let l = likelihood.clamp(0.01, 0.99);
    (p * l) / (p * l + (1.0 - p) * (1.0 - l))
}

pub fn evidence_for(prior: f64, strength: f64) -> f64 {
    bayes(prior, 0.5 + 0.4 * strength.clamp(0.0, 1.0))
}

pub fn evidence_against(prior: f64, strength: f64) -> f64 {
    bayes(prior, 0.5 - 0.4 * strength.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supporting_evidence_raises_belief() {
        let p = evidence_for(0.5, 0.8);
        assert!(p > 0.5, "{p}");
    }

    #[test]
    fn contrary_evidence_lowers_belief() {
        let p = evidence_against(0.7, 0.8);
        assert!(p < 0.7, "{p}");
    }
}
