//! IDF-weighted anchor slots — port of `engine/idf_anchor.py`.

/// `log((N+1)/(1+df)) / log(N+1)` → [0, 1].
pub fn idf(df: u32, n_docs: u32) -> f64 {
    if n_docs == 0 {
        return 1.0;
    }
    let denom = (n_docs as f64 + 1.0).ln();
    if denom == 0.0 {
        return 1.0;
    }
    ((n_docs as f64 + 1.0) / (1.0 + df as f64)).ln() / denom
}

/// Rare terms get up to 5 slots; common terms get 1.
pub fn slots_for_idf(score: f64) -> usize {
    (1.0 + score.clamp(0.0, 1.0) * 4.0).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rare_higher_than_common() {
        assert!(idf(1, 20) > idf(15, 20));
    }

    #[test]
    fn rare_gets_more_slots() {
        assert!(slots_for_idf(0.95) > slots_for_idf(0.1));
        assert_eq!(slots_for_idf(0.0), 1);
        assert_eq!(slots_for_idf(1.0), 5);
    }
}
