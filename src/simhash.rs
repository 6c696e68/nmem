//! Token simhash — port of `utils/simhash.py` (64-bit).

pub fn hash64(s: &str) -> u64 {
    // FNV-1a 64
    let mut h = 0xcbf29ce484222325u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn simhash(tokens: &[String]) -> u64 {
    if tokens.is_empty() {
        return 0;
    }
    let mut acc = [0i32; 64];
    for t in tokens {
        let h = hash64(&t.to_lowercase());
        for i in 0..64 {
            if (h >> i) & 1 == 1 {
                acc[i] += 1;
            } else {
                acc[i] -= 1;
            }
        }
    }
    let mut out = 0u64;
    for i in 0..64 {
        if acc[i] > 0 {
            out |= 1u64 << i;
        }
    }
    out
}

pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

pub fn is_near_duplicate(a: &[String], b: &[String], max_distance: u32) -> bool {
    hamming(simhash(a), simhash(b)) <= max_distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_dup_close_sentences() {
        let a = ["jwt", "expiry", "caused", "outage", "cron"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let mut b = a.clone();
        b.push("again".into());
        assert!(is_near_duplicate(&a, &b, 16), "d={}", hamming(simhash(&a), simhash(&b)));
    }

    #[test]
    fn far_apart() {
        let a = vec!["redis".into(), "session".into()];
        let b = vec!["tuesday".into(), "outage".into()];
        assert!(!is_near_duplicate(&a, &b, 3));
    }
}
