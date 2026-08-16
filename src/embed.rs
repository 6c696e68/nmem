//! Cheap hashed n-gram embedding. No model, no GPU, no download.
//! Complements spreading activation — never replaces the graph.

pub const DIM: usize = 128;

/// Feature-hashed token + char-trigram vector, L2-normalized.
pub fn embed(text: &str) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    let norm = crate::extract::normalize(text);
    if norm.is_empty() {
        return v;
    }
    for tok in norm.split_whitespace() {
        if tok.len() < 2 {
            continue;
        }
        accum(&mut v, tok.as_bytes());
        let b = tok.as_bytes();
        if b.len() >= 3 {
            for w in b.windows(3) {
                accum(&mut v, w);
            }
        } else {
            accum(&mut v, b);
        }
    }
    l2_normalize(&mut v);
    v
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
    }
    dot.clamp(-1.0, 1.0)
}

fn accum(v: &mut [f32], bytes: &[u8]) {
    let h = fnv1a(bytes);
    let i = (h as usize) % v.len();
    let sign = if (h >> 32) & 1 == 0 { 1.0 } else { -1.0 };
    v[i] += sign;
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn l2_normalize(v: &mut [f32]) {
    let mut s = 0.0f32;
    for x in v.iter() {
        s += *x * *x;
    }
    if s <= 1e-12 {
        return;
    }
    let inv = s.sqrt().recip();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similar_text_high_cosine() {
        let a = embed("JWT expiry caused the Tuesday outage");
        let b = embed("the tuesday outage was caused by jwt expiry");
        let c = embed("alice prefers dark mode in the editor");
        assert!(cosine(&a, &b) > cosine(&a, &c) + 0.08);
        assert!(cosine(&a, &b) > 0.25);
    }

    #[test]
    fn empty_is_zero() {
        let z = embed("");
        assert!(z.iter().all(|x| *x == 0.0));
        assert_eq!(cosine(&z, &embed("x")), 0.0);
    }
}
