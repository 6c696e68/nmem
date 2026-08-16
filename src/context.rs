//! Token-budget context pack — lean stand-in for Python `token_budget` /
//! `context_compiler`. For injecting memories into an LLM prompt on a weak box.

use crate::retrieval::{recall, RecallOpts};
use crate::store::Store;

#[derive(Debug, Clone)]
pub struct ContextPack {
    pub query: String,
    pub text: String,
    pub tokens: usize,
    pub memories: usize,
}

fn estimate_tokens(s: &str) -> usize {
    let words = s.split_whitespace().count();
    let by_chars = s.chars().count().div_ceil(4);
    words.max(by_chars)
}

pub fn pack<S: Store>(store: &mut S, query: &str, budget: usize) -> ContextPack {
    let budget = budget.max(32);
    let r = recall(
        store,
        query,
        RecallOpts {
            limit: 16,
            ..Default::default()
        },
    );
    let mut text = String::new();
    let mut used = 0usize;
    let mut n = 0usize;
    for m in &r.memories {
        let block = format!(
            "- [{}|{}|{:.2}] {}\n",
            m.fiber.memory_type.as_str(),
            m.fiber.stage.as_str(),
            m.confidence,
            m.fiber.summary
        );
        let t = estimate_tokens(&block);
        if used + t > budget && n > 0 {
            break;
        }
        text.push_str(&block);
        used += t;
        n += 1;
    }
    ContextPack {
        query: query.to_string(),
        text,
        tokens: used,
        memories: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_nonzero() {
        assert!(estimate_tokens("hello world from the brain") >= 4);
    }
}
