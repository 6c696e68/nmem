//! Query expansion — port of `engine/query_expander.py`.
//! Synonyms, abbreviations, EN↔VI. No embeddings.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

fn synonym_groups() -> &'static [&'static [&'static str]] {
    &[
        &["cost", "expense", "spending", "expenditure"],
        &["revenue", "income", "earnings", "sales"],
        &["error", "bug", "issue", "failure", "fault", "lỗi", "sự cố"],
        &["auth", "authentication", "authorization", "login"],
        &["deploy", "deployment", "release", "ship", "triển khai"],
        &["config", "configuration", "settings", "preferences", "cấu hình"],
        &["database", "db", "datastore", "cơ sở dữ liệu"],
        &["api", "endpoint", "route"],
        &["test", "testing", "spec", "unittest", "kiểm thử"],
        &["perf", "performance", "speed", "latency"],
        &["user", "account", "profile", "người dùng", "tài khoản"],
        &["log", "logging", "logger"],
        &["cache", "caching", "memoize"],
        &["retry", "retries", "backoff"],
        &["queue", "job", "task", "worker"],
        &["outage", "incident", "downtime", "sự cố"],
        &["jwt", "token"],
        &["cron", "schedule", "scheduler"],
    ]
}

fn abbreviations() -> &'static [(&'static str, &'static str)] {
    &[
        ("api", "application programming interface"),
        ("db", "database"),
        ("jwt", "json web token"),
        ("pr", "pull request"),
        ("ci", "continuous integration"),
        ("sql", "structured query language"),
        ("cli", "command line interface"),
    ]
}

fn cross_lang() -> &'static [(&'static str, &'static str)] {
    &[
        ("error", "lỗi"),
        ("deploy", "triển khai"),
        ("decision", "quyết định"),
        ("workflow", "quy trình"),
        ("user", "người dùng"),
        ("config", "cấu hình"),
        ("test", "kiểm thử"),
        ("database", "cơ sở dữ liệu"),
        ("account", "tài khoản"),
        ("memory", "bộ nhớ"),
        ("cause", "nguyên nhân"),
        ("outage", "sự cố"),
    ]
}

fn synonym_map() -> &'static HashMap<String, Vec<String>> {
    static M: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m: HashMap<String, Vec<String>> = HashMap::new();
        for g in synonym_groups() {
            for w in *g {
                let others: Vec<String> = g
                    .iter()
                    .filter(|x| *x != w)
                    .map(|x| (*x).to_string())
                    .collect();
                m.insert((*w).to_string(), others);
            }
        }
        m
    })
}

/// Expand keywords: original + synonyms + abbreviations + EN↔VI.
pub fn expand_terms(keywords: &[String], max_per: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let syn = synonym_map();
    for kw in keywords {
        let low = kw.to_lowercase();
        if low.is_empty() || !seen.insert(low.clone()) {
            continue;
        }
        out.push(low.clone());
        let mut added = 0usize;
        if let Some(others) = syn.get(&low) {
            for o in others {
                if added >= max_per {
                    break;
                }
                if seen.insert(o.clone()) {
                    out.push(o.clone());
                    added += 1;
                }
            }
        }
        for (abbr, full) in abbreviations() {
            if added >= max_per {
                break;
            }
            if low == *abbr {
                for part in full.split_whitespace() {
                    if part.len() < 4 {
                        continue;
                    }
                    if seen.insert(part.to_string()) {
                        out.push(part.to_string());
                        added += 1;
                    }
                }
            }
            // Do NOT reverse-map generic words ("web", "json", "line") back to the abbr.
        }
        for (en, vi) in cross_lang() {
            if added >= max_per {
                break;
            }
            if low == *en && seen.insert((*vi).to_string()) {
                out.push((*vi).to_string());
                added += 1;
            } else if low == *vi && seen.insert((*en).to_string()) {
                out.push((*en).to_string());
                added += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_error_to_bug_and_loi() {
        let e = expand_terms(&["error".into()], 5);
        assert!(e.iter().any(|x| x == "bug"), "{e:?}");
        assert!(e.iter().any(|x| x == "lỗi"), "{e:?}");
    }

    #[test]
    fn expands_jwt_abbr() {
        let e = expand_terms(&["jwt".into()], 6);
        assert!(e.iter().any(|x| x == "token" || x == "json"), "{e:?}");
        let web = expand_terms(&["web".into()], 6);
        assert!(!web.iter().any(|x| x == "jwt"), "generic 'web' must not map to jwt: {web:?}");
    }
}
