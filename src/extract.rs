//! Keyword / entity / type extraction — port of
//! `extraction/keywords.py`, `core/memory_types.suggest_memory_type`,
//! and the encode-time extractors.

use crate::types::{MemoryType, SynapseType};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const STOP_EN: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between", "under", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "and", "but", "if", "or", "because", "until", "while", "this",
    "that", "these", "those", "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "what", "which", "who", "whom", "also",
    "about", "out", "up", "new", "now", "always", "never", "like", "really", "actually",
    "dont", "cant", "wont", "im", "ive", "thats",
];

const STOP_VI: &[&str] = &[
    "và", "của", "là", "có", "được", "cho", "với", "này", "trong", "để", "các", "những", "một",
    "đã", "tôi", "bạn", "anh", "chị", "em", "ở", "tại", "khi", "thì", "mà", "nếu", "vì", "cũng",
    "như", "từ", "đến", "lại", "ra", "vào", "lên", "xuống", "rồi", "sẽ", "đang", "vẫn", "còn",
    "chỉ", "rất", "quá", "làm", "gì", "sao", "nào", "đâu", "ai", "bao", "nhiêu", "không", "hay",
    "hoặc", "nhưng", "bị", "nên", "cái", "đó", "kia", "mình",
];

fn stop_set() -> &'static HashSet<String> {
    static SET: OnceLock<HashSet<String>> = OnceLock::new();
    SET.get_or_init(|| {
        STOP_EN
            .iter()
            .chain(STOP_VI.iter())
            .map(|s| s.to_string())
            .collect()
    })
}

pub fn is_vietnamese(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(
            c,
            'ă' | 'â' | 'đ' | 'ê' | 'ô' | 'ơ' | 'ư'
                | 'ắ' | 'ằ' | 'ẳ' | 'ẵ' | 'ặ'
                | 'ấ' | 'ầ' | 'ẩ' | 'ẫ' | 'ậ'
                | 'ế' | 'ề' | 'ể' | 'ễ' | 'ệ'
                | 'ố' | 'ồ' | 'ổ' | 'ỗ' | 'ộ'
                | 'ớ' | 'ờ' | 'ở' | 'ỡ' | 'ợ'
                | 'ứ' | 'ừ' | 'ử' | 'ữ' | 'ự'
                | 'Ă' | 'Â' | 'Đ' | 'Ê' | 'Ô' | 'Ơ' | 'Ư'
        )
    })
}

pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = true;
    for c in text.chars() {
        if c.is_alphanumeric() || matches!(c, '.' | '_' | '/' | '-') {
            for lc in c.to_lowercase() {
                out.push(lc);
            }
            prev_space = false;
        } else if !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }
    out.trim().to_string()
}

pub fn tokenize(text: &str) -> Vec<String> {
    normalize(text)
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

pub fn keywords(text: &str, max: usize) -> Vec<String> {
    let stop = stop_set();
    let words: Vec<String> = tokenize(text)
        .into_iter()
        .filter(|w| w.chars().count() >= 2 && !stop.contains(w.as_str()))
        .collect();
    let mut counts: HashMap<String, u32> = HashMap::new();
    for w in &words {
        *counts.entry(w.clone()).or_default() += 1;
    }
    let mut ranked: Vec<(String, u32)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut out: Vec<String> = ranked.into_iter().map(|(w, _)| w).take(max).collect();
    for i in 0..words.len().saturating_sub(1) {
        if out.len() >= max {
            break;
        }
        if words[i].chars().count() >= 3 && words[i + 1].chars().count() >= 3 {
            let bg = format!("{} {}", words[i], words[i + 1]);
            if !out.iter().any(|x| x == &bg) {
                out.push(bg);
            }
        }
    }
    out
}

fn has_kw(hay: &str, needle: &str) -> bool {
    if needle.contains(' ') {
        return hay.contains(needle);
    }
    hay.split(|c: char| !c.is_alphanumeric())
        .any(|w| !w.is_empty() && w.to_lowercase() == needle)
}

struct TypeRule {
    ty: MemoryType,
    words: &'static [&'static str],
}

const TYPE_RULES: &[TypeRule] = &[
    TypeRule {
        ty: MemoryType::Todo,
        words: &[
            "todo", "fixme", "need to", "have to", "remember to", "should", "must", "cần", "phải",
            "nhớ", "việc cần",
        ],
    },
    TypeRule {
        ty: MemoryType::Decision,
        words: &[
            "decided",
            "chose",
            "picked",
            "selected",
            "opted for",
            "going with",
            "instead of",
            "switched to",
            "rejected",
            "went with",
            "quyết định",
            "chọn",
            "chuyển sang",
        ],
    },
    TypeRule {
        ty: MemoryType::Error,
        words: &[
            "error", "bug", "crash", "exception", "traceback", "failed", "broken", "outage",
            "incident", "lỗi", "sập", "treo", "hỏng", "sự cố",
        ],
    },
    TypeRule {
        ty: MemoryType::Hypothesis,
        words: &[
            "i think",
            "we think",
            "hypothesis",
            "we believe",
            "giả thuyết",
            "có lẽ",
            "có thể là",
        ],
    },
    TypeRule {
        ty: MemoryType::Prediction,
        words: &["predict", "will fail", "forecast", "dự đoán", "sẽ bị"],
    },
    TypeRule {
        ty: MemoryType::Insight,
        words: &[
            "learned",
            "realized",
            "discovered",
            "found that",
            "turns out",
            "root cause",
            "the trick",
            "key insight",
            "lesson learned",
            "noticed that",
            "figured out",
            "the pattern",
            "nhận ra",
            "học được",
            "hóa ra",
            "nguyên nhân",
        ],
    },
    TypeRule {
        ty: MemoryType::Boundary,
        words: &[
            "never send",
            "don't ever",
            "always ask before",
            "không được",
            "cấm",
        ],
    },
    TypeRule {
        ty: MemoryType::Instruction,
        words: &[
            "always use",
            "never use",
            "make sure",
            "don't forget",
            "always",
            "never",
            "luôn",
            "không bao giờ",
            "phải luôn",
        ],
    },
    TypeRule {
        ty: MemoryType::Preference,
        words: &[
            "prefer",
            "prefers",
            "preferred",
            "favorite",
            "hate",
            "dislike",
            "thích",
            "không thích",
            "ưa",
        ],
    },
    TypeRule {
        ty: MemoryType::Workflow,
        words: &[
            "workflow",
            "pipeline",
            "deploy",
            "ci/cd",
            "release process",
            "process",
            "flow",
            "quy trình",
        ],
    },
    TypeRule {
        ty: MemoryType::Reference,
        words: &["http://", "https://", "docs", "documentation", "tài liệu"],
    },
    TypeRule {
        ty: MemoryType::Tool,
        words: &["effective for", "use grep", "tooling"],
    },
    TypeRule {
        ty: MemoryType::Context,
        words: &["working on", "currently", "đang làm"],
    },
];

pub fn suggest_memory_type(content: &str) -> MemoryType {
    let hay = content.to_lowercase();
    for rule in TYPE_RULES {
        if rule.words.iter().any(|w| has_kw(&hay, w)) {
            if rule.ty == MemoryType::Todo
                && ["because", "root cause", "pattern", "architecture"]
                    .iter()
                    .any(|w| has_kw(&hay, w))
            {
                continue;
            }
            return rule.ty;
        }
    }
    MemoryType::Fact
}

pub fn extract_times(content: &str) -> Vec<String> {
    static PATS: OnceLock<Vec<Regex>> = OnceLock::new();
    let pats = PATS.get_or_init(|| {
        vec![
            Regex::new(r"(?i)\b(yesterday|today|tomorrow|tonight)\b").unwrap(),
            Regex::new(r"(?i)\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")
                .unwrap(),
            Regex::new(r"(?i)\b(hôm qua|hôm nay|ngày mai|tối nay)\b").unwrap(),
            Regex::new(r"(?i)\b(\d{1,2}:\d{2})\s*(am|pm|utc)?").unwrap(),
            Regex::new(r"(?i)\b(last week|next week|this week|tuần trước|tuần này|tuần sau)\b")
                .unwrap(),
            Regex::new(r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\b")
                .unwrap(),
            Regex::new(r"(?i)\b(tháng\s+\d{1,2})\b").unwrap(),
        ]
    });
    let mut out = Vec::new();
    for p in pats {
        if let Some(m) = p.find(content) {
            let s = m.as_str().trim().to_lowercase();
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }
    out
}

const ACTION_VERBS: &[&str] = &[
    "fixed",
    "fix",
    "deployed",
    "deploy",
    "decided",
    "chose",
    "added",
    "removed",
    "updated",
    "reviewed",
    "merged",
    "broke",
    "crashed",
    "rotated",
    "failed",
    "suggested",
    "implemented",
    "refactored",
    "migrated",
    "tested",
    "shipped",
    "sửa",
    "triển",
    "quyết",
    "thêm",
    "xóa",
];

pub fn extract_actions(content: &str) -> Vec<String> {
    tokenize(content)
        .into_iter()
        .filter(|w| ACTION_VERBS.contains(&w.as_str()))
        .collect()
}

pub fn extract_entities(content: &str) -> Vec<String> {
    static PROPER: OnceLock<Regex> = OnceLock::new();
    static CODE: OnceLock<Regex> = OnceLock::new();
    static ACRONYM: OnceLock<Regex> = OnceLock::new();
    static TICKET: OnceLock<Regex> = OnceLock::new();
    let proper = PROPER.get_or_init(|| {
        Regex::new(r"\b[A-Z][a-zA-Z0-9_+.-]{1,}(?:\s+[A-Z][a-zA-Z0-9_+.-]+)*\b").unwrap()
    });
    let code = CODE.get_or_init(|| {
        Regex::new(r"\b[a-zA-Z][\w.-]*\.(py|ts|tsx|js|go|rs|java|rb|md)\b").unwrap()
    });
    let acronym = ACRONYM.get_or_init(|| Regex::new(r"\b[A-Z]{2,6}\b").unwrap());
    let ticket = TICKET.get_or_init(|| Regex::new(r"#\d+\b").unwrap());

    let mut out: Vec<String> = Vec::new();
    let stop = stop_set();
    for m in proper.find_iter(content) {
        let s = m.as_str();
        if !stop.contains(&s.to_lowercase()) {
            push_unique(&mut out, s.to_string());
        }
    }
    for m in code.find_iter(content) {
        push_unique(&mut out, m.as_str().to_string());
    }
    for m in acronym.find_iter(content) {
        push_unique(&mut out, m.as_str().to_string());
    }
    for m in ticket.find_iter(content) {
        push_unique(&mut out, m.as_str().to_string());
    }
    out.truncate(10);
    out
}

const PLACES: &[&str] = &[
    "office",
    "home",
    "warehouse",
    "datacenter",
    "data center",
    "hanoi",
    "hà nội",
    "saigon",
    "sài gòn",
    "production",
    "staging",
];

pub fn extract_places(content: &str) -> Vec<String> {
    let hay = content.to_lowercase();
    let mut out = Vec::new();
    for p in PLACES {
        if hay.contains(p) {
            push_unique(&mut out, (*p).to_string());
        }
    }
    static AT: OnceLock<Regex> = OnceLock::new();
    let at = AT.get_or_init(|| {
        Regex::new(r"(?i)\b(?:at|tại|gần)\s+(?:the\s+)?([A-Za-zÀ-ỹ][\w-]{2,})\b").unwrap()
    });
    const PLACE_STOP: &[&str] = &[
        "the", "this", "that", "our", "my", "his", "her", "its", "least", "most",
        "first", "last", "same", "time", "moment", "point", "end", "start",
        "morning", "afternoon", "evening", "night", "fact", "case", "order",
        "addition", "general", "particular", "short", "long", "future", "past",
    ];
    for cap in at.captures_iter(content) {
        if let Some(m) = cap.get(1) {
            let s = m.as_str().to_lowercase();
            if s.chars().count() >= 3 && !PLACE_STOP.contains(&s.as_str()) {
                push_unique(&mut out, s);
            }
        }
    }
    out.truncate(6);
    out
}

const INTENT_PHRASES: &[&str] = &[
    "want to",
    "going to",
    "plan to",
    "aim to",
    "intend to",
    "muốn",
    "định",
];

pub fn extract_intents(content: &str) -> Vec<String> {
    let hay = content.to_lowercase();
    let mut out = Vec::new();
    for p in INTENT_PHRASES {
        if hay.contains(p) {
            push_unique(&mut out, (*p).to_string());
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct DetectedRelation {
    pub type_: SynapseType,
    pub hint: &'static str,
}

pub fn detect_relations(content: &str) -> Vec<DetectedRelation> {
    static CAUSE: OnceLock<Regex> = OnceLock::new();
    static LEADS: OnceLock<Regex> = OnceLock::new();
    static RESOLVE: OnceLock<Regex> = OnceLock::new();
    static CONTRA: OnceLock<Regex> = OnceLock::new();
    let hay = content.to_lowercase();
    let mut out = Vec::new();
    let cause = CAUSE.get_or_init(|| {
        Regex::new(r"(?i)(because|caused by|root cause|\bdo\b|bởi vì|nguyên nhân|khiến)").unwrap()
    });
    let leads = LEADS.get_or_init(|| {
        Regex::new(r"(?i)(leads to|\bthen\b|resulting|dẫn đến|kết quả)").unwrap()
    });
    let resolve = RESOLVE
        .get_or_init(|| Regex::new(r"(?i)(fixed|resolved|patched|sửa|khắc phục)").unwrap());
    let contra = CONTRA
        .get_or_init(|| Regex::new(r"(?i)(instead of|contradict|không còn|thay vì)").unwrap());
    static EVID: OnceLock<Regex> = OnceLock::new();
    let evid = EVID.get_or_init(|| {
        Regex::new(r"(?i)(confirms?|proves?|supports?|chứng minh|ủng hộ|bác bỏ|refutes?)").unwrap()
    });
    if cause.is_match(&hay) {
        out.push(DetectedRelation {
            type_: SynapseType::CausedBy,
            hint: "causal language",
        });
    }
    if leads.is_match(&hay) {
        out.push(DetectedRelation {
            type_: SynapseType::LeadsTo,
            hint: "sequence language",
        });
    }
    if resolve.is_match(&hay) {
        out.push(DetectedRelation {
            type_: SynapseType::ResolvedBy,
            hint: "resolution language",
        });
    }
    if contra.is_match(&hay) {
        out.push(DetectedRelation {
            type_: SynapseType::Contradicts,
            hint: "conflict language",
        });
    }
    if evid.is_match(&hay) {
        let against = hay.contains("bác bỏ") || hay.contains("refute");
        out.push(DetectedRelation {
            type_: if against {
                SynapseType::EvidenceAgainst
            } else {
                SynapseType::EvidenceFor
            },
            hint: "evidence language",
        });
    }
    out
}

#[derive(Debug, Clone)]
pub struct ExtractedRelation {
    pub source_span: String,
    pub target_span: String,
    pub type_: SynapseType,
    pub confidence: f64,
}

/// Clause-level extract: "X because Y" → X CAUSED_BY Y (source=effect, target=cause).
pub fn extract_relations(content: &str) -> Vec<ExtractedRelation> {
    static PATS: OnceLock<Vec<(Regex, SynapseType, f64)>> = OnceLock::new();
    let pats = PATS.get_or_init(|| {
        let mk = |p: &str, ty: SynapseType, c: f64| (Regex::new(p).unwrap(), ty, c);
        vec![
            mk(
                r"(?i)(.{3,80}?)\s+because\s+(.{3,80}?)(?:\.|;|$)",
                SynapseType::CausedBy,
                0.80,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:caused\s+by|due\s+to)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::CausedBy,
                0.85,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+as\s+a\s+result\s+of\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::CausedBy,
                0.80,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:therefore|thus|hence|consequently)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::LeadsTo,
                0.75,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:leads?\s+to|results?\s+in|causes?)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::LeadsTo,
                0.85,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:vì|do|bởi\s+vì)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::CausedBy,
                0.80,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:nên|cho\s+nên|vì\s+vậy|do\s+đó)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::LeadsTo,
                0.80,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:fixed|resolved|patched|sửa|khắc phục)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::ResolvedBy,
                0.75,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:instead of|thay vì)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::Contradicts,
                0.70,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:before|trước)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::Before,
                0.60,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:after|sau)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::After,
                0.60,
            ),
            mk(
                r"(?i)(.{3,80}?)\s+(?:supports?|proves?|confirms?|chứng minh|ủng hộ)\s+(.{3,80}?)(?:\.|;|$)",
                SynapseType::EvidenceFor,
                0.75,
            ),
            mk(
                r"(?i)(.{5,80}?)\s+(?:disproves?|refutes?|bác bỏ)\s+(.{5,80}?)(?:\.|;|$)",
                SynapseType::EvidenceAgainst,
                0.75,
            ),
        ]
    });
    let mut out = Vec::new();
    for (re, ty, conf) in pats {
        for cap in re.captures_iter(content) {
            let src = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let tgt = cap.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if src.chars().count() < 3 || tgt.chars().count() < 3 {
                continue;
            }
            out.push(ExtractedRelation {
                source_span: src.to_string(),
                target_span: tgt.to_string(),
                type_: *ty,
                confidence: *conf,
            });
        }
    }
    out
}

pub fn expand_query(query: &str) -> Vec<String> {
    let mut toks = keywords(query, 8);
    let hay = query.to_lowercase();
    if hay.contains("why")
        || hay.contains("tại sao")
        || hay.contains("tai sao")
        || hay.contains("nguyên nhân")
        || hay.contains("nguyen nhan")
    {
        for e in ["cause", "caused", "because", "root"] {
            if !toks.iter().any(|t| t == e) {
                toks.push(e.to_string());
            }
        }
    }
    crate::query::expand_terms(&toks, 4)
}

pub fn jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let aa: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let bb: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let inter = aa.intersection(&bb).count() as f64;
    let union = aa.union(&bb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn push_unique(out: &mut Vec<String>, s: String) {
    let low = s.to_lowercase();
    if !out.iter().any(|x| x.to_lowercase() == low) {
        out.push(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_decision() {
        assert_eq!(
            suggest_memory_type("We decided to use PostgreSQL"),
            MemoryType::Decision
        );
    }

    #[test]
    fn detects_error() {
        assert_eq!(
            suggest_memory_type("Fixed auth bug with null check"),
            MemoryType::Error
        );
    }

    #[test]
    fn detects_vietnamese_insight() {
        assert_eq!(
            suggest_memory_type("Nhận ra rằng cron timezone lệch 7 tiếng"),
            MemoryType::Insight
        );
    }

    #[test]
    fn extracts_jwt_entity() {
        let ents = extract_entities("JWT expiry caused the Tuesday outage");
        assert!(ents.iter().any(|e| e == "JWT"), "{ents:?}");
    }

    #[test]
    fn clause_because_is_caused_by() {
        let rels = extract_relations(
            "JWT expiry caused the Tuesday outage because rotation cron never ran",
        );
        assert!(
            rels.iter().any(|r| r.type_ == SynapseType::CausedBy
                && r.target_span.to_lowercase().contains("cron")),
            "{rels:?}"
        );
    }
}
