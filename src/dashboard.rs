//! Tiny HTTP dashboard. One binary, no Node, no GPU.

use crate::types::{now_ms, MemoryType, SynapseType};
use crate::Brain;
use crate::RecallOpts;
use crate::Store;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const PAGE: &str = include_str!("dashboard.html");

const DEMO: &[&str] = &[
    "Tuesday production outage at 15:00 UTC — API 502 for 18 minutes",
    "JWT expiry caused the Tuesday outage because rotation cron never ran after the deploy",
    "We decided to use Redis for the session store instead of JWT-only auth",
    "Fixed auth bug with null check in login.py:42 — empty token now returns 401",
    "Always rotate JWT signing keys on a 12-hour cron and page if the job fails",
    "Nhận ra rằng cron timezone UTC/ICT lệch 7 tiếng là nguyên nhân rotation miss",
];

pub fn run(brain: Brain, host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let mut brain = brain;
    if brain.store().fiber_count() == 0 {
        for line in DEMO {
            let _ = brain.remember(line);
        }
        let _ = brain.save();
    }
    let path = format!("{host}:{port}");
    let listener = TcpListener::bind(&path)?;
    eprintln!("nmem dashboard on http://{path}");
    let state = Arc::new(Mutex::new(brain));
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let st = Arc::clone(&state);
                if let Err(e) = handle(s, &st) {
                    eprintln!("dash: {e}");
                }
            }
            Err(e) => eprintln!("dash accept: {e}"),
        }
    }
    Ok(())
}

pub fn serve_path(
    brain_path: PathBuf,
    host: &str,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    run(Brain::open(brain_path)?, host, port)
}

fn handle(mut stream: TcpStream, state: &Arc<Mutex<Brain>>) -> Result<(), Box<dyn std::error::Error>> {
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
    let mut buf = vec![0u8; 16 * 1024];
    let n = stream.read(&mut buf)?;
    if n == 0 {
        return Ok(());
    }
    let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
    let mut lines = raw.split("\r\n");
    let req = lines.next().unwrap_or("");
    let mut parts = req.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let target = parts.next().unwrap_or("/");
    let (path, query) = split_target(target);

    let mut content_len = 0usize;
    for line in lines.by_ref() {
        if line.is_empty() {
            break;
        }
        let (k, v) = match line.split_once(':') {
            Some(p) => p,
            None => continue,
        };
        if k.eq_ignore_ascii_case("content-length") {
            content_len = v.trim().parse().unwrap_or(0);
        }
    }
    let header_end = raw.find("\r\n\r\n").map(|i| i + 4).unwrap_or(n);
    let mut body = raw[header_end.min(raw.len())..].as_bytes().to_vec();
    while body.len() < content_len {
        let m = stream.read(&mut buf)?;
        if m == 0 {
            break;
        }
        body.extend_from_slice(&buf[..m]);
    }
    if body.len() > content_len {
        body.truncate(content_len);
    }
    let body_s = String::from_utf8_lossy(&body);

    let (status, ctype, payload) = route(method, path, query, &body_s, state);
    let bytes = payload.into_bytes();
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        bytes.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(&bytes)?;
    stream.flush()?;
    Ok(())
}

fn split_target(t: &str) -> (&str, &str) {
    match t.split_once('?') {
        Some((p, q)) => (p, q),
        None => (t, ""),
    }
}

fn qparam(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        if k == key {
            return Some(url_decode(v));
        }
    }
    None
}

fn url_decode(s: &str) -> String {
    let mut out = String::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() => {
                let hex = &s[i + 1..i + 3];
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    out.push(v as char);
                    i += 3;
                } else {
                    out.push('%');
                    i += 1;
                }
            }
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

fn route(
    method: &str,
    path: &str,
    query: &str,
    body: &str,
    state: &Arc<Mutex<Brain>>,
) -> (&'static str, &'static str, String) {
    if method == "OPTIONS" {
        return ("204 No Content", "text/plain", String::new());
    }
    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => ("200 OK", "text/html", PAGE.to_string()),
        ("GET", "/api/meta") => json_ok(meta_json()),
        ("GET", "/api/health") => {
            let b = state.lock().unwrap();
            json_ok(serde_json::to_value(b.health()).unwrap_or(json!({})))
        }
        ("GET", "/api/memories") => {
            let limit: usize = qparam(query, "limit").and_then(|s| s.parse().ok()).unwrap_or(24);
            let b = state.lock().unwrap();
            json_ok(memories_json(&b, limit))
        }
        ("GET", "/api/recall") => {
            let q = qparam(query, "q").unwrap_or_default();
            let mut b = state.lock().unwrap();
            json_ok(recall_json(&mut b, &q))
        }
        ("GET", "/api/causal") => {
            let q = qparam(query, "q").unwrap_or_default();
            let hops: u32 = qparam(query, "hops").and_then(|s| s.parse().ok()).unwrap_or(6);
            let dir = qparam(query, "dir").unwrap_or_else(|| "causes".into());
            let b = state.lock().unwrap();
            let r = if dir == "effects" {
                b.effects(&q, hops)
            } else {
                b.causes(&q, hops)
            };
            json_ok(serde_json::to_value(r).unwrap_or(json!({})))
        }
        ("GET", "/api/context") => {
            let q = qparam(query, "q").unwrap_or_default();
            let tok: usize = qparam(query, "tokens").and_then(|s| s.parse().ok()).unwrap_or(400);
            let mut b = state.lock().unwrap();
            let p = b.context(&q, tok);
            json_ok(json!({ "query": p.query, "tokens": p.tokens, "memories": p.memories, "text": p.text }))
        }
        ("POST", "/api/remember") => {
            let v: Value = serde_json::from_str(body).unwrap_or(json!({}));
            let content = v.get("content").and_then(|x| x.as_str()).unwrap_or("").trim();
            if content.is_empty() {
                return json_ok(json!({ "error": "empty" }));
            }
            let ty = v
                .get("type")
                .and_then(|x| x.as_str())
                .and_then(MemoryType::parse);
            let mut b = state.lock().unwrap();
            match b.remember_typed(content, ty, vec![], 5) {
                Ok(r) => {
                    let _ = b.save();
                    json_ok(json!({
                        "id": r.fiber.id,
                        "type": r.fiber.memory_type.as_str(),
                        "neurons": r.neurons_created.len(),
                        "synapses": r.synapses_created.len()
                    }))
                }
                Err(e) => json_ok(json!({ "error": e.to_string() })),
            }
        }
        ("POST", "/api/forget") => {
            let v: Value = serde_json::from_str(body).unwrap_or(json!({}));
            let q = v.get("query").and_then(|x| x.as_str()).unwrap_or("");
            let mut b = state.lock().unwrap();
            let id = b.forget(q);
            let _ = b.save();
            json_ok(json!({ "forgot": id }))
        }
        ("POST", "/api/consolidate") => {
            let mut b = state.lock().unwrap();
            let r = b.consolidate();
            let _ = b.save();
            json_ok(serde_json::to_value(r).unwrap_or(json!({})))
        }
        ("POST", "/api/link") => {
            let v: Value = serde_json::from_str(body).unwrap_or(json!({}));
            let from = v.get("from").and_then(|x| x.as_str()).unwrap_or("");
            let to = v.get("to").and_then(|x| x.as_str()).unwrap_or("");
            let ty = v
                .get("type")
                .and_then(|x| x.as_str())
                .unwrap_or("caused_by");
            let syn = match ty {
                "leads_to" => SynapseType::LeadsTo,
                "related_to" => SynapseType::RelatedTo,
                _ => SynapseType::CausedBy,
            };
            let mut b = state.lock().unwrap();
            let ok = b.link(from, to, syn, 0.85).is_some();
            let _ = b.save();
            json_ok(json!({ "ok": ok }))
        }
        _ => ("404 Not Found", "application/json", json!({"error":"not found"}).to_string()),
    }
}

fn json_ok(v: Value) -> (&'static str, &'static str, String) {
    ("200 OK", "application/json", v.to_string())
}

fn meta_json() -> Value {
    let tz = crate::temporal::tz_info();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "now_ms": now_ms(),
        "now_local": crate::temporal::format_local(now_ms()),
        "embed_dim": crate::embed::DIM,
        "tz": {
            "label": tz.label,
            "source": tz.source,
            "offset_ms": tz.offset_ms,
            "offset_hours": tz.offset_hours
        }
    })
}

fn memories_json(brain: &Brain, limit: usize) -> Value {
    let now = now_ms();
    let mut fs = brain.store().fibers_vec();
    fs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    let memories: Vec<Value> = fs
        .into_iter()
        .take(limit)
        .map(|f| {
            json!({
                "id": f.id,
                "summary": f.summary,
                "type": f.memory_type.as_str(),
                "status": f.status.as_str(),
                "stage": f.stage.as_str(),
                "salience": f.salience,
                "when": crate::temporal::format_local(f.created_at),
                "live": f.is_live(now)
            })
        })
        .collect();
    json!({ "memories": memories })
}

fn recall_json(brain: &mut Brain, q: &str) -> Value {
    let r = brain.recall_opts(
        q,
        RecallOpts {
            limit: 8,
            ..Default::default()
        },
    );
    let memories: Vec<Value> = r
        .memories
        .iter()
        .map(|m| {
            json!({
                "id": m.fiber.id,
                "summary": m.fiber.summary,
                "type": m.fiber.memory_type.as_str(),
                "score": m.score,
                "confidence": m.confidence,
                "hop": m.hop,
                "embed": m.embed,
                "reason": m.reason
            })
        })
        .collect();
    json!({
        "query": r.query,
        "elapsed_ms": r.elapsed_ms,
        "activated": r.activations.len(),
        "memories": memories
    })
}
