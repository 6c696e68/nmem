//! MCP stdio server — JSON-RPC 2.0, newline-delimited (OpenClaw / Claude Code).
//!
//! Lean tool set Claw actually calls. Not the 63-tool Python kitchen sink
//! (that blows the agent's context window on weak hardware).

use crate::types::{MemoryType, SynapseType, now_ms};
use crate::Brain;
use crate::RecallOpts;
use crate::Store;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::PathBuf;

const PROTOCOL: &str = "2024-11-05";
const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn default_brain_path() -> PathBuf {
    if let Ok(p) = std::env::var("NMEM_BRAIN") {
        return PathBuf::from(p);
    }
    let name = std::env::var("NEURALMEMORY_BRAIN").unwrap_or_else(|_| "default".into());
    let name = if name.ends_with(".json") {
        name
    } else {
        format!("{name}.json")
    };
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".neuralmemory")
        .join("brains")
        .join(name)
}

pub fn run_stdio(brain_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut brain = Brain::open(&brain_path)?;
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut lines = stdin.lock().lines();
    while let Some(res) = lines.next() {
        let line = match res {
            Ok(l) => l,
            Err(_) => continue, // bad utf-8 / interrupted — don't die
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let mutating = is_mutating(&req);
        let mut resp = handle(&mut brain, &req);
        if mutating {
            if let Err(e) = brain.save() {
                eprintln!("nmem persist: {e}");
                if let Some(r) = resp.as_mut() {
                    r["result"] = json!({
                        "content": [{ "type": "text", "text": format!("persist failed: {e}") }],
                        "isError": true
                    });
                }
            }
        }
        if let Some(resp) = resp {
            writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
            stdout.flush()?;
        }
    }
    Ok(())
}

fn is_mutating(req: &Value) -> bool {
    if req.get("method").and_then(|m| m.as_str()) != Some("tools/call") {
        return false;
    }
    matches!(
        req.pointer("/params/name").and_then(|n| n.as_str()).unwrap_or(""),
        "nmem_remember"
            | "nmem_todo"
            | "nmem_forget"
            | "nmem_link"
            | "nmem_auto"
            | "nmem_consolidate"
            | "nmem_recall"
            | "nmem_context"
    )
}

pub fn handle(brain: &mut Brain, req: &Value) -> Option<Value> {
    let id = req.get("id").cloned();
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    if id.is_none() {
        return None; // notification
    }
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": "nmem", "version": VERSION }
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_schemas() })),
        "resources/list" => Ok(json!({ "resources": [] })),
        "prompts/list" => Ok(json!({ "prompts": [] })),
        "tools/call" => {
            let params = req.get("params").cloned().unwrap_or(json!({}));
            let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            match call_tool(brain, name, &args) {
                Ok(text) => Ok(json!({
                    "content": [{ "type": "text", "text": text }],
                    "isError": false
                })),
                Err(e) => Ok(json!({
                    "content": [{ "type": "text", "text": e }],
                    "isError": true
                })),
            }
        }
        _ => Err(json!({ "code": -32601, "message": format!("unknown method {method}") })),
    };
    Some(match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err(e) => json!({ "jsonrpc": "2.0", "id": id, "error": e }),
    })
}

fn tool_schemas() -> Vec<Value> {
    vec![
        tool(
            "nmem_remember",
            "Store a memory (fact, decision, error, preference, insight). Auto-detects type.",
            json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string" },
                    "type": { "type": "string" },
                    "priority": { "type": "integer" },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "expires_days": { "type": "integer" }
                },
                "required": ["content"]
            }),
        ),
        tool(
            "nmem_recall",
            "Query memories via spreading activation. Use for past decisions, errors, context.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "depth": { "type": "integer" },
                    "max_tokens": { "type": "integer" },
                    "limit": { "type": "integer" }
                },
                "required": ["query"]
            }),
        ),
        tool(
            "nmem_context",
            "Get recent / relevant memories as prompt context.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer" },
                    "token_budget": { "type": "integer" }
                }
            }),
        ),
        tool(
            "nmem_todo",
            "Store a TODO (auto-expires in 30 days).",
            json!({
                "type": "object",
                "properties": {
                    "task": { "type": "string" },
                    "priority": { "type": "integer" }
                },
                "required": ["task"]
            }),
        ),
        tool(
            "nmem_stats",
            "Brain statistics: neuron / synapse / fiber counts.",
            json!({ "type": "object", "properties": {} }),
        ),
        tool(
            "nmem_health",
            "Brain health grade and issues.",
            json!({ "type": "object", "properties": {} }),
        ),
        tool(
            "nmem_forget",
            "Delete a memory by id or query.",
            json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        ),
        tool(
            "nmem_causal",
            "Walk CAUSED_BY / LEADS_TO from a memory.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "hops": { "type": "integer" },
                    "direction": { "type": "string", "enum": ["causes", "effects"] }
                },
                "required": ["query"]
            }),
        ),
        tool(
            "nmem_consolidate",
            "Decay, promote stages, merge near-duplicates, expire stale memories.",
            json!({ "type": "object", "properties": {} }),
        ),
        tool(
            "nmem_show",
            "Full fiber by id.",
            json!({
                "type": "object",
                "properties": { "memory_id": { "type": "string" } },
                "required": ["memory_id"]
            }),
        ),
        tool(
            "nmem_link",
            "Manually wire two memories (caused_by, leads_to, ...).",
            json!({
                "type": "object",
                "properties": {
                    "from": { "type": "string" },
                    "to": { "type": "string" },
                    "type": { "type": "string" },
                    "weight": { "type": "number" }
                },
                "required": ["from", "to"]
            }),
        ),
        tool(
            "nmem_auto",
            "Auto-capture text as memories (OpenClaw session end / compaction).",
            json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string" },
                    "text": { "type": "string" }
                },
                "required": ["text"]
            }),
        ),
    ]
}

fn tool(name: &str, description: &str, input_schema: Value) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": input_schema
    })
}

fn call_tool(brain: &mut Brain, name: &str, args: &Value) -> Result<String, String> {
    match name {
        "nmem_remember" => {
            let content = args_str(args, "content")?;
            let ty = args
                .get("type")
                .and_then(|v| v.as_str())
                .and_then(MemoryType::parse);
            let tags = args_tags(args);
            let pri = args.get("priority").and_then(|v| v.as_u64()).unwrap_or(5) as u8;
            let r = brain
                .remember_typed(&content, ty, tags, pri)
                .map_err(|e| e.to_string())?;
            if let Some(days) = args.get("expires_days").and_then(|v| v.as_u64()) {
                if let Some(f) = brain.store_mut().get_fiber_mut(&r.fiber.id) {
                    f.expires_at = Some(now_ms() + days * 86_400_000);
                }
            }
            Ok(format!(
                "ok type={} id={} n+{} s+{}",
                r.fiber.memory_type.as_str(),
                r.fiber.id,
                r.neurons_created.len(),
                r.synapses_created.len()
            ))
        }
        "nmem_recall" => {
            let query = args_str(args, "query")?;
            let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(1);
            let hops = match depth {
                0 => 1,
                1 => 3,
                2 => 4,
                _ => 5,
            };
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
            let budget = args
                .get("max_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(500) as usize;
            let r = brain.recall_opts(
                &query,
                RecallOpts {
                    depth: Some(hops),
                    limit,
                    ..Default::default()
                },
            );
            Ok(format_recall(&r, budget))
        }
        "nmem_context" => {
            let q = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let budget = args
                .get("token_budget")
                .or_else(|| args.get("max_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(400) as usize;
            if q.is_empty() {
                Ok(recent_block(brain, args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize, budget))
            } else {
                let pack = brain.context(&q, budget);
                Ok(if pack.text.is_empty() {
                    "(no memories)".into()
                } else {
                    pack.text
                })
            }
        }
        "nmem_todo" => {
            let task = args
                .get("task")
                .or_else(|| args.get("content"))
                .and_then(|v| v.as_str())
                .ok_or("missing task")?
                .to_string();
            let pri = args.get("priority").and_then(|v| v.as_u64()).unwrap_or(5) as u8;
            let r = brain
                .remember_typed(&task, Some(MemoryType::Todo), vec!["todo".into()], pri)
                .map_err(|e| e.to_string())?;
            Ok(format!("todo id={}", r.fiber.id))
        }
        "nmem_stats" => {
            let s = brain.store();
            let tz = crate::temporal::tz_info();
            Ok(format!(
                "neurons={} synapses={} fibers={} session={} tz={} offset_h={:+} embed_dim={}",
                s.neuron_count(),
                s.synapse_count(),
                s.fiber_count(),
                brain.session_size(),
                tz.label,
                tz.offset_hours,
                crate::embed::DIM
            ))
        }
        "nmem_health" => {
            let h = brain.health();
            let mut out = format!(
                "health {}/100 grade={} n={} s={} f={}",
                h.score, h.grade, h.neurons, h.synapses, h.fibers
            );
            for i in h.issues {
                out.push_str("\n- ");
                out.push_str(&i);
            }
            Ok(out)
        }
        "nmem_forget" => {
            let q = args
                .get("query")
                .or_else(|| args.get("memory_id"))
                .and_then(|v| v.as_str())
                .ok_or("missing query")?;
            match brain.forget(q) {
                Some(id) => Ok(format!("forgot {id}")),
                None => Ok("nothing matched".into()),
            }
        }
        "nmem_causal" => {
            let q = args_str(args, "query")?;
            let hops = args.get("hops").and_then(|v| v.as_u64()).unwrap_or(6) as u32;
            let dir = args.get("direction").and_then(|v| v.as_str()).unwrap_or("causes");
            let r = if dir == "effects" {
                brain.effects(&q, hops)
            } else {
                brain.causes(&q, hops)
            };
            let mut out = format!(
                "{} seed={} conf={:.2}",
                dir,
                r.seed.unwrap_or_else(|| "-".into()),
                r.confidence
            );
            for hop in r.chain {
                out.push_str(&format!(
                    "\n[{}] --{}--> {}",
                    hop.depth, hop.synapse, hop.to_content
                ));
            }
            Ok(out)
        }
        "nmem_consolidate" => {
            let r = brain.consolidate();
            Ok(format!(
                "decayed={} pruned={} merged={} expired={} promoted={}",
                r.synapses_decayed,
                r.synapses_pruned,
                r.fibers_merged,
                r.expired,
                r.stages_promoted
            ))
        }
        "nmem_show" => {
            let id = args_str(args, "memory_id")?;
            let f = brain
                .store()
                .get_fiber(&id)
                .ok_or("not found")?
                .clone();
            Ok(format!(
                "[{}|{}|{}] {}\nid={} belief={:.2} cond={:.2} live={}",
                f.memory_type.as_str(),
                f.stage.as_str(),
                f.status.as_str(),
                f.summary,
                f.id,
                f.belief,
                f.conductivity,
                f.is_live(now_ms())
            ))
        }
        "nmem_link" => {
            let a = args_str(args, "from")?;
            let b = args_str(args, "to")?;
            let ty = args
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("caused_by");
            let w = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.85);
            match brain.link(&a, &b, parse_syn(ty), w) {
                Some(s) => Ok(format!("linked --{}-->", s.type_.as_str())),
                None => Err("could not find both memories".into()),
            }
        }
        "nmem_auto" => {
            let text = args_str(args, "text")?;
            let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("process");
            if action == "flush" || action == "process" {
                let chunks = split_capture(&text);
                let mut n = 0u32;
                for c in chunks {
                    if brain.remember(&c).is_ok() {
                        n += 1;
                    }
                }
                Ok(format!("auto stored={n}"))
            } else {
                Ok("noop".into())
            }
        }
        other => Err(format!("unknown tool {other}")),
    }
}

fn args_str(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("missing {key}"))
}

fn args_tags(args: &Value) -> Vec<String> {
    args.get("tags")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

fn format_recall(r: &crate::RecallResult, budget: usize) -> String {
    if r.memories.is_empty() {
        return "(no memories)".into();
    }
    let mut out = String::new();
    for m in &r.memories {
        let line = format!(
            "- [{}|{:.2}] {}\n",
            m.fiber.memory_type.as_str(),
            m.confidence,
            m.fiber.summary
        );
        if out.len() + line.len() > budget.saturating_mul(4) && !out.is_empty() {
            break;
        }
        out.push_str(&line);
    }
    out
}

fn recent_block(brain: &Brain, limit: usize, budget: usize) -> String {
    let now = now_ms();
    let mut fs = brain.store().fibers_vec();
    fs.retain(|f| f.is_live(now));
    fs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    let mut out = String::new();
    for f in fs.into_iter().take(limit) {
        let line = format!("- [{}] {}\n", f.memory_type.as_str(), f.summary);
        if out.len() + line.len() > budget.saturating_mul(4) && !out.is_empty() {
            break;
        }
        out.push_str(&line);
    }
    if out.is_empty() {
        "(no memories)".into()
    } else {
        out
    }
}

fn split_capture(text: &str) -> Vec<String> {
    let t = text.trim();
    if t.is_empty() {
        return vec![];
    }
    let parts: Vec<String> = t
        .split("\n\n")
        .map(|s| s.trim().to_string())
        .filter(|s| s.chars().count() >= 12)
        .take(8)
        .collect();
    if parts.is_empty() {
        vec![t.chars().take(2000).collect()]
    } else {
        parts
    }
}

fn parse_syn(s: &str) -> SynapseType {
    match s {
        "caused_by" => SynapseType::CausedBy,
        "leads_to" => SynapseType::LeadsTo,
        "related_to" => SynapseType::RelatedTo,
        "contradicts" => SynapseType::Contradicts,
        "resolved_by" => SynapseType::ResolvedBy,
        "supersedes" => SynapseType::Supersedes,
        "enables" => SynapseType::Enables,
        "prevents" => SynapseType::Prevents,
        _ => SynapseType::RelatedTo,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rpc(method: &str, id: u64, params: Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
    }

    #[test]
    fn initialize_and_list_contains_claw_tools() {
        let mut b = Brain::new("mcp");
        let init = handle(&mut b, &rpc("initialize", 1, json!({}))).unwrap();
        assert_eq!(init["result"]["protocolVersion"], PROTOCOL);
        let listed = handle(&mut b, &rpc("tools/list", 2, json!({}))).unwrap();
        let names: Vec<String> = listed["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        for need in [
            "nmem_remember",
            "nmem_recall",
            "nmem_context",
            "nmem_todo",
            "nmem_stats",
            "nmem_health",
            "nmem_auto",
            "nmem_consolidate",
        ] {
            assert!(names.iter().any(|n| n == need), "missing {need} in {names:?}");
        }
    }

    #[test]
    fn remember_then_recall_roundtrip() {
        let mut b = Brain::new("mcp2");
        let call = |name: &str, args: Value| {
            rpc(
                "tools/call",
                3,
                json!({ "name": name, "arguments": args }),
            )
        };
        let r = handle(
            &mut b,
            &call(
                "nmem_remember",
                json!({ "content": "JWT expiry caused the Tuesday outage" }),
            ),
        )
        .unwrap();
        assert_eq!(r["result"]["isError"], false);
        let r = handle(
            &mut b,
            &call("nmem_recall", json!({ "query": "why outage" })),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.to_lowercase().contains("jwt") || text.to_lowercase().contains("outage"), "{text}");
    }
}
