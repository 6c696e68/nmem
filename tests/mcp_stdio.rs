//! Real MCP stdio tests — spawn `nmem mcp`, speak JSON-RPC NDJSON like OpenClaw.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

struct Mcp {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    id: u64,
}

impl Mcp {
    fn spawn(brain: &PathBuf) -> Self {
        let bin = env!("CARGO_BIN_EXE_nmem");
        let mut child = Command::new(bin)
            .arg("mcp")
            .env("NMEM_BRAIN", brain)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn nmem mcp");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Self {
            child,
            stdin,
            stdout,
            id: 0,
        }
    }

    fn rpc(&mut self, method: &str, params: Value) -> Value {
        self.id += 1;
        let req = json!({
            "jsonrpc": "2.0",
            "id": self.id,
            "method": method,
            "params": params
        });
        writeln!(self.stdin, "{}", req).expect("write");
        self.stdin.flush().expect("flush");
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .expect("read mcp line");
        assert!(!line.is_empty(), "mcp closed unexpectedly on {method}");
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("bad json {e}: {line}"))
    }

    fn notify(&mut self, method: &str, params: Value) {
        let req = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        writeln!(self.stdin, "{}", req).expect("notify");
        self.stdin.flush().expect("flush");
    }

    fn call(&mut self, name: &str, args: Value) -> (bool, String) {
        let r = self.rpc(
            "tools/call",
            json!({ "name": name, "arguments": args }),
        );
        let err = r["result"]["isError"].as_bool().unwrap_or(true);
        let text = r["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();
        (!err, text)
    }
}

impl Drop for Mcp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn tmp_brain(tag: &str) -> PathBuf {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("nmem-mcp-{tag}-{n}.json"));
    let _ = std::fs::remove_file(&p);
    p
}

fn handshake(brain: &PathBuf) -> Mcp {
    let mut m = Mcp::spawn(brain);
    let init = m.rpc(
        "initialize",
        json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "openclaw-test", "version": "0" }
        }),
    );
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(init["result"]["serverInfo"]["name"], "nmem");
    m.notify("notifications/initialized", json!({}));
    m
}

#[test]
fn claw_handshake_and_six_tools() {
    let brain = tmp_brain("hs");
    let mut m = handshake(&brain);
    let listed = m.rpc("tools/list", json!({}));
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
    let _ = std::fs::remove_file(brain);
}

#[test]
fn remember_recall_via_stdio() {
    let brain = tmp_brain("rr");
    let mut m = handshake(&brain);
    let (ok, text) = m.call(
        "nmem_remember",
        json!({ "content": "JWT expiry caused the Tuesday outage because the cron failed" }),
    );
    assert!(ok, "{text}");
    let (ok, text) = m.call("nmem_recall", json!({ "query": "why did the outage happen" }));
    assert!(ok, "{text}");
    assert!(
        text.to_lowercase().contains("jwt") || text.to_lowercase().contains("outage"),
        "{text}"
    );
    let _ = std::fs::remove_file(brain);
}

#[test]
fn persist_survives_process_restart() {
    let brain = tmp_brain("persist");
    {
        let mut m = handshake(&brain);
        let (ok, t) = m.call(
            "nmem_remember",
            json!({ "content": "We decided to use Redis for the session store" }),
        );
        assert!(ok, "{t}");
        drop(m);
    }
    assert!(brain.exists(), "brain file must exist after remember");
    {
        let mut m = handshake(&brain);
        let (ok, text) = m.call("nmem_recall", json!({ "query": "session store" }));
        assert!(ok, "{text}");
        assert!(text.to_lowercase().contains("redis"), "lost after restart: {text}");
    }
    let _ = std::fs::remove_file(brain);
}

#[test]
fn claw_session_hooks() {
    let brain = tmp_brain("hooks");
    let mut m = handshake(&brain);
    let (ok, t) = m.call(
        "nmem_auto",
        json!({
            "action": "process",
            "text": "Fixed auth bug in login.py. JWT cron still failing."
        }),
    );
    assert!(ok, "{t}");
    let (ok, t) = m.call("nmem_consolidate", json!({}));
    assert!(ok, "{t}");
    let (ok, stats) = m.call("nmem_stats", json!({}));
    assert!(ok, "{stats}");
    assert!(stats.contains("fibers="), "{stats}");
    let (ok, health) = m.call("nmem_health", json!({}));
    assert!(ok, "{health}");
    let (ok, ctx) = m.call("nmem_context", json!({ "query": "auth", "token_budget": 200 }));
    assert!(ok, "{ctx}");
    let _ = std::fs::remove_file(brain);
}

#[test]
fn todo_forget_causal() {
    let brain = tmp_brain("tfc");
    let mut m = handshake(&brain);
    let (ok, _) = m.call(
        "nmem_remember",
        json!({ "content": "Tuesday production outage at 15:00 UTC" }),
    );
    assert!(ok);
    let (ok, _) = m.call(
        "nmem_remember",
        json!({ "content": "JWT expiry caused the outage because cron failed" }),
    );
    assert!(ok);
    let (ok, todo) = m.call("nmem_todo", json!({ "task": "page if JWT cron fails" }));
    assert!(ok, "{todo}");
    let (ok, causal) = m.call("nmem_causal", json!({ "query": "outage", "hops": 4 }));
    assert!(ok, "{causal}");
    assert!(
        causal.contains("caused_by") || causal.contains("leads_to") || causal.contains("seed="),
        "{causal}"
    );
    let (ok, forgot) = m.call("nmem_forget", json!({ "query": "page if JWT" }));
    assert!(ok, "{forgot}");
    let _ = std::fs::remove_file(brain);
}

#[test]
fn vietnamese_and_string_id() {
    let brain = tmp_brain("vi");
    let mut m = handshake(&brain);
    // string JSON-RPC id
    let req = r#"{"jsonrpc":"2.0","id":"abc","method":"ping","params":{}}"#;
    writeln!(m.stdin, "{req}").unwrap();
    m.stdin.flush().unwrap();
    let mut line = String::new();
    m.stdout.read_line(&mut line).unwrap();
    let v: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(v["id"], "abc");

    let (ok, t) = m.call(
        "nmem_remember",
        json!({ "content": "Nhận ra rằng cron timezone UTC/ICT lệch 7 tiếng" }),
    );
    assert!(ok, "{t}");
    let (ok, text) = m.call("nmem_recall", json!({ "query": "timezone cron lệch" }));
    assert!(ok, "{text}");
    assert!(
        text.contains("lệch") || text.to_lowercase().contains("timezone"),
        "{text}"
    );
    let _ = std::fs::remove_file(brain);
}

#[test]
fn empty_remember_is_error_and_unknown_tool() {
    let brain = tmp_brain("err");
    let mut m = handshake(&brain);
    let (ok, t) = m.call("nmem_remember", json!({ "content": "   " }));
    assert!(!ok, "empty should error, got {t}");
    let (ok, t) = m.call("nmem_not_a_tool", json!({}));
    assert!(!ok, "unknown tool should error, got {t}");
    let listed = m.rpc("resources/list", json!({}));
    assert!(listed["result"]["resources"].is_array());
    let _ = std::fs::remove_file(brain);
}

#[test]
fn initialize_does_not_wipe_brain() {
    let brain = tmp_brain("wipe");
    {
        let mut m = handshake(&brain);
        let (ok, _) = m.call(
            "nmem_remember",
            json!({ "content": "Always rotate JWT keys every 12 hours" }),
        );
        assert!(ok);
        drop(m);
    }
    {
        let mut m = handshake(&brain);
        // initialize already ran; stats must see the fiber
        let (ok, stats) = m.call("nmem_stats", json!({}));
        assert!(ok, "{stats}");
        assert!(
            !stats.contains("fibers=0"),
            "initialize wiped the brain: {stats}"
        );
    }
    let _ = std::fs::remove_file(brain);
}
