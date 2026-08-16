//! Hit the real dashboard HTTP server.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::Duration;

fn wait_port(addr: &str) {
    for _ in 0..50 {
        if TcpStream::connect(addr).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(40));
    }
    panic!("dashboard did not bind {addr}");
}

fn http_get(addr: &str, path: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes())
        .unwrap();
    let mut buf = String::new();
    s.read_to_string(&mut buf).unwrap();
    let status = buf
        .split_whitespace()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    (status, buf)
}

#[test]
fn dashboard_serves_page_and_recall() {
    let port = 18080 + (std::process::id() % 400) as u16;
    let addr = format!("127.0.0.1:{port}");
    let brain = std::env::temp_dir().join(format!("nmem-dash-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&brain);
    let mut child = Command::new(env!("CARGO_BIN_EXE_nmem"))
        .args(["--brain", brain.to_str().unwrap(), "dash", "--host", "127.0.0.1", "--port", &port.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_port(&addr);
    let (st, body) = http_get(&addr, "/");
    assert_eq!(st, 200);
    assert!(body.contains("Neural memory"), "{body}");
    let (st, body) = http_get(&addr, "/api/meta");
    assert_eq!(st, 200);
    assert!(body.contains("embed_dim"), "{body}");
    assert!(body.contains("offset_hours"), "{body}");
    let (st, body) = http_get(&addr, "/api/health");
    assert_eq!(st, 200);
    assert!(body.contains("\"fibers\""), "{body}");
    let (st, body) = http_get(&addr, "/api/recall?q=outage");
    assert_eq!(st, 200);
    assert!(
        body.to_lowercase().contains("jwt") || body.to_lowercase().contains("outage"),
        "{body}"
    );
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_file(brain);
}
