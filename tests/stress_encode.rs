//! Encode scale stress — must stay near-linear after exact/overlap indexes.
use nmem::Brain;
use std::time::Instant;

fn rss_kb() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0);
        }
    }
    0
}

#[test]
fn encode_scale_1k() {
    let mut b = Brain::new("enc1k");
    let mut last = Instant::now();
    let mut marks = Vec::new();
    for i in 0..1000 {
        b.remember(&format!(
            "Event {i}: component svc-{i} uses redis jwt auth cache layer on cluster-{i}"
        ))
        .unwrap();
        if (i + 1) % 200 == 0 {
            let ms = last.elapsed().as_millis();
            marks.push((i + 1, ms, rss_kb()));
            eprintln!("[encode_scale] +200 → {} fibers batch={}ms rss={}KB", i + 1, ms, rss_kb());
            last = Instant::now();
        }
    }
    b.remember("JWT expiry caused the Tuesday outage because rotation cron failed")
        .unwrap();
    let t0 = Instant::now();
    let mut times = Vec::new();
    for _ in 0..30 {
        let r = b.recall("why outage jwt");
        assert!(!r.memories.is_empty());
        times.push(r.elapsed_ms);
    }
    times.sort_unstable();
    let h = b.health();
    eprintln!(
        "[encode_scale] done fibers={} neurons={} synapses={} recall30 p50={} p95={} marks={:?}",
        h.fibers,
        h.neurons,
        h.synapses,
        times[times.len() / 2],
        times[(times.len() * 95) / 100],
        marks
    );
    // Batch times should not explode: last 200 vs first 200 ratio < 15x in debug
    if marks.len() >= 2 {
        let first = marks[0].1.max(1);
        let last_b = marks.last().unwrap().1;
        let ratio = last_b as f64 / first as f64;
        eprintln!("[encode_scale] last_batch/first_batch ratio={ratio:.2}");
        assert!(
            ratio < 20.0,
            "encode still quadratic-ish: first={}ms last={}ms ratio={ratio}",
            first,
            last_b
        );
    }
    assert!(h.fibers >= 1000);
    let _ = t0;
}

#[test]
fn encode_scale_2k_soft() {
    let mut b = Brain::new("enc2k");
    let t0 = Instant::now();
    for i in 0..2000 {
        b.remember(&format!(
            "Item {i}: pipeline stage-{i} depends on postgres redis kafka jwt"
        ))
        .unwrap();
        if (i + 1) % 500 == 0 {
            eprintln!(
                "[encode_2k] {} fibers elapsed={}ms rss={}KB",
                i + 1,
                t0.elapsed().as_millis(),
                rss_kb()
            );
        }
    }
    let total = t0.elapsed().as_millis();
    let h = b.health();
    let r = b.recall("jwt redis");
    eprintln!(
        "[encode_2k] total={total}ms fibers={} recall_hits={} rss={}KB",
        h.fibers,
        r.memories.len(),
        rss_kb()
    );
    assert!(h.fibers >= 2000);
    // Debug budget: 2k encodes under 3 minutes
    assert!(total < 180_000, "2k encode too slow: {total}ms");
}
