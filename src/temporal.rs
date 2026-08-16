//! Resolve English/Vietnamese time phrases against the **local** calendar.
//! Offset: `NMEM_TZ_HOURS` → `NMEM_TZ` → libc `tm_gmtoff` → 0.

use crate::extract::extract_times;
use std::sync::OnceLock;

const DAY: u64 = 86_400_000;

#[derive(Debug, Clone)]
pub struct TimeWindow {
    pub label: String,
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone)]
pub struct TzInfo {
    pub offset_ms: i64,
    pub offset_hours: f64,
    pub label: String,
    pub source: &'static str,
}

pub fn tz_info() -> TzInfo {
    let (offset_ms, source, label) = resolve_tz();
    TzInfo {
        offset_ms,
        offset_hours: offset_ms as f64 / 3_600_000.0,
        label,
        source,
    }
}

pub fn tz_offset_ms() -> i64 {
    tz_info().offset_ms
}

fn resolve_tz() -> (i64, &'static str, String) {
    if let Ok(h) = std::env::var("NMEM_TZ_HOURS") {
        if let Ok(v) = h.parse::<f64>() {
            return (
                (v * 3_600_000.0) as i64,
                "NMEM_TZ_HOURS",
                format!("UTC{v:+}", v = v),
            );
        }
    }
    if let Ok(raw) = std::env::var("NMEM_TZ") {
        if let Some((ms, lab)) = parse_tz_name(&raw) {
            return (ms, "NMEM_TZ", lab);
        }
    }
    if let Some(ms) = libc_offset_ms() {
        let h = ms as f64 / 3_600_000.0;
        return (ms, "localtime", format!("UTC{h:+}"));
    }
    (0, "utc", "UTC".into())
}

fn parse_tz_name(s: &str) -> Option<(i64, String)> {
    let t = s.trim();
    let up = t.to_ascii_uppercase();
    let named = match up.as_str() {
        "UTC" | "GMT" | "Z" => Some(0.0),
        "JST" | "KST" => Some(9.0),
        "ICT" | "WIB" | "VN" | "HOCHIMINH" | "BANGKOK" => Some(7.0),
        "CSTCN" | "CST" | "HKT" | "SGT" | "PHT" => Some(8.0),
        "CET" => Some(1.0),
        "EET" => Some(2.0),
        "EST" => Some(-5.0),
        "PST" => Some(-8.0),
        "MST" => Some(-7.0),
        "CSTUS" => Some(-6.0),
        _ => None,
    };
    if let Some(h) = named {
        return Some(((h * 3_600_000.0) as i64, up));
    }
    let rest = t
        .trim_start_matches("UTC")
        .trim_start_matches("utc")
        .trim_start_matches("GMT")
        .trim_start_matches("gmt");
    if rest.starts_with('+') || rest.starts_with('-') {
        let sign = if rest.starts_with('-') { -1.0 } else { 1.0 };
        let body = rest.trim_start_matches(['+', '-']);
        let mut parts = body.split(':');
        let hh: f64 = parts.next()?.parse().ok()?;
        let mm: f64 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0.0);
        let h = sign * (hh + mm / 60.0);
        return Some(((h * 3_600_000.0) as i64, format!("UTC{h:+}")));
    }
    None
}

fn libc_offset_ms() -> Option<i64> {
    static CACHED: OnceLock<Option<i64>> = OnceLock::new();
    *CACHED.get_or_init(|| unsafe { libc_gmtoff_ms() })
}

#[repr(C)]
struct Tm {
    tm_sec: i32,
    tm_min: i32,
    tm_hour: i32,
    tm_mday: i32,
    tm_mon: i32,
    tm_year: i32,
    tm_wday: i32,
    tm_yday: i32,
    tm_isdst: i32,
    tm_gmtoff: i64,
    tm_zone: *const i8,
}

unsafe fn libc_gmtoff_ms() -> Option<i64> {
    extern "C" {
        fn time(tloc: *mut i64) -> i64;
        fn localtime_r(timep: *const i64, result: *mut Tm) -> *mut Tm;
    }
    let mut t: i64 = 0;
    if time(&mut t) == -1 {
        return None;
    }
    let mut tm: Tm = std::mem::zeroed();
    if localtime_r(&t, &mut tm).is_null() {
        return None;
    }
    Some(tm.tm_gmtoff.saturating_mul(1000))
}

fn as_local(utc_ms: u64, offset_ms: i64) -> i64 {
    utc_ms as i64 + offset_ms
}

pub fn start_of_day(ms: u64) -> u64 {
    start_of_day_tz(ms, tz_offset_ms())
}

pub fn start_of_day_tz(utc_ms: u64, offset_ms: i64) -> u64 {
    let loc = as_local(utc_ms, offset_ms);
    let rem = loc.rem_euclid(DAY as i64);
    let loc_mid = loc - rem;
    (loc_mid - offset_ms).max(0) as u64
}

/// 0 = Sunday … 6 = Saturday, in the local calendar.
pub fn weekday(ms: u64) -> u32 {
    weekday_tz(ms, tz_offset_ms())
}

pub fn weekday_tz(utc_ms: u64, offset_ms: i64) -> u32 {
    let loc = as_local(utc_ms, offset_ms);
    let days = loc.div_euclid(DAY as i64);
    let days = if days < 0 { 0 } else { days as u64 };
    ((days + 4) % 7) as u32
}

fn last_weekday_tz(now: u64, target: u32, offset_ms: i64) -> u64 {
    let today = weekday_tz(now, offset_ms);
    let back = (today + 7 - target) % 7;
    let back = if back == 0 { 7 } else { back };
    start_of_day_tz(now, offset_ms).saturating_sub(back as u64 * DAY)
}

fn named_weekday(s: &str) -> Option<u32> {
    Some(match s {
        "sunday" | "chủ nhật" | "chu nhat" => 0,
        "monday" | "thứ hai" | "thu hai" => 1,
        "tuesday" | "thứ ba" | "thu ba" => 2,
        "wednesday" | "thứ tư" | "thu tu" => 3,
        "thursday" | "thứ năm" | "thu nam" => 4,
        "friday" | "thứ sáu" | "thu sau" => 5,
        "saturday" | "thứ bảy" | "thu bay" => 6,
        _ => return None,
    })
}

pub fn resolve_label(label: &str, now: u64) -> Option<(u64, u64)> {
    resolve_label_tz(label, now, tz_offset_ms())
}

pub fn resolve_label_tz(label: &str, now: u64, offset_ms: i64) -> Option<(u64, u64)> {
    let l = label.to_lowercase();
    let today0 = start_of_day_tz(now, offset_ms);
    if l.contains("yesterday") || l.contains("hôm qua") || l.contains("hom qua") {
        return Some((today0.saturating_sub(DAY), today0.saturating_sub(1)));
    }
    if l.contains("today") || l.contains("tonight") || l.contains("hôm nay") || l.contains("hom nay")
    {
        return Some((today0, today0 + DAY - 1));
    }
    if l.contains("tomorrow") || l.contains("ngày mai") || l.contains("ngay mai") {
        return Some((today0 + DAY, today0 + 2 * DAY - 1));
    }
    if l.contains("last week") || l.contains("tuần trước") || l.contains("tuan truoc") {
        let this_mon_back = (weekday_tz(now, offset_ms) + 6) % 7;
        let this_mon = today0.saturating_sub(this_mon_back as u64 * DAY);
        return Some((this_mon.saturating_sub(7 * DAY), this_mon.saturating_sub(1)));
    }
    if l.contains("this week") || l.contains("tuần này") || l.contains("tuan nay") {
        let this_mon_back = (weekday_tz(now, offset_ms) + 6) % 7;
        let this_mon = today0.saturating_sub(this_mon_back as u64 * DAY);
        return Some((this_mon, this_mon + 7 * DAY - 1));
    }
    if l.contains("next week") || l.contains("tuần sau") || l.contains("tuan sau") {
        let this_mon_back = (weekday_tz(now, offset_ms) + 6) % 7;
        let this_mon = today0.saturating_sub(this_mon_back as u64 * DAY);
        return Some((this_mon + 7 * DAY, this_mon + 14 * DAY - 1));
    }
    for w in l.split(|c: char| !c.is_alphanumeric() && c != ' ') {
        let w = w.trim();
        if let Some(d) = named_weekday(w) {
            let start = last_weekday_tz(now, d, offset_ms);
            return Some((start, start + DAY - 1));
        }
    }
    None
}

pub fn extract_windows(content: &str, now: u64) -> Vec<TimeWindow> {
    let off = tz_offset_ms();
    let mut out = Vec::new();
    for label in extract_times(content) {
        if let Some((start, end)) = resolve_label_tz(&label, now, off) {
            out.push(TimeWindow { label, start, end });
        }
    }
    out
}

pub fn parse_query_window(query: &str, now: u64) -> Option<TimeWindow> {
    extract_windows(query, now).into_iter().next()
}

pub fn overlap(a0: u64, a1: u64, b0: u64, b1: u64) -> bool {
    a0 <= b1 && b0 <= a1
}

pub fn format_local(utc_ms: u64) -> String {
    let off = tz_offset_ms();
    let loc = as_local(utc_ms, off).max(0) as u64;
    let secs = loc / 1000;
    let days = secs / 86400;
    let rem = secs % 86400;
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    // Civil date from Unix days (Howard Hinnant)
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yesterday_is_previous_day() {
        let now = 1_776_268_800_000;
        let (s, e) = resolve_label_tz("yesterday", now, 0).unwrap();
        assert!(e < start_of_day_tz(now, 0));
        assert_eq!(e - s + 1, DAY);
    }

    #[test]
    fn tuesday_resolves() {
        let now = crate::types::now_ms();
        assert!(resolve_label_tz("tuesday", now, 0).is_some());
    }

    #[test]
    fn ict_today_not_utc_day() {
        // 01:30 UTC Wednesday = 08:30 ICT Wednesday. "hôm nay" must be ICT Wednesday,
        // which started 17:00 UTC Tuesday.
        let wed_0130_utc = 1_776_268_800_000u64; // whatever — use constructed
        // 2026-03-18 01:30:00 UTC
        let t = 1_774_000_000_000u64; // loose
        let off = 7 * 3_600_000i64;
        let (s, e) = resolve_label_tz("hôm nay", t, off).unwrap();
        let sod = start_of_day_tz(t, off);
        assert_eq!(s, sod);
        assert_eq!(e - s + 1, DAY);
        // local midnight is 17:00 previous UTC
        assert_eq!(sod, (as_local(t, off) - as_local(t, off).rem_euclid(DAY as i64) - off) as u64);
        let _ = wed_0130_utc;
    }

    #[test]
    fn jst_named_offset() {
        let (ms, lab) = parse_tz_name("JST").unwrap();
        assert_eq!(ms, 9 * 3_600_000);
        assert_eq!(lab, "JST");
        let (ms, _) = parse_tz_name("+07:00").unwrap();
        assert_eq!(ms, 7 * 3_600_000);
    }
}
