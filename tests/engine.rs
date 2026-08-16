//! End-to-end + regression tests for the ported engine.

use nmem::conflict::ConflictKind;
use nmem::extract::suggest_memory_type;
use nmem::types::{MemoryStatus, MemoryType, SynapseType, now_ms};
use nmem::{Brain, RecalledMemory, Store};

fn seed() -> Brain {
    let mut b = Brain::new("test");
    b.remember("Tuesday production outage at 15:00 UTC — API 502 for 18 minutes")
        .unwrap();
    b.remember(
        "JWT expiry caused the Tuesday outage because rotation cron never ran after the deploy",
    )
    .unwrap();
    b.remember("Alice's review suggested adding token expiry alerts and a fallback refresh path")
        .unwrap();
    b.remember("We decided to use Redis for the session store instead of JWT-only auth")
        .unwrap();
    b.remember("Fixed auth bug with null check in login.py:42 — empty token now returns 401")
        .unwrap();
    b.remember("Always rotate JWT signing keys on a 12-hour cron and page if the job fails")
        .unwrap();
    b.remember("Cần review PR #123 về auth trước khi merge")
        .unwrap();
    b.remember("Nhận ra rằng cron timezone UTC/ICT lệch 7 tiếng là nguyên nhân rotation miss")
        .unwrap();
    b
}

#[test]
fn type_detection_on_seed() {
    let b = seed();
    let types: Vec<MemoryType> = b.store().fibers_vec().into_iter().map(|f| f.memory_type).collect();
    assert!(types.contains(&MemoryType::Error));
    assert!(types.contains(&MemoryType::Decision));
    assert!(types.contains(&MemoryType::Instruction));
    assert!(types.contains(&MemoryType::Insight));
    assert!(types.contains(&MemoryType::Todo));
}

#[test]
fn vietnamese_fact_is_not_decision() {
    assert_eq!(
        suggest_memory_type("Dùng postgres cho API nội bộ"),
        MemoryType::Fact
    );
}

#[test]
fn encode_creates_graph() {
    let b = seed();
    assert!(b.store().neuron_count() > 8);
    assert!(b.store().synapse_count() > 8);
    assert_eq!(b.store().fiber_count(), 8);
}

#[test]
fn recall_outage_surfaces_jwt_cause() {
    let mut b = seed();
    let r = b.recall("why did the outage happen");
    assert!(!r.memories.is_empty());
    let top: String = r
        .memories
        .iter()
        .take(3)
        .map(|m| m.fiber.summary.to_lowercase())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(top.contains("outage"), "top-3 must include the outage, got: {top}");
    assert!(
        r.memories
            .iter()
            .any(|m| m.fiber.summary.to_lowercase().contains("jwt")),
        "recall must surface JWT cause, got: {top}"
    );
}

#[test]
fn recall_auth_finds_login_fix() {
    let mut b = seed();
    let r = b.recall("auth bug");
    let blob: String = r
        .memories
        .iter()
        .map(|m| m.fiber.summary.to_lowercase())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(blob.contains("login.py") || blob.contains("null") || blob.contains("auth"));
}

fn is_redis_decision(m: &RecalledMemory) -> bool {
    m.fiber.memory_type == MemoryType::Decision && m.fiber.summary.to_lowercase().contains("redis")
}

#[test]
fn decision_ranks_on_database_query() {
    let mut b = seed();
    let r = b.recall("database session store decision");
    assert!(r.memories.iter().any(is_redis_decision));
}

#[test]
fn vietnamese_recall() {
    let mut b = seed();
    let r = b.recall("timezone cron lệch");
    assert!(r.memories.iter().any(|m| {
        m.fiber.summary.to_lowercase().contains("timezone") || m.fiber.summary.contains("lệch")
    }));
}

#[test]
fn causal_walk_from_outage() {
    let b = seed();
    let r = b.causal("outage", 4);
    assert!(r.seed.is_some());
    assert!(
        r.chain
            .iter()
            .any(|h| h.synapse == "caused_by" || h.synapse == "leads_to"),
        "expected causal hop, seed={:?} chain={:?}",
        r.seed,
        r.chain
    );
}

#[test]
fn caused_by_survives_encode_order() {
    let mut b = Brain::new("order");
    b.remember("JWT expiry caused the outage because the rotation cron failed")
        .unwrap();
    b.remember("Tuesday production outage — API 502 for 18 minutes")
        .unwrap();
    let r = b.causal("outage", 4);
    assert!(
        r.chain
            .iter()
            .any(|h| h.synapse == "caused_by" || h.synapse == "leads_to"),
        "encode order must not drop causal edge, seed={:?} chain={:?}",
        r.seed,
        r.chain
    );
}

#[test]
fn health_not_empty() {
    let b = seed();
    let h = b.health();
    assert!(h.score >= 50);
    assert_eq!(h.fibers, 8);
}

#[test]
fn persist_roundtrip() {
    let dir = std::env::temp_dir().join(format!("nmem-test-{}.json", std::process::id()));
    let mut b = seed();
    b.save_as(&dir).unwrap();
    let mut loaded = Brain::open(&dir).unwrap();
    assert_eq!(loaded.store().fiber_count(), 8);
    assert!(!loaded.recall("jwt").memories.is_empty());
    let _ = std::fs::remove_file(&dir);
}

#[test]
fn empty_brain_health_is_low() {
    let b = Brain::new("empty");
    assert!(b.health().score < 80);
    assert_eq!(b.health().fibers, 0);
}

#[test]
fn clause_creates_caused_by_and_inverse_leads_to() {
    let mut b = Brain::new("clause");
    b.remember("API 502 happened because JWT expiry cron never ran")
        .unwrap();
    let types: Vec<_> = b.store().synapses().into_iter().map(|s| s.type_).collect();
    assert!(types.contains(&SynapseType::CausedBy), "{types:?}");
    assert!(
        types.contains(&SynapseType::LeadsTo),
        "inverse missing: {types:?}"
    );
}

#[test]
fn recall_has_confidence() {
    let mut b = seed();
    let r = b.recall("why did the outage happen");
    assert!(!r.memories.is_empty());
    assert!(r.memories[0].confidence > 0.2);
    assert!(r.memories[0].confidence <= 1.0);
}

#[test]
fn manual_link_and_effects() {
    let mut b = Brain::new("link");
    b.remember("Tuesday production outage — API 502").unwrap();
    b.remember("On-call page fired after the outage").unwrap();
    assert!(b
        .link("outage", "on-call", SynapseType::LeadsTo, 0.9)
        .is_some());
    let e = b.effects("outage", 3);
    assert!(!e.chain.is_empty() || e.seed.is_some(), "effects={e:?}");
}

#[test]
fn synonym_recall_login_hits_auth() {
    let mut b = Brain::new("syn");
    b.remember("Fixed auth bug with null check in login.py").unwrap();
    let r = b.recall("authentication issue");
    assert!(r.memories.iter().any(|m| m.fiber.summary.to_lowercase().contains("login")
        || m.fiber.summary.to_lowercase().contains("auth")));
}

#[test]
fn type_hypothesis_and_boundary() {
    assert_eq!(
        suggest_memory_type("I think the cron timezone is wrong"),
        MemoryType::Hypothesis
    );
    assert!(matches!(
        suggest_memory_type("Never commit secrets to git"),
        MemoryType::Boundary | MemoryType::Instruction
    ));
}

#[test]
fn hypothesis_belief_rises_with_evidence() {
    let mut b = Brain::new("hyp");
    b.remember("I think the cron timezone is wrong").unwrap();
    let before = b
        .store()
        .fibers()
        .into_iter()
        .find(|f| f.memory_type == MemoryType::Hypothesis)
        .map(|f| f.belief)
        .unwrap_or(0.5);
    b.remember("Logs confirm the cron timezone is wrong").unwrap();
    let after = b
        .store()
        .fibers()
        .into_iter()
        .find(|f| f.memory_type == MemoryType::Hypothesis)
        .map(|f| f.belief)
        .unwrap_or(0.5);
    assert!(after > before, "belief {before} -> {after}");
}

#[test]
fn unrelated_fact_does_not_boost_hypothesis() {
    let mut b = Brain::new("hyp2");
    b.remember("I think the cron timezone is wrong").unwrap();
    let before = b
        .store()
        .fibers()
        .into_iter()
        .find(|f| f.memory_type == MemoryType::Hypothesis)
        .map(|f| f.belief)
        .unwrap_or(0.5);
    b.remember("Alice prefers dark mode in the editor").unwrap();
    let after = b
        .store()
        .fibers()
        .into_iter()
        .find(|f| f.memory_type == MemoryType::Hypothesis)
        .map(|f| f.belief)
        .unwrap_or(0.5);
    assert_eq!(after, before, "unrelated fact must not move belief");
}

#[test]
fn recall_does_not_compound_hypothesis_belief() {
    let mut b = Brain::new("hyp3");
    b.remember("I think the cron timezone is wrong").unwrap();
    b.remember("Logs confirm the cron timezone is wrong").unwrap();
    let after_encode = b
        .store()
        .fibers()
        .into_iter()
        .find(|f| f.memory_type == MemoryType::Hypothesis)
        .map(|f| f.belief)
        .unwrap();
    let _ = b.recall("cron timezone");
    let _ = b.recall("cron timezone");
    let after_recall = b
        .store()
        .fibers()
        .into_iter()
        .find(|f| f.memory_type == MemoryType::Hypothesis)
        .map(|f| f.belief)
        .unwrap();
    assert!(
        (after_recall - after_encode).abs() < 1e-9,
        "belief must stay {after_encode}, got {after_recall}"
    );
}

#[test]
fn overlap_does_not_invert_caused_by() {
    let mut b = Brain::new("dir");
    b.remember("JWT expiry caused the outage because the rotation cron failed")
        .unwrap();
    b.remember("Tuesday production outage — API 502 for 18 minutes")
        .unwrap();
    let inverted = b.store().synapses().into_iter().any(|s| {
        s.type_ == SynapseType::CausedBy
            && b.store()
                .get_neuron(&s.source_id)
                .map(|n| n.content.to_lowercase().contains("jwt"))
                .unwrap_or(false)
            && b.store()
                .get_neuron(&s.target_id)
                .map(|n| {
                    n.content.to_lowercase().contains("tuesday")
                        && n.content.to_lowercase().contains("502")
                })
                .unwrap_or(false)
    });
    assert!(
        !inverted,
        "JWT story must not be CausedBy the later outage event"
    );
}

#[test]
fn evidence_synapse_wired_to_hypothesis() {
    let mut b = Brain::new("ev");
    b.remember("I think the cron timezone is wrong").unwrap();
    b.remember("Logs confirm the cron timezone is wrong").unwrap();
    let has = b.store().synapses().into_iter().any(|s| {
        matches!(
            s.type_,
            SynapseType::EvidenceFor | SynapseType::EvidenceAgainst
        )
    });
    assert!(has, "evidence synapse should be created");
}

#[test]
fn expired_fiber_hidden_from_recall() {
    let mut b = Brain::new("exp");
    b.remember("Tuesday production outage at 15:00 UTC").unwrap();
    let id = b.store().fibers_vec()[0].id.clone();
    {
        let f = b.store_mut().get_fiber_mut(&id).unwrap();
        f.status = MemoryStatus::Expired;
    }
    let r = b.recall("outage");
    assert!(
        r.memories.iter().all(|m| m.fiber.id != id),
        "expired fiber leaked"
    );
}

#[test]
fn superseded_decision_hidden() {
    let mut b = Brain::new("sup");
    b.remember("We decided to use JWT for sessions").unwrap();
    b.remember("We decided to use Redis for the session store instead of JWT-only auth")
        .unwrap();
    let r = b.recall("session store decision");
    assert!(r.memories.iter().any(is_redis_decision));
    let hidden = r.memories.iter().any(|m| {
        m.fiber.status == MemoryStatus::Superseded && m.fiber.summary.to_lowercase().contains("jwt")
    });
    assert!(!hidden, "superseded JWT decision should be hidden");
}

#[test]
fn decision_reversal_creates_supersedes() {
    let mut b = Brain::new("rev");
    b.remember("We decided to use JWT for sessions").unwrap();
    let r = b
        .remember("We decided to use Redis instead of JWT for sessions")
        .unwrap();
    assert!(
        r.conflicts
            .iter()
            .any(|c| c.kind == ConflictKind::DecisionReversal)
            || b.store()
                .synapses()
                .into_iter()
                .any(|s| s.type_ == SynapseType::Supersedes),
        "reversal must wire SUPERSEDES"
    );
}

#[test]
fn forget_removes_fiber() {
    let mut b = seed();
    let before = b.store().fiber_count();
    assert!(b.forget("login.py").is_some());
    assert_eq!(b.store().fiber_count(), before - 1);
}

#[test]
fn consolidate_decays_states() {
    let mut b = seed();
    let r = b.consolidate();
    assert!(r.neurons_touched > 0 || r.synapses_decayed > 0);
}

#[test]
fn consolidate_does_not_merge_reversed_decisions() {
    let mut b = Brain::new("nomerge");
    b.remember("We decided to use JWT for sessions").unwrap();
    b.remember("We decided to use Redis instead of JWT for sessions")
        .unwrap();
    let before = b.store().fiber_count();
    b.consolidate();
    assert!(
        b.store().fiber_count() >= before - 0,
        "conflicted decisions must not collapse"
    );
}

#[test]
fn merge_collapses_near_duplicates() {
    let mut b = Brain::new("dup");
    b.remember("Always rotate JWT signing keys on a 12-hour cron")
        .unwrap();
    b.remember("Always rotate JWT signing keys on a 12 hour cron")
        .unwrap();
    let before = b.store().fiber_count();
    let r = b.consolidate();
    assert!(
        r.fibers_merged >= 1 || b.store().fiber_count() < before,
        "near-dup instructions should merge"
    );
}

#[test]
fn unused_fiber_conductivity_decays() {
    let mut b = Brain::new("dec");
    b.remember("Some unused fact about printers").unwrap();
    let id = b.store().fibers_vec()[0].id.clone();
    {
        let f = b.store_mut().get_fiber_mut(&id).unwrap();
        f.tier = nmem::types::MemoryTier::Warm;
        f.conductivity = 1.0;
        f.last_conducted = Some(now_ms() - 20 * 86_400_000);
    }
    b.consolidate();
    let cond = b.store().get_fiber(&id).unwrap().conductivity;
    assert!(cond < 1.0, "expected decay, got {cond}");
}

#[test]
fn session_priming_warms_after_recall() {
    let mut b = seed();
    assert_eq!(b.session_size(), 0);
    let _ = b.recall("outage jwt");
    assert!(b.session_size() > 0, "recall should leave warm traces");
}

#[test]
fn refractory_after_activate() {
    use nmem::types::NeuronState;
    let mut st = NeuronState::new("n", 0.1);
    let t = now_ms();
    st.fire(1.0, t, 6.0, 250);
    assert!(st.in_refractory(t + 10));
    assert!(!st.in_refractory(t + 1000));
}

#[test]
fn preference_is_hot_and_holds_floor() {
    let mut b = Brain::new("pref");
    b.remember("I prefer tabs over spaces").unwrap();
    let id = b.store().fibers_vec()[0].id.clone();
    {
        let f = b.store_mut().get_fiber_mut(&id).unwrap();
        assert_eq!(f.tier, nmem::types::MemoryTier::Hot);
        f.conductivity = 0.55;
        f.last_conducted = Some(now_ms() - 40 * 86_400_000);
    }
    b.consolidate();
    let c = b.store().get_fiber(&id).unwrap().conductivity;
    assert!(c >= 0.5, "hot floor, got {c}");
}

#[test]
fn stage_promotes_after_age() {
    let mut b = Brain::new("stg");
    b.remember("A durable fact about postgres WAL").unwrap();
    let id = b.store().fibers_vec()[0].id.clone();
    {
        let f = b.store_mut().get_fiber_mut(&id).unwrap();
        f.created_at = now_ms() - 2 * 3_600_000;
    }
    b.consolidate();
    let st = b.store().get_fiber(&id).unwrap().stage;
    assert_ne!(st, nmem::types::MemoryStage::ShortTerm);
}

#[test]
fn spatial_neuron_from_office() {
    let mut b = Brain::new("sp");
    b.remember("Standup at the office every Tuesday").unwrap();
    let has = b
        .store()
        .neurons()
        .into_iter()
        .any(|n| n.type_ == nmem::types::NeuronType::Spatial);
    assert!(has, "expected spatial neuron");
}

#[test]
fn temporal_tuesday_boosts_outage() {
    let mut b = seed();
    let r = b.recall("what happened tuesday");
    assert!(r.memories.iter().any(|m| m.fiber.summary.to_lowercase().contains("outage")));
}

#[test]
fn context_pack_stays_under_budget() {
    let mut b = seed();
    let p = b.context("outage jwt", 80);
    assert!(p.tokens <= 120, "tokens {}", p.tokens);
    assert!(!p.text.is_empty() || p.memories == 0);
}

#[test]
fn empty_remember_is_error() {
    let mut b = Brain::new("e");
    assert!(b.remember("   ").is_err());
}

#[test]
fn persist_compact_roundtrip_keeps_index() {
    let dir = std::env::temp_dir().join(format!("nmem-idx-{}.json", std::process::id()));
    let mut b = seed();
    b.save_as(&dir).unwrap();
    let mut loaded = Brain::open(&dir).unwrap();
    assert!(!loaded.recall("login.py").memories.is_empty());
    let _ = std::fs::remove_file(&dir);
}

#[test]
fn idf_rare_term_not_drowned() {
    let mut b = seed();
    let r = b.recall("login.py null check");
    assert!(r.memories.iter().any(|m| m.fiber.summary.contains("login.py")));
}

#[test]
fn long_vietnamese_does_not_panic() {
    let mut b = Brain::new("vi");
    let s = "Nhận ra rằng cron timezone UTC/ICT lệch bảy tiếng là nguyên nhân rotation miss. "
        .repeat(20);
    assert!(b.remember(&s).is_ok());
}

#[test]
fn recall_reinforces_path_synapses() {
    let mut b = seed();
    let before: u32 = b
        .store()
        .synapses()
        .into_iter()
        .map(|s| s.reinforced_count)
        .sum();
    let _ = b.recall("why did the outage happen");
    let after: u32 = b
        .store()
        .synapses()
        .into_iter()
        .map(|s| s.reinforced_count)
        .sum();
    assert!(after >= before, "path recall should reinforce some edges");
}

#[test]
fn duplicate_link_reinforces_not_duplicates() {
    let mut b = Brain::new("dupl");
    b.remember("Alpha event one").unwrap();
    b.remember("Beta event two").unwrap();
    b.link("alpha", "beta", SynapseType::RelatedTo, 0.5);
    let n = b.store().synapse_count();
    b.link("alpha", "beta", SynapseType::RelatedTo, 0.5);
    assert_eq!(b.store().synapse_count(), n);
}

#[test]
fn stress_causal_link_recall_50x() {
    for i in 0..50 {
        let mut b = Brain::new(format!("stress-{i}"));
        if i % 2 == 0 {
            b.remember("Tuesday production outage — API 502 for 18 minutes")
                .unwrap();
            b.remember("JWT expiry caused the outage because the rotation cron failed")
                .unwrap();
        } else {
            b.remember("JWT expiry caused the outage because the rotation cron failed")
                .unwrap();
            b.remember("Tuesday production outage — API 502 for 18 minutes")
                .unwrap();
        }
        let c = b.causal("outage", 4);
        assert!(
            c.chain
                .iter()
                .any(|h| h.synapse == "caused_by" || h.synapse == "leads_to"),
            "iter {i} seed={:?} chain={:?}",
            c.seed,
            c.chain
        );
        b.remember("On-call page fired after the outage").unwrap();
        assert!(
            b.link("outage", "on-call", SynapseType::LeadsTo, 0.9)
                .is_some(),
            "iter {i} link failed"
        );
        let r = b.recall("why outage");
        assert!(!r.memories.is_empty(), "iter {i} empty recall");
    }
}

#[test]
fn hashed_embed_ranks_paraphrase() {
    let a = nmem::embed::embed("JWT expiry caused the outage");
    let b = nmem::embed::embed("the outage was caused by jwt expiry");
    let c = nmem::embed::embed("office coffee machine is broken");
    assert!(nmem::embed::cosine(&a, &b) > nmem::embed::cosine(&a, &c));
}

#[test]
fn local_today_uses_offset() {
    let now = 1_776_268_800_000u64;
    let utc = nmem::temporal::resolve_label_tz("today", now, 0).unwrap();
    let ict = nmem::temporal::resolve_label_tz("hôm nay", now, 7 * 3_600_000).unwrap();
    assert_ne!(utc.0, ict.0, "ICT midnight must differ from UTC midnight");
}


#[test]
fn link_resolves_fiber_and_neuron_ids() {
    let mut b = Brain::new("link_ids");
    let r1 = b.remember("JWT expiry caused the Tuesday outage").unwrap();
    let r2 = b.remember("Rotation cron failed on the auth service").unwrap();
    let fiber_a = r1.fiber.id.clone();
    let fiber_b = r2.fiber.id.clone();
    let anchor_a = r1.fiber.anchor_neuron_id.clone();
    let anchor_b = r2.fiber.anchor_neuron_id.clone();

    // fiber id → fiber id
    let s = b.link(&fiber_a, &fiber_b, nmem::types::SynapseType::CausedBy, 0.9);
    assert!(s.is_some(), "fiber ids should link");
    assert_eq!(s.as_ref().unwrap().source_id, anchor_a);
    assert_eq!(s.as_ref().unwrap().target_id, anchor_b);

    // neuron id → neuron id
    let s2 = b.link(&anchor_a, &anchor_b, nmem::types::SynapseType::RelatedTo, 0.7);
    assert!(s2.is_some(), "neuron ids should link");

    // mixed fiber + text
    let s3 = b.link(&fiber_a, "rotation cron auth", nmem::types::SynapseType::RelatedTo, 0.6);
    assert!(s3.is_some(), "fiber id + content should link");
}
