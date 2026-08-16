use clap::{Parser, Subcommand};
use nmem::types::{MemoryType, SynapseType};
use nmem::{Brain, RecallOpts, Store};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nmem", about = "Neural Memory — spreading-activation brain (Rust)")]
struct Cli {
    #[arg(global = true, short, long)]
    brain: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Remember {
        content: String,
        #[arg(long)]
        r#type: Option<String>,
        #[arg(long)]
        tag: Vec<String>,
    },
    Recall {
        query: String,
        #[arg(long, default_value_t = 4)]
        depth: u32,
        #[arg(long, default_value_t = 8)]
        limit: usize,
    },
    /// What caused this? (BFS on CAUSED_BY)
    Causal {
        query: String,
        #[arg(long, default_value_t = 6)]
        hops: u32,
    },
    /// What did this lead to? (BFS on LEADS_TO)
    Effects {
        query: String,
        #[arg(long, default_value_t = 6)]
        hops: u32,
    },
    /// Manually wire two memories
    Link {
        from: String,
        to: String,
        #[arg(long, default_value = "caused_by")]
        r#type: String,
        #[arg(long, default_value_t = 0.85)]
        weight: f64,
    },
    /// Delete a fiber by id or query
    Forget { query: String },
    Health,
    List {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    Consolidate,
    Export {
        #[arg(short, long)]
        output: PathBuf,
    },
    Import { input: PathBuf },
    Stats,
    /// Pack memories into a token-budgeted prompt (no LLM, just text)
    Context {
        query: String,
        #[arg(long, default_value_t = 400)]
        tokens: usize,
    },
    /// MCP stdio server for OpenClaw / Claude Code (JSON-RPC NDJSON)
    Mcp,
    /// Local web dashboard (no Node)
    Dash {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Show detected local timezone
    Tz,
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
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
        "before" => SynapseType::Before,
        "after" => SynapseType::After,
        _ => SynapseType::RelatedTo,
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let path = cli
        .brain
        .unwrap_or_else(nmem::mcp::default_brain_path);
    if matches!(cli.cmd, Cmd::Mcp) {
        return nmem::mcp::run_stdio(path);
    }
    if let Cmd::Dash { host, port } = &cli.cmd {
        return nmem::dashboard::serve_path(path, host, *port);
    }
    if matches!(cli.cmd, Cmd::Tz) {
        let tz = nmem::temporal::tz_info();
        println!(
            "tz={} source={} offset_hours={:+} now={}",
            tz.label,
            tz.source,
            tz.offset_hours,
            nmem::temporal::format_local(nmem::types::now_ms())
        );
        return Ok(());
    }
    let mut brain = Brain::open(&path)?;
    match cli.cmd {
        Cmd::Remember {
            content,
            r#type,
            tag,
        } => {
            let ty = r#type.as_deref().and_then(MemoryType::parse);
            let r = brain.remember_typed(&content, ty, tag, 5)?;
            brain.save()?;
            println!(
                "stored {} n+{} s+{} fiber={}",
                r.fiber.memory_type.as_str(),
                r.neurons_created.len(),
                r.synapses_created.len(),
                r.fiber.summary
            );
            for c in r.conflicts {
                println!(
                    "  ! {:?} vs: {}",
                    c.kind,
                    c.existing_content.chars().take(80).collect::<String>()
                );
            }
        }
        Cmd::Recall {
            query,
            depth,
            limit,
        } => {
            let r = brain.recall_opts(
                &query,
                RecallOpts {
                    depth: Some(depth),
                    limit,
                    ..Default::default()
                },
            );
            brain.save()?;
            println!(
                "recall \"{}\"  {} hits  {} activated  {}ms",
                r.query,
                r.memories.len(),
                r.activations.len(),
                r.elapsed_ms
            );
            if r.trace.stopped_early {
                println!("  diminishing-returns: {}", r.trace.stop_reason);
            }
            for (i, m) in r.memories.iter().enumerate() {
                println!(
                    "  {}. [{:.3} conf={:.2}] hop={} {}  {}",
                    i + 1,
                    m.score,
                    m.confidence,
                    m.hop,
                    m.fiber.memory_type.as_str(),
                    m.fiber.summary
                );
                println!("      {}", m.reason);
            }
        }
        Cmd::Causal { query, hops } => {
            let r = brain.causes(&query, hops);
            match r.seed {
                None => println!("no seed for \"{}\"", r.query),
                Some(s) => {
                    println!("causes of: {s}  conf={:.2}", r.confidence);
                    for hop in r.chain {
                        println!(
                            "  [{}] --{}--> {}",
                            hop.depth, hop.synapse, hop.to_content
                        );
                    }
                }
            }
        }
        Cmd::Effects { query, hops } => {
            let r = brain.effects(&query, hops);
            match r.seed {
                None => println!("no seed for \"{}\"", r.query),
                Some(s) => {
                    println!("effects of: {s}  conf={:.2}", r.confidence);
                    for hop in r.chain {
                        println!(
                            "  [{}] --{}--> {}",
                            hop.depth, hop.synapse, hop.to_content
                        );
                    }
                }
            }
        }
        Cmd::Link {
            from,
            to,
            r#type,
            weight,
        } => {
            match brain.link(&from, &to, parse_syn(&r#type), weight) {
                Some(s) => println!("linked {} --{}--> {}", s.source_id, s.type_.as_str(), s.target_id),
                None => println!("could not resolve both endpoints (need fiber id, neuron id, or matching text)"),
            }
            brain.save()?;
        }
        Cmd::Forget { query } => {
            match brain.forget(&query) {
                Some(id) => println!("forgot {id}"),
                None => println!("nothing matched"),
            }
            brain.save()?;
        }
        Cmd::Health => {
            let h = brain.health();
            println!("health {}/100  grade {}", h.score, h.grade);
            println!(
                "  neurons={} synapses={} fibers={} orphans={} density={:.4}",
                h.neurons, h.synapses, h.fibers, h.orphans, h.density
            );
            for i in h.issues {
                println!("  ! {i}");
            }
        }
        Cmd::List { limit } => {
            let mut fs = brain.store().fibers_vec();
            fs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            for (i, f) in fs.into_iter().take(limit).enumerate() {
                println!(
                    "  {}. [{}] {}  id={}",
                    i + 1,
                    f.memory_type.as_str(),
                    f.summary,
                    f.id
                );
            }
        }
        Cmd::Consolidate => {
            let r = brain.consolidate();
            brain.save()?;
            println!(
                "consolidated decayed={} pruned_syn={} merged={} removed_fib={} orphans={}",
                r.synapses_decayed,
                r.synapses_pruned,
                r.fibers_merged,
                r.fibers_removed,
                r.neurons_pruned
            );
        }
        Cmd::Export { output } => {
            std::fs::write(output, brain.export_json()?)?;
        }
        Cmd::Import { input } => {
            let raw = std::fs::read_to_string(input)?;
            let snap = serde_json::from_str(&raw)?;
            brain.import_snapshot(snap);
            brain.save()?;
            println!("imported");
        }
        Cmd::Stats => {
            let s = brain.store();
            println!(
                "neurons={} synapses={} fibers={}",
                s.neuron_count(),
                s.synapse_count(),
                s.fiber_count()
            );
        }
        Cmd::Context { query, tokens } => {
            let pack = brain.context(&query, tokens);
            println!(
                "# context q={:?} tokens~{} memories={}",
                pack.query, pack.tokens, pack.memories
            );
            print!("{}", pack.text);
        }
        Cmd::Mcp | Cmd::Dash { .. } | Cmd::Tz => unreachable!("handled before match"),
    }
    Ok(())
}
