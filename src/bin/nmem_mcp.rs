//! Alias binary so configs can run `nmem-mcp` like the Python package.

fn main() {
    let path = nmem::mcp::default_brain_path();
    if let Err(e) = nmem::mcp::run_stdio(path) {
        eprintln!("nmem-mcp: {e}");
        std::process::exit(1);
    }
}
