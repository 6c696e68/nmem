"""OpenClaw compatibility: python -m neural_memory.mcp → nmem mcp."""

from __future__ import annotations

import os
import shutil
import sys


def main() -> None:
    nmem = os.environ.get("NMEM_BIN") or shutil.which("nmem")
    if not nmem:
        sys.stderr.write(
            "nmem not on PATH. Build the Rust binary and put it on PATH, "
            "or set NMEM_BIN.\n"
        )
        sys.exit(127)
    os.execvp(nmem, [nmem, "mcp"])


if __name__ == "__main__":
    main()
