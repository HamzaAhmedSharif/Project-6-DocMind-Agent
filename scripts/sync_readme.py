"""Keeps the results block in README.md identical to the tables in evaluation/RESULTS.md.

Usage: python scripts/sync_readme.py          # rewrite the block
       python scripts/sync_readme.py --check  # exit 1 if the README is out of date
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "evaluation" / "RESULTS.md"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def extract_block(results_md: str) -> str:
    """The answer-quality and cost tables of RESULTS.md, with headings demoted for the README."""
    text = results_md.replace("\r\n", "\n")
    start = text.index("## Answer quality")
    end = text.index("`api_seconds` is time spent")
    block = text[start:end].strip()
    block = block.replace("(see the Paraphrase review section)", "(details in evaluation/RESULTS.md)")
    return block.replace("\n## ", "\n#### ").replace("## Answer quality", "#### Answer quality", 1)


def sync(readme_text: str, block: str) -> str:
    text = readme_text.replace("\r\n", "\n")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block}\n{END}{tail}"


def main(argv) -> int:
    results = RESULTS.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    updated = sync(readme, extract_block(results))
    if "--check" in argv:
        if updated != readme.replace("\r\n", "\n"):
            print("README results block is out of date; run: python scripts/sync_readme.py")
            return 1
        print("README results block is up to date")
        return 0
    README.write_text(updated, encoding="utf-8", newline="\n")
    print("README results block updated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
