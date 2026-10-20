import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_sync():
    spec = importlib.util.spec_from_file_location("sync_readme", ROOT / "scripts" / "sync_readme.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_results_block_matches_results_md():
    sync = load_sync()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = (ROOT / "evaluation" / "RESULTS.md").read_text(encoding="utf-8")
    assert sync.sync(readme, sync.extract_block(results)) == readme.replace("\r\n", "\n")


def test_extract_block_demotes_headings_and_keeps_the_tables():
    sync = load_sync()
    results = (
        "# T\n\nintro\n\n## Answer quality\n\n| a |\n|---|\n| 1 |\n\n"
        "## Cost per question (mean): x\n\n| b |\n\n`api_seconds` is time spent\n\n## Per-question detail\n"
    )
    block = sync.extract_block(results)
    assert block.startswith("#### Answer quality")
    assert "#### Cost per question (mean): x" in block
    assert "api_seconds" not in block and "Per-question" not in block
