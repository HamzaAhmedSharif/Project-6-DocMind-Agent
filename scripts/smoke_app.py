"""Headless smoke test for app.py using Streamlit's AppTest. Makes ONE LLM call (baseline mode)."""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest


def main() -> int:
    at = AppTest.from_file("app.py", default_timeout=240)
    at.run()
    assert not at.exception, f"app raised on load: {[e.value for e in at.exception]}"
    print("loaded:", len(at.radio), "radio,", len(at.text_area), "text area,", len(at.button), "buttons")

    at.radio[0].set_value("Baseline (M2 prompt)")
    at.text_area[0].set_value("What is self-attention?")
    at.run()
    next(b for b in at.button if b.label == "Ask").click()
    at.run()
    assert not at.exception, f"app raised on Ask: {[e.value for e in at.exception]}"
    print("subheaders:", [s.value for s in at.subheader])
    print("metrics:", {m.label: m.value for m in at.metric})
    assert any(s.value == "Answer" for s in at.subheader), "no Answer section rendered"
    assert at.metric[0].value == "1", "baseline mode should use exactly one LLM call"
    print("APP SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
