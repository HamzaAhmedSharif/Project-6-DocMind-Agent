"""DocMind Agent: Streamlit front end over the evaluated pipelines.

Index a document first:  python agents/ingest.py   (PDFs in data/)
                         or upload a PDF in the app itself.
Run the app:             streamlit run app.py
"""
import os
import sys
import time

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="DocMind Agent", layout="wide")

import config
from agents.llm import STATS, reset_stats

MODES = {
    "Multi-agent (route, retrieve, write, review)": "multi",
    "Single call (writer prompt)": "single_writer",
    "Baseline (M2 prompt)": "baseline",
}
EXAMPLES = [
    "What is self-attention?",
    "Why are the dot products scaled by the square root of d_k?",
    "What are the latest developments in transformer architectures in 2026?",
]
RESULTS_PATH = os.path.join("evaluation", "RESULTS.md")


@st.cache_resource(show_spinner="Loading the retrieval index and models...")
def load_pipelines():
    from agents.graph import run_query
    from agents.rag_chain import answer_question
    from agents.researcher import retriever
    from agents.writer import write_answer
    return retriever, run_query, answer_question, write_answer


def set_question(text):
    st.session_state["question"] = text


def run_selected(mode, question):
    retriever, run_query, answer_question, write_answer = load_pipelines()
    reset_stats()
    started = time.time()
    if mode == "baseline":
        result = answer_question(question, retriever)
        out = {"answer": result["answer"], "route": None, "doc_sources": sorted(result["sources"]),
               "web_urls": [], "revisions": None}
    elif mode == "single_writer":
        chunks = retriever.retrieve(question)
        state = {"query": question, "doc_context": chunks, "web_context": [], "review_feedback": ""}
        out = {"answer": write_answer(state)["draft_answer"], "route": None,
               "doc_sources": sorted({c["source"] for c in chunks}), "web_urls": [], "revisions": None}
    else:
        state = run_query(question)
        out = {
            "answer": state["final_answer"],
            "route": "web search + document" if state.get("needs_web_search") else "document only",
            "doc_sources": sorted({c["source"] for c in state.get("doc_context", [])}),
            "web_urls": [w["url"] for w in state.get("web_context", [])],
            "revisions": state.get("revision_count", 0),
        }
    out["stats"] = dict(STATS)
    out["seconds"] = time.time() - started
    return out



def index_uploaded_pdf(uploaded) -> int:
    """Save an uploaded PDF into data/ and rebuild the index from it alone.

    Returns the chunk count. Raises ValueError (with the existing index
    untouched) for PDFs without a usable text layer.
    """
    from agents.ingest import build_index_for_file

    os.makedirs(config.DATA_DIR, exist_ok=True)
    safe_name = os.path.basename(uploaded.name).replace("/", "_").replace("\\", "_")
    target = os.path.join(config.DATA_DIR, safe_name)
    with open(target, "wb") as f:
        f.write(uploaded.getbuffer())
    chunks = build_index_for_file(target)
    load_pipelines()[0].reload()  # the cached retriever must see the new index
    return chunks

st.title("DocMind Agent")
st.caption(
    "Question answering over an indexed PDF. In multi-agent mode a supervisor routes each question, "
    "a researcher retrieves from the document (and the web when needed), a writer drafts, and the supervisor reviews."
)

with st.sidebar:
    st.header("Pipeline")
    mode_label = st.radio("Mode", list(MODES), label_visibility="collapsed")
    st.caption(f"Backend: {config.LLM_PROVIDER} / {config.LLM_MODEL}")
    st.caption(f"Retrieval: {config.CHUNK_SIZE}-word chunks, top-{config.TOP_K_RESULTS}")
    st.caption("Free-tier rate limits can add waiting time; it is reported next to each answer.")

uploaded = st.file_uploader(
    "Upload a PDF to ask questions about it",
    type="pdf",
    help="The PDF is indexed on the spot. Text-based PDFs only: scanned images have no text layer to read. "
         "On this hosted demo the uploaded document replaces the current one for everyone.",
)
if uploaded is not None:
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.get("indexed_upload") != file_id:
        try:
            with st.spinner(f"Indexing {uploaded.name}..."):
                chunk_count = index_uploaded_pdf(uploaded)
        except ValueError as e:
            st.error(str(e))
        except Exception as e:  # corrupt file, parser failure, ...
            st.error(f"Could not index that PDF: {type(e).__name__}: {e}")
        else:
            st.session_state["indexed_upload"] = file_id
            st.success(f"Indexed {chunk_count} chunks from {uploaded.name}. Ask a question below.")
    else:
        st.caption(f"Current document: {uploaded.name} (indexed from your upload)")

if not os.path.exists(f"{config.INDEX_PATH}.bin"):
    st.error("No index found. Upload a PDF above to index it, or put a PDF in `data/` and run `python agents/ingest.py`, then reload this page.")
    st.stop()
load_pipelines()

question = st.text_area("Question", key="question", placeholder="Ask about the indexed document...", height=90)
example_cols = st.columns(len(EXAMPLES))
for col, example in zip(example_cols, EXAMPLES):
    col.button(example, key=f"example_{example}", on_click=set_question, args=(example,))

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Working..."):
            out = run_selected(MODES[mode_label], question.strip())
    except Exception as e:  # show the provider or limit message instead of a traceback
        st.error(f"{type(e).__name__}: {e}")
        st.stop()

    st.subheader("Answer")
    st.markdown(out["answer"])

    left, right = st.columns(2)
    with left:
        st.subheader("How it was answered")
        if out["route"]:
            st.markdown(f"**Routing:** {out['route']}")
        if out["revisions"] is not None:
            st.markdown(f"**Revisions requested by the reviewer:** {out['revisions']}")
        st.markdown("**Document sources:** " + (", ".join(out["doc_sources"]) or "none"))
        if out["web_urls"]:
            st.markdown("**Web sources:**")
            for url in out["web_urls"]:
                st.markdown(f"- {url}")
    with right:
        st.subheader("Cost")
        s = out["stats"]
        st.metric("LLM calls", s["calls"])
        st.metric("Prompt tokens", s["prompt_tokens"])
        st.metric("API seconds", f"{s['api_seconds']:.1f}")
        st.metric("Waiting for the rate limit (s)", f"{s['throttle_seconds']:.1f}")

if os.path.exists(RESULTS_PATH):
    with st.expander("Comparison results (evaluation/RESULTS.md)"):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            st.markdown(f.read())
