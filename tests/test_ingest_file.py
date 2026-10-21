"""Offline tests for the upload path: build_index_for_file + Retriever.reload.

A hand-built minimal PDF and a fake embedding model keep these tests free of
network, API keys and model downloads, like the rest of the suite.
"""
import pickle

import pytest

# agents.ingest / agents.retriever pull in the heavy document stack (pypdf,
# faiss, sentence-transformers, numpy). Those live in requirements.txt, not
# in the deliberately light requirements-ci.txt, so on CI this module skips;
# in the full dev environment every test below runs.
pytest.importorskip("pypdf")
pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")
pytest.importorskip("numpy")

import numpy as np

import agents.ingest as ingest
import agents.retriever as retriever_mod


def make_pdf(text_pages):
    """Build a minimal valid PDF. text_pages: one string per page ('' = no text)."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(text_pages)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(text_pages)} >>".encode())
    font_obj = 3 + 2 * len(text_pages)
    for i, text in enumerate(text_pages):
        page_obj = 3 + 2 * i
        content_obj = page_obj + 1
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_obj} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>".encode()
        )
        if text:
            stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        else:
            stream = b""
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 8), dtype=np.float32)


TEXT = " ".join(["the transformer model uses self attention over the input sequence"] * 10)  # 110 words


@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "SentenceTransformer", lambda name: FakeEmbedder())
    monkeypatch.setattr(ingest, "INDEX_PATH", str(tmp_path / "idx"))
    monkeypatch.setattr(retriever_mod, "SentenceTransformer", lambda name: FakeEmbedder())
    monkeypatch.setattr(retriever_mod, "INDEX_PATH", str(tmp_path / "idx"))
    return tmp_path


def test_build_index_for_file_writes_index_and_meta(fake_models, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf([TEXT]))

    chunks = ingest.build_index_for_file(str(pdf))

    assert chunks == 1
    assert (tmp_path / "idx.bin").exists()
    with open(tmp_path / "idx_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    assert meta["sources"] == ["paper.pdf"]
    assert "self attention" in meta["chunks"][0]


def test_build_index_for_file_rejects_scanned_pdf(fake_models, tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(make_pdf(["", "", ""]))  # pages exist, but none has a text layer

    with pytest.raises(ValueError, match="scanned"):
        ingest.build_index_for_file(str(pdf))

    assert not (tmp_path / "idx.bin").exists()  # existing index (if any) stays untouched


def test_retriever_reload_picks_up_rebuilt_index(fake_models, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf([" ".join(["alpha bravo charlie delta echo foxtrot"] * 20)]))
    ingest.build_index_for_file(str(pdf))
    retriever = retriever_mod.Retriever()
    assert "alpha" in retriever.chunks[0]

    pdf.write_bytes(make_pdf([" ".join(["golf hotel india juliet kilo lima"] * 20)]))
    ingest.build_index_for_file(str(pdf))
    retriever.reload()

    assert "golf" in retriever.chunks[0]
    assert "alpha" not in retriever.chunks[0]
