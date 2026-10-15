"""Streamlit dashboard for the local NLPulse AI platform."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.database import Database
from src.ingestion import ExtractionError, extract_text
from src.pipeline import analyze_text
from src.search_engine import SearchEngine


ROOT = Path(__file__).resolve().parents[1]
DATABASE = Database(ROOT / "data" / "nlpulse.db")


def validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to analyze.")
    return text.strip()


def save_result(name: str, result: dict[str, Any], source: str = "dashboard") -> int:
    document_id = DATABASE.create_document(name, result["text"], source)
    DATABASE.save_analysis(document_id, result)
    for entity in result["entities"]:
        DATABASE.save_entity(document_id, entity["text"], entity["type"])
    return document_id


def render_result(result: dict[str, Any], st: Any) -> None:
    classification, urgency, sentiment = result["classification"], result["urgency"], result["sentiment"]
    columns = st.columns(3)
    columns[0].metric("Classification", classification["label"], f"{classification['confidence']:.0%}")
    columns[1].metric("Urgency", urgency["level"], f"score {urgency['score']}")
    columns[2].metric("Sentiment", sentiment["label"], f"score {sentiment['score']:.2f}")
    st.subheader("Summary")
    st.write(result["summary"] or "No summary available.")
    left, right = st.columns(2)
    with left:
        st.subheader("Entities")
        st.dataframe(result["entities"], use_container_width=True)
    with right:
        st.subheader("Keywords")
        st.write(", ".join(result["keywords"]) or "No keywords found.")
    with st.expander("Original text and details"):
        st.text_area("Normalized text", result["text"], height=150, disabled=True)
        st.json(result)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="NLPulse AI", page_icon="🧠", layout="wide")
    st.title("NLPulse AI")
    st.caption("Intelligent document and email understanding platform")
    analyze_tab, search_tab, history_tab = st.tabs(["Analyze", "Search", "History"])

    with analyze_tab:
        mode = st.radio("Input type", ["Paste text", "Upload document"], horizontal=True)
        text = ""
        name = "pasted-text"
        if mode == "Paste text":
            name = st.text_input("Document name", value="pasted-text")
            text = st.text_area("Email or document text", height=220, placeholder="Paste text here...")
        else:
            uploaded = st.file_uploader("Upload TXT, CSV, PDF, or DOCX", type=["txt", "csv", "pdf", "docx"])
            if uploaded:
                name = uploaded.name
                temporary = ROOT / "data" / ".dashboard_upload"
                temporary.write_bytes(uploaded.getvalue())
                try:
                    text = extract_text(temporary)
                except ExtractionError as exc:
                    st.error(str(exc))
                finally:
                    temporary.unlink(missing_ok=True)
                st.text_area("Extracted text", text, height=160, disabled=True)
        if st.button("Analyze and save", type="primary"):
            try:
                result = analyze_text(validate_text(text))
                document_id = save_result(name or "pasted-text", result)
                st.success(f"Saved document #{document_id}")
                render_result(result, st)
            except (ValueError, ExtractionError) as exc:
                st.error(str(exc))

    with search_tab:
        query = st.text_input("Search similar documents", placeholder="invoice overdue customer")
        if st.button("Search") and query.strip():
            try:
                engine = SearchEngine(DATABASE.list_documents())
                results = engine.search(query, limit=20)
                if results:
                    st.dataframe(results, use_container_width=True)
                else:
                    st.info("No similar documents found.")
            except RuntimeError as exc:
                st.error(str(exc))

    with history_tab:
        records = DATABASE.list_documents(limit=100)
        st.metric("Stored documents", len(records))
        if records:
            st.dataframe(records, use_container_width=True)
        else:
            st.info("Analyze a document to create history.")


if __name__ == "__main__":
    main()
