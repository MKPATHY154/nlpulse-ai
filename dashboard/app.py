"""Streamlit dashboard for the local NLPulse AI platform."""

from __future__ import annotations

import csv
import io
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


def documents_csv(records: list[dict[str, Any]]) -> str:
    """Create a safe, correctly quoted CSV export for the dashboard."""
    output = io.StringIO(newline="")
    fields = ("id", "name", "category", "sentiment", "urgency", "created_at")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: record.get(field, "") for field in fields} for record in records)
    return output.getvalue()


def _chips(values: list[Any], st: Any, prefix: str = "") -> None:
    if not values:
        st.caption("Nothing detected yet")
        return
    labels = [value if isinstance(value, str) else value.get("text", "") for value in values]
    st.markdown(" ".join(f"`{label}`" for label in labels if label))


def render_result(result: dict[str, Any], st: Any) -> None:
    classification, urgency, sentiment = result["classification"], result["urgency"], result["sentiment"]
    st.markdown("### Analysis ready")
    columns = st.columns(3)
    columns[0].metric("Classification", classification["label"], f"{classification['confidence']:.0%} confidence")
    columns[1].metric("Urgency", urgency["level"], f"score {urgency['score']}")
    columns[2].metric("Sentiment", sentiment["label"], f"score {sentiment['score']:.2f}")
    result_tab, signals_tab, source_tab = st.tabs(["Overview", "Signals", "Source & details"])
    with result_tab:
        st.markdown("#### Executive summary")
        st.info(result["summary"] or "No summary available.")
        st.progress(min(1.0, max(0.0, float(classification["confidence"]))), text="Classification confidence")
    with signals_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Entities")
            if result["entities"]:
                st.dataframe(result["entities"], use_container_width=True, hide_index=True)
            else:
                st.caption("No named entities found in this text.")
        with right:
            st.markdown("#### Keywords")
            _chips(result["keywords"], st)
            st.markdown("#### Matched urgency terms")
            _chips(urgency.get("matched_terms", []), st)
    with source_tab:
        with st.expander("View normalized text", expanded=True):
            st.text_area("Normalized text", result["text"], height=150, disabled=True, label_visibility="collapsed")
        with st.expander("View complete result JSON"):
            st.json(result)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="NLPulse AI", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
    .block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1400px}.hero{padding:1.5rem 0 1rem}.hero h1{font-size:3.2rem;letter-spacing:-.06em;margin-bottom:.25rem;background:linear-gradient(90deg,#8d7cff,#50d8d0);-webkit-background-clip:text;color:transparent}.hero p{color:#8490ad;font-size:1.05rem}.stMetric{background:linear-gradient(135deg,#161d38,#11172c);border:1px solid #ffffff14;padding:1rem;border-radius:14px}.stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{padding:10px 16px;border-radius:9px}.empty{padding:3.5rem 1rem;text-align:center;border:1px dashed #ffffff25;border-radius:16px;color:#8792ad}.sample{font-size:.8rem;color:#8490ad}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## ✦ NLPulse AI")
        st.caption("Local intelligence workspace")
        st.divider()
        st.markdown("**Quick start**")
        st.caption("Choose a sample to populate the analyzer, or bring your own document.")
        samples = {
            "Overdue invoice": "Urgent: the invoice is overdue and payment must be completed before Friday.",
            "Happy customer": "Thanks for the excellent support. The team solved my issue quickly and kindly.",
            "Support request": "My account is locked and I cannot access any of my files. Please help.",
        }
        for label, sample in samples.items():
            if st.button(label, use_container_width=True):
                st.session_state["sample_text"] = sample
        st.divider()
        st.markdown("**Pipeline**")
        st.success("Analysis engine online")
        st.caption("Results are processed locally and saved to the document history when you analyze.")
    st.markdown('<div class="hero"><h1>Make every message meaningful.</h1><p>Transform unstructured text into clear, actionable intelligence.</p></div>', unsafe_allow_html=True)
    analyze_tab, search_tab, history_tab, analytics_tab = st.tabs(["✦ Analyze", "⌕ Search", "◷ History", "▥ Analytics"])
    with analyze_tab:
        mode = st.radio("Input type", ["Paste text", "Upload document"], horizontal=True)
        text, name = st.session_state.pop("sample_text", ""), "pasted-text"
        if mode == "Paste text":
            name = st.text_input("Document name", value="pasted-text")
            text = st.text_area("Email or document text", value=text, height=220, placeholder="Paste an email, support request, or document excerpt here…")
            st.caption("Tip: try one of the sample prompts in the sidebar.")
        else:
            uploaded = st.file_uploader("Upload TXT, CSV, PDF, or DOCX", type=["txt", "csv", "pdf", "docx"])
            if uploaded:
                name = uploaded.name
                temporary = ROOT / "data" / ".dashboard_upload"
                temporary.write_bytes(uploaded.getvalue())
                try:
                    text = extract_text(temporary)
                except ExtractionError as exc:
                    st.error(f"We couldn't read that file: {exc}")
                finally:
                    temporary.unlink(missing_ok=True)
                st.text_area("Extracted text", text, height=160, disabled=True)
            else:
                st.markdown('<div class="empty">Drop a document here to extract and analyze its text.</div>', unsafe_allow_html=True)
        if st.button("Analyze and save  →", type="primary", use_container_width=True):
            try:
                clean_text = validate_text(text)
                with st.spinner("Finding the signals in your text…"):
                    result = analyze_text(clean_text)
                    document_id = save_result(name or "pasted-text", result)
                st.success(f"Analysis saved to history as document #{document_id}.")
                render_result(result, st)
            except (ValueError, ExtractionError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Analysis unavailable right now: {exc}")
    with search_tab:
        st.markdown("### Find related documents")
        st.caption("Search your saved analyses by meaning, topic, or phrase.")
        query = st.text_input("Search query", placeholder="invoice overdue customer")
        if st.button("Search documents", type="primary"):
            if not query.strip():
                st.warning("Enter a search phrase first.")
            else:
                try:
                    with st.spinner("Searching your document intelligence…"):
                        results = SearchEngine(DATABASE.list_documents()).search(query, limit=20)
                    if results:
                        st.dataframe(results, use_container_width=True, hide_index=True)
                    else:
                        st.info("No similar documents found. Try a broader phrase.")
                except RuntimeError as exc:
                    st.error(str(exc))
    with history_tab:
        st.markdown("### Filter your document history")
        filter_left, filter_mid, filter_right = st.columns(3)
        with filter_left:
            category_filter = st.selectbox("Category", ["All", "complaint", "request", "question", "feedback", "other"])
        with filter_mid:
            sentiment_filter = st.selectbox("Sentiment", ["All", "positive", "negative", "neutral"])
        with filter_right:
            urgency_filter = st.selectbox("Urgency", ["All", "low", "medium", "high", "critical"])
        records = DATABASE.list_documents(
            limit=100,
            category=None if category_filter == "All" else category_filter,
            sentiment=None if sentiment_filter == "All" else sentiment_filter,
            urgency=None if urgency_filter == "All" else urgency_filter,
        )
        st.markdown("### Your analysis history")
        st.metric("Stored documents", len(records), help="Documents saved from the Analyze tab")
        if records:
            st.dataframe(records, use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered CSV",
                data=documents_csv(records),
                file_name="nlpulse-documents.csv",
                mime="text/csv",
            )
        else:
            st.markdown('<div class="empty">Your saved analyses will appear here.<br>Run your first analysis to start building history.</div>', unsafe_allow_html=True)

    with analytics_tab:
        st.markdown("### Intelligence overview")
        overview = DATABASE.analytics_overview(recent_limit=5)
        metric_columns = st.columns(4)
        metric_columns[0].metric("Documents", overview["document_count"])
        metric_columns[1].metric("Urgent", sum(value for key, value in overview["urgency_distribution"].items() if key in {"high", "critical"}))
        metric_columns[2].metric("Negative", overview["sentiment_distribution"].get("negative", 0))
        metric_columns[3].metric("Categories", len(overview["category_distribution"]))
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("#### Category distribution")
            st.bar_chart(overview["category_distribution"] or {"none": 0})
        with chart_right:
            st.markdown("#### Urgency distribution")
            st.bar_chart(overview["urgency_distribution"] or {"none": 0})
        st.markdown("#### Recent documents")
        if overview["recent_documents"]:
            st.dataframe(overview["recent_documents"], use_container_width=True, hide_index=True)
        else:
            st.info("Analyze a document to populate analytics.")


if __name__ == "__main__":
    main()
