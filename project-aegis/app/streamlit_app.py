"""
app/streamlit_app.py

Streamlit frontend for Project Aegis.

Pages:
  1.  Ask a Question    — full RAG query with answer and source display
  2.  Upload Document   — ingest new policy documents via the API
  3.  Pipeline Debug    — inspect query expansion, HyDE, reranking step-by-step
  4.  Evaluation        — RAGAS metrics overview and sample QA runner

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Project Aegis – Policy Assistant",
    page_icon="🛡️",
    layout="wide",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get(path: str, **kw) -> dict:
    return requests.get(f"{API}{path}", timeout=5, **kw).json()

def _post(path: str, **kw) -> dict:
    r = requests.post(f"{API}{path}", timeout=120, **kw)
    r.raise_for_status()
    return r.json()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ Project Aegis")
    st.markdown("*Advanced Enterprise RAG*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["💬 Ask a Question", "📄 Upload Document", "🔍 Pipeline Debug", "📊 Evaluation"],
    )

    st.divider()
    # API health badge
    try:
        health = _get("/health")
        info   = health.get("collection", {})
        st.success(f"✅ API connected")
        st.caption(f"{info.get('points_count', '?')} chunks indexed")
    except Exception:
        st.error("❌ API unavailable")
        st.caption(f"Expected at {API}")

    st.divider()
    st.caption("IIT Madras · Adv. Certificate in AI\nModule 9 · Project Aegis")


# ── Page 1 — Ask a Question ──────────────────────────────────────────────────

if page == "💬 Ask a Question":
    st.title("💬 Ask a Policy Question")
    st.markdown(
        "Ask any question about your organisation's policies. "
        "The system searches all indexed documents and returns a grounded, cited answer."
    )

    with st.form("query_form"):
        query = st.text_area(
            "Your question",
            placeholder="e.g. Can I claim Uber expenses during international travel?",
            height=90,
        )
        col1, col2, col3 = st.columns(3)
        use_expansion   = col1.checkbox("Query Expansion", value=True,
                                        help="Generate 4 semantically equivalent variants of your query")
        use_hyde        = col2.checkbox("HyDE", value=True,
                                        help="Embed a hypothetical policy answer for better retrieval")
        filter_category = col3.checkbox("Category Filter", value=True,
                                        help="Auto-detect policy category and pre-filter the search space")
        top_k = st.slider("Chunks to use for answer generation", 1, 10, 5)
        submitted = st.form_submit_button("🔍 Search", type="primary")

    if submitted and query.strip():
        with st.spinner("Running retrieval pipeline..."):
            try:
                result = _post("/query", json={
                    "query":           query,
                    "use_expansion":   use_expansion,
                    "use_hyde":        use_hyde,
                    "filter_category": filter_category,
                    "top_k_retrieval": 25,
                    "top_k_rerank":    top_k,
                })

                st.markdown("### 📋 Answer")
                st.info(result["answer"])

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Chunks retrieved",      result["chunks_retrieved"])
                col_b.metric("After reranking",        result["chunks_after_rerank"])
                col_c.metric("Model",                  result["model"])

                if result.get("sources"):
                    with st.expander("📎 Source documents used", expanded=False):
                        for i, src in enumerate(result["sources"], 1):
                            doc   = src.get("document_id", "—")
                            cat   = src.get("policy_category", "")
                            h1    = src.get("h1_header", "")
                            h2    = src.get("h2_header", "")
                            eff   = src.get("effective_date", "")
                            st.markdown(
                                f"**{i}.** `{doc}` · **{cat}** · {h1} > {h2}"
                                + (f"  *(effective {eff})*" if eff else "")
                            )

            except requests.HTTPError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    elif submitted:
        st.warning("Please enter a question before searching.")


# ── Page 2 — Upload Document ─────────────────────────────────────────────────

elif page == "📄 Upload Document":
    st.title("📄 Ingest a Policy Document")
    st.markdown("Upload a policy document to index it into the Qdrant vector database.")

    uploaded = st.file_uploader(
        "Select document",
        type=["pdf", "docx", "md", "html"],
    )

    with st.form("ingest_form"):
        c1, c2 = st.columns(2)
        policy_category  = c1.selectbox(
            "Policy Category",
            ["", "Travel", "HR", "Legal", "Finance", "Procurement", "Security", "Insurance", "General"],
            help="Leave blank to auto-detect from content",
        )
        policy_owner     = c2.text_input("Department / Owner Code", placeholder="e.g. GCT-RM")
        effective_date   = c1.text_input("Effective Date (YYYY-MM-DD)", placeholder="2026-01-01")
        document_version = c2.text_input("Version", value="V1", placeholder="V3")

        ingest_btn = st.form_submit_button(
            "⬆️ Ingest Document", type="primary", disabled=(uploaded is None)
        )

    if ingest_btn and uploaded:
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                result = requests.post(
                    f"{API}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    data={
                        "policy_category":  policy_category,
                        "policy_owner":     policy_owner,
                        "effective_date":   effective_date,
                        "document_version": document_version,
                    },
                    timeout=180,
                )
                result.raise_for_status()
                data = result.json()
                st.success(f"✅ {data['message']}")
                col1, col2 = st.columns(2)
                col1.metric("Chunks indexed", data["chunks_ingested"])
                col2.code(f"Document ID: {data['document_id']}")
            except requests.HTTPError as e:
                st.error(f"Ingestion failed: {e.response.text}")


# ── Page 3 — Pipeline Debug ───────────────────────────────────────────────────

elif page == "🔍 Pipeline Debug":
    st.title("🔍 Pipeline Debugger")
    st.markdown(
        "Step through every stage of the retrieval pipeline for a query. "
        "Useful for viva demonstration and understanding how each component contributes."
    )

    debug_query = st.text_input(
        "Debug query",
        value="Can I expense a taxi during business travel?",
    )

    if st.button("▶ Run Pipeline Debug", type="primary"):
        # Import here so Streamlit doesn't fail if models aren't loaded
        from retrieval.hyde import generate_hypothetical_doc
        from retrieval.pipeline import detect_category, retrieve
        from retrieval.query_expansion import expand_query

        st.divider()

        # Step 1 — Category
        st.subheader("Step 1 · Category Detection")
        cat = detect_category(debug_query)
        if cat:
            st.success(f"Detected category: **{cat}**  → Qdrant filter will be applied")
        else:
            st.info("No category detected → unfiltered search across all policies")

        # Step 2 — Query expansion
        st.subheader("Step 2 · Query Expansion")
        with st.spinner("Generating variants..."):
            variants = expand_query(debug_query, n=4)
        for i, v in enumerate(variants):
            st.markdown(f"{'🔵 **Original**' if i == 0 else f'🔹 Variant {i}'}: {v}")

        # Step 3 — HyDE
        st.subheader("Step 3 · HyDE — Hypothetical Policy Document")
        with st.spinner("Generating hypothetical document..."):
            hyde_doc = generate_hypothetical_doc(debug_query)
        st.info(hyde_doc)

        # Step 4 — Retrieval + reranking
        st.subheader("Step 4 · Vector Retrieval + Cross-Encoder Reranking")
        with st.spinner("Running full pipeline (retrieval → reranking)..."):
            chunks = retrieve(debug_query, top_k_retrieval=25, top_k_rerank=5)

        st.success(f"**{len(chunks)} chunks** returned after reranking")
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            with st.expander(
                f"Chunk {i}  ·  rerank={chunk.get('rerank_score', 0):.3f}  "
                f"·  {meta.get('document_id', '?')}  "
                f"·  {meta.get('h1_header', '')}",
                expanded=(i == 1),
            ):
                st.text_area("Text", chunk["chunk_text"][:800], height=120, key=f"chunk_{i}")
                st.json(meta)


# ── Page 4 — Evaluation ───────────────────────────────────────────────────────

elif page == "📊 Evaluation":
    st.title("📊 Evaluation Metrics")
    st.markdown(
        "Project Aegis is evaluated using the [RAGAS framework](https://github.com/explodinggradients/ragas). "
        "Run `scripts/evaluate.py` from the terminal for full metric computation."
    )

    # Baseline vs Aegis results from the submission doc
    st.subheader("Baseline vs Project Aegis")
    import pandas as pd

    df = pd.DataFrame({
        "Metric":         ["Retrieval Precision", "Hallucination Rate", "Query Accuracy", "Avg Latency", "Table Understanding"],
        "Basic RAG":      ["61%", "28%", "65%", "5.4s", "Poor"],
        "Project Aegis":  ["91%", "7%",  "93%", "3.2s", "Excellent"],
        "Improvement":    ["+30pp", "−21pp", "+28pp", "−2.2s", "Major"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("RAGAS Metric Definitions")
    st.markdown("""
| Metric | What it measures |
|---|---|
| **Faithfulness** | Are answers grounded in the retrieved context? |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Precision** | Of the retrieved chunks, how many are relevant? |
| **Context Recall** | Are all relevant chunks being retrieved? |
""")

    st.divider()
    st.subheader("Run Evaluation")
    st.code(
        "# Prepare eval_qa.json with question / ground_truth pairs\n"
        "python scripts/evaluate.py --qa_file data/eval_qa.json",
        language="bash",
    )

    sample = json.dumps([
        {"question": "What is the per diem for USA travel?", "ground_truth": "$120 per day"},
        {"question": "How many weeks of maternity leave are employees entitled to?", "ground_truth": "26 weeks"},
    ], indent=2)
    st.code(sample, language="json")
