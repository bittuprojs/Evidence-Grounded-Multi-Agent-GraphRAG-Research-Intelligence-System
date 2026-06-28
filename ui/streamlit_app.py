import sys
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from ingestion.pdf_loader import extract_pdf_text
from system_builder import build_system
from generation.paper_writer import PaperWriter
from analytics.trend import domain_distribution, year_distribution, top_keywords
from evaluation.benchmark import EvaluationSuite
st.set_page_config(
    page_title="Research Intelligence Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem;}
    .stMetric {border: 1px solid rgba(128,128,128,0.2); padding: 10px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_system():
    answer_engine, docs, graph = build_system()
    writer = PaperWriter(llm_fn=answer_engine.llm_fn)
    return answer_engine, docs, graph, writer

answer_engine, docs, graph, writer = load_system()
from agents.research_workflow import ResearchWorkflow
workflow = ResearchWorkflow(
    retriever=answer_engine.retriever,
    llm_fn=answer_engine.llm_fn,
    all_docs=docs
)
evaluator = EvaluationSuite(answer_engine)
if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = ""

st.title("📚 Research Intelligence Assistant")
st.caption("Multi-domain research QA • uploaded paper analysis • compare mode • summarization • writing studio • analytics")

tab1, tab2, tab3, tab4, tab5, tab6,tab7,tab8 = st.tabs([
    "Corpus QA",
    "Compare",
    "Summarization",
    "Uploaded Paper",
    "Writing Studio",
    "Analytics Dashboard",
    "Multi-Agent Workflow",
    "Evaluation Suite"
])

with st.sidebar:
    st.header("Settings")
    domain = st.selectbox(
        "Domain",
        ["All Domains", "AI/ML", "Medical", "Finance", "General"],
        index=0
    )

    st.divider()

    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)

        text = extract_pdf_text(str(tmp_path))
        st.session_state.uploaded_text = text
        st.session_state.uploaded_name = uploaded_file.name
        st.success(f"Loaded: {uploaded_file.name}")
        st.text_area("Preview", value=text[:3000], height=200)

# ----------------------------------------------------
# TAB 1: CORPUS QA
# ----------------------------------------------------
with tab1:
    st.subheader("Ask a question from the research corpus")

    q = st.text_area("Your question", placeholder="What is machine learning?", height=100)
    ask_btn = st.button("Ask Corpus")

    if ask_btn and q.strip():
        result = answer_engine.answer(q.strip(), domain=domain)

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", result["confidence"])
        c2.metric("Score", f"{result['confidence_score']:.4f}")
        c3.metric("Time (s)", f"{result['processing_time']:.2f}")

        st.markdown("### Answer")
        st.write(result["answer"])

        st.markdown("### Rewritten Query")
        st.code(result.get("rewritten_query", ""), language="text")

        st.markdown("### Sources")
        for s in result["sources"]:
            st.markdown(
                f"- **{s['title']}** ({s.get('year', 'n.d.')}) — score: {s['score']}\n\n"
                f"  {s['excerpt']}"
            )

        st.markdown("### Citations")
        st.code(result.get("citations", ""), language="text")

        st.markdown("### Research Gaps")
        gaps = result.get("research_gaps", {})
        if isinstance(gaps, dict):
            for gap in gaps.get("gaps", []):
                st.write(f"- {gap}")
        else:
            st.write(gaps)
        
        
        st.markdown("### Verification")
        verification = result.get("verification", {})
        if isinstance(verification, dict):
            evidence = verification.get("evidence", {})
            contradiction = verification.get("contradiction", {})

            st.write(f"**Evidence label:** {evidence.get('label', 'unknown')}")
            st.write(f"**Evidence support score:** {evidence.get('support_score', 0.0)}")
            for issue in evidence.get("issues", []):
                st.write(f"- {issue}")

            st.write(f"**Contradiction label:** {contradiction.get('label', 'unknown')}")
            st.write(f"**Contradiction score:** {contradiction.get('contradiction_score', 0.0)}")
            for issue in contradiction.get("issues", []):
                st.write(f"- {issue}")
        else:
            st.write(verification)

# ----------------------------------------------------
# TAB 2: COMPARE
# ----------------------------------------------------
with tab2:
    st.subheader("Compare mode")

    compare_mode = st.radio(
        "What do you want to compare?",
        [
            "Two corpus topics",
            "Uploaded paper vs topic",
        ],
        horizontal=True
    )

    if compare_mode == "Two corpus topics":
        col1, col2 = st.columns(2)
        with col1:
            topic_a = st.text_input("Topic A", placeholder="BERT")
        with col2:
            topic_b = st.text_input("Topic B", placeholder="RoBERTa")

        compare_domain = st.selectbox(
            "Domain for comparison",
            ["All Domains", "AI/ML", "Medical", "Finance", "General"],
            index=0,
            key="compare_domain"
        )

        if st.button("Compare Topics"):
            if topic_a.strip() and topic_b.strip():
                result = answer_engine.compare_topics(
                    topic_a.strip(),
                    topic_b.strip(),
                    domain=compare_domain
                )

                c1, c2 = st.columns(2)
                c1.metric("Confidence", result["confidence"])
                c2.metric("Score", f"{result['confidence_score']:.4f}")

                st.markdown("### Comparison")
                st.write(result["answer"])

                st.markdown("### Sources")
                for s in result["sources"]:
                    st.markdown(
                        f"- **{s['title']}** ({s.get('year', 'n.d.')}) — score: {s['score']}\n\n"
                        f"  {s['excerpt']}"
                    )

                st.markdown("### Citations")
                st.code(result.get("citations", ""), language="text")

    else:
        if st.session_state.uploaded_text:
            st.info(f"Current paper: {st.session_state.uploaded_name}")

            topic = st.text_input("Topic to compare with", placeholder="Retrieval-Augmented Generation")

            if st.button("Compare Uploaded Paper"):
                if topic.strip():
                    comparison = writer.compare_uploaded_vs_topic(
                        st.session_state.uploaded_text,
                        st.session_state.uploaded_name,
                        topic.strip()
                    )
                    st.markdown("### Comparison")
                    st.write(comparison)
        else:
            st.warning("Upload a paper from the sidebar first.")

# ----------------------------------------------------
# TAB 3: SUMMARIZATION
# ----------------------------------------------------
with tab3:
    st.subheader("Summarization")

    summary_mode = st.radio(
        "Summarize",
        [
            "Corpus topic",
            "Uploaded paper",
        ],
        horizontal=True
    )

    if summary_mode == "Corpus topic":
        topic = st.text_input("Topic", placeholder="Transformer architecture")
        summary_domain = st.selectbox(
            "Domain for summary",
            ["All Domains", "AI/ML", "Medical", "Finance", "General"],
            index=0,
            key="summary_domain"
        )

        if st.button("Summarize Topic"):
            if topic.strip():
                result = answer_engine.summarize_topic(topic.strip(), domain=summary_domain)

                c1, c2 = st.columns(2)
                c1.metric("Confidence", result["confidence"])
                c2.metric("Score", f"{result['confidence_score']:.4f}")

                st.markdown("### Summary")
                st.write(result["answer"])

                st.markdown("### Sources")
                for s in result["sources"]:
                    st.markdown(
                        f"- **{s['title']}** ({s.get('year', 'n.d.')}) — score: {s['score']}\n\n"
                        f"  {s['excerpt']}"
                    )

                st.markdown("### Citations")
                st.code(result.get("citations", ""), language="text")

    else:
        if st.session_state.uploaded_text:
            st.info(f"Current paper: {st.session_state.uploaded_name}")

            if st.button("Summarize Uploaded Paper"):
                summary = writer.summarize_paper(
                    st.session_state.uploaded_text,
                    title=st.session_state.uploaded_name
                )
                st.markdown("### Summary")
                st.write(summary)
        else:
            st.warning("Upload a paper from the sidebar first.")

# ----------------------------------------------------
# TAB 4: UPLOADED PAPER
# ----------------------------------------------------
with tab4:
    st.subheader("Uploaded paper workspace")

    if st.session_state.uploaded_text:
        st.info(f"Current paper: {st.session_state.uploaded_name}")

        mode = st.selectbox(
            "What do you want to do?",
            [
                "Ask a question about the uploaded paper",
                "Analyze uploaded paper",
                "Generate paper draft outline"
            ]
        )

        if mode == "Ask a question about the uploaded paper":
            paper_q = st.text_area("Question about this paper", placeholder="What is the main contribution?", height=100)
            if st.button("Ask About Paper"):
                if paper_q.strip():
                    answer = writer.answer_about_paper(
                        paper_q.strip(),
                        st.session_state.uploaded_text,
                        title=st.session_state.uploaded_name
                    )
                    st.markdown("### Answer")
                    st.write(answer)

        elif mode == "Analyze uploaded paper":
            if st.button("Analyze Paper"):
                analysis = writer.analyze_paper(
                    st.session_state.uploaded_text,
                    title=st.session_state.uploaded_name
                )
                st.markdown("### Paper Analysis")
                st.write(analysis)

        elif mode == "Generate paper draft outline":
            topic = st.text_input("Topic for outline", placeholder="Research topic")
            if st.button("Generate Outline"):
                if topic.strip():
                    outline = writer.generate_outline(topic.strip())
                    st.markdown("### Outline")
                    st.write(outline)
    else:
        st.warning("Upload a PDF from the sidebar first.")

# ----------------------------------------------------
# TAB 5: WRITING STUDIO
# ----------------------------------------------------
with tab5:
    st.subheader("Research writing studio")

    topic = st.text_input("Topic", placeholder="Retrieval-Augmented Generation for Research Assistants")
    writing_mode = st.selectbox(
        "What do you want to create?",
        ["Outline", "Literature Review", "Improve Paragraph"],
        index=0
    )

    if writing_mode == "Improve Paragraph":
        paragraph = st.text_area("Paste your paragraph", height=180)
        if st.button("Improve Writing"):
            if paragraph.strip():
                improved = writer.improve_paragraph(paragraph.strip())
                st.markdown("### Improved Paragraph")
                st.write(improved)
    else:
        if st.button("Generate"):
            if topic.strip():
                if writing_mode == "Outline":
                    out = writer.generate_outline(topic.strip())
                    st.markdown("### Outline")
                    st.write(out)
                else:
                    lit = writer.draft_literature_review(topic.strip())
                    st.markdown("### Literature Review Draft")
                    st.write(lit)

# ----------------------------------------------------
# TAB 6: ANALYTICS DASHBOARD
# ----------------------------------------------------
with tab6:
    st.subheader("Corpus analytics")

    domain_data = domain_distribution(docs)
    year_data = year_distribution(docs)
    keywords = top_keywords(docs, top_k=15)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", len(docs))
    m2.metric("AI/ML", domain_data.get("AI/ML", 0))
    m3.metric("Medical", domain_data.get("Medical", 0))
    m4.metric("Finance", domain_data.get("Finance", 0))

    st.markdown("### Domain distribution")
    st.bar_chart(domain_data)

    st.markdown("### Year distribution")
    if year_data:
        st.line_chart(year_data)

    st.markdown("### Top keywords")
    st.write(keywords)

    st.markdown("### Graph visualization")
    graph_topic = st.text_input(
        "Graph focus topic",
        value="machine learning",
        help="Type a topic such as RAG, transformers, medical AI, finance, or any keyword."
    )

    graph_papers = st.slider(
        "Number of papers in graph",
        min_value=8,
        max_value=25,
        value=15
    )

    graph_keywords = st.slider(
        "Number of keywords in graph",
        min_value=5,
        max_value=15,
        value=10
    )

    if st.button("Build Graph"):
        fig = graph.plot_topic_graph(
            topic=graph_topic,
            max_papers=graph_papers,
            max_keywords=graph_keywords
        )
        st.pyplot(fig)

    st.markdown("### Graph summary")
    st.write(graph.domain_summary())
with tab7:
    st.subheader("Multi-Agent Workflow")

    wf_question = st.text_area(
        "Workflow question",
        placeholder="What is machine learning?",
        height=100
    )

    wf_domain = st.selectbox(
        "Workflow domain",
        ["All Domains", "AI/ML", "Medical", "Finance", "General"],
        index=0,
        key="workflow_domain"
    )

    if st.button("Run Workflow"):
        if wf_question.strip():
            result = workflow.run(wf_question.strip(), domain=wf_domain)

            c1, c2, c3 = st.columns(3)
            c1.metric("Intent", result.get("intent", ""))
            c2.metric("Confidence", result.get("confidence", ""))
            c3.metric("Score", f"{result.get('confidence_score', 0.0):.4f}")

            st.markdown("### Plan")
            st.json(result.get("plan", {}))

            st.markdown("### Retrieval")
            st.json(result.get("retrieval", {}))

            st.markdown("### Answer")
            st.write(result.get("answer", ""))

            st.markdown("### Verification")
            st.json(result.get("verification", {}))

            st.markdown("### Citations")
            st.code(result.get("citations", ""), language="text")

            st.markdown("### Research Gaps")
            st.json(result.get("research_gaps", {}))

            st.markdown("### Processing Time")
            st.write(result.get("processing_time", 0.0))
# ----------------------------------------------------
# TAB 8: EVALUATION SUITE
# ----------------------------------------------------
with tab8:
    st.subheader("Evaluation Suite")

    st.markdown("Run the built-in benchmark or test a custom question.")

    mode = st.radio(
        "Evaluation mode",
        ["Built-in benchmark", "Custom case"],
        horizontal=True
    )

    if mode == "Built-in benchmark":
        if st.button("Run Benchmark"):
            benchmark = evaluator.run_benchmark()

            st.markdown("### Summary")
            st.json(benchmark["summary"])

            st.markdown("### Results")
            st.dataframe(benchmark["results"], use_container_width=True)

            st.markdown("### Best / Weakest Cases")
            sorted_rows = sorted(
                benchmark["results"],
                key=lambda x: x["overall_quality"],
                reverse=True
            )

            if sorted_rows:
                st.write("Best case:")
                st.json(sorted_rows[0])

                st.write("Weakest case:")
                st.json(sorted_rows[-1])

    else:
        q = st.text_area("Question", placeholder="What is machine learning?", height=100)
        domain = st.selectbox(
            "Domain",
            ["All Domains", "AI/ML", "Medical", "Finance", "General"],
            index=0,
            key="eval_domain"
        )
        keywords = st.text_input(
            "Expected keywords (comma separated)",
            placeholder="machine learning, supervised learning, deep learning"
        )

        if st.button("Run Custom Evaluation"):
            expected = [k.strip() for k in keywords.split(",") if k.strip()]
            case_result = evaluator.run_case(
                question=q.strip(),
                domain=domain,
                expected_keywords=expected,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", case_result["status"])
            c2.metric("Support", f"{case_result['support_score']:.4f}")
            c3.metric("Contradiction", f"{case_result['contradiction_score']:.4f}")
            c4.metric("Overall", f"{case_result['overall_quality']:.4f}")

            st.markdown("### Answer")
            st.write(case_result["answer"])

            st.markdown("### Verification")
            st.json(case_result["verification"])

            st.markdown("### Sources")
            st.dataframe(case_result["sources"], use_container_width=True)