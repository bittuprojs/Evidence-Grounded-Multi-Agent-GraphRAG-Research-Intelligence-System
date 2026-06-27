# Evidence-Grounded Multi-Agent GraphRAG Research Intelligence System

A multi-agent GraphRAG research assistant that combines Retrieval-Augmented Generation (RAG), hybrid retrieval, evidence verification, graph analytics, and Gemini-powered reasoning over 650+ research papers.

## Features

- Multi-domain research corpus (650+ papers)
- Hybrid Retrieval (BM25 + Semantic Search)
- ChromaDB Vector Database
- Gemini-powered Question Answering
- Research Paper Summarization
- Paper Comparison Engine
- Uploaded PDF Analysis
- Evidence Verification
- Contradiction Detection
- Confidence Scoring
- Research Gap Detection
- Graph Visualization
- Multi-Agent Workflow
- Evaluation Dashboard

## Tech Stack

Python, Gemini API, ChromaDB, SentenceTransformers, BM25, CrossEncoder, Streamlit, NetworkX, Pandas, NumPy, PyPDF

## Run

```bash
pip install -r requirements.txt

streamlit run ui/streamlit_app.py