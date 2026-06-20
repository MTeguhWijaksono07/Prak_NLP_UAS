"""
streamlit_app.py - Demo UI untuk sistem NLP/LLM
Menampilkan penggunaan LangChain, LangGraph, dan LangSmith secara visual
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.agent_graph import run_agent
from app.chains.summarize_chain import run_summarize
from app.config import LANGCHAIN_PROJECT

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="NLP/LLM System - UAS",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Sistem NLP/LLM Pribadi")
st.caption(f"Menggunakan LangChain + LangGraph + LangSmith | Project: `{LANGCHAIN_PROJECT}`")

# ============================================================
# Sidebar - Info Library
# ============================================================
with st.sidebar:
    st.header("📚 Library yang Digunakan")

    with st.expander("🔗 LangChain", expanded=True):
        st.markdown("""
        - **RetrievalQA Chain** untuk tanya-jawab berbasis dokumen
        - **Map-Reduce Summarization** untuk meringkas teks panjang
        - **PromptTemplate** untuk custom prompting
        - **FAISS Vector Store** untuk penyimpanan embedding
        """)

    with st.expander("🕸️ LangGraph", expanded=True):
        st.markdown("""
        - **StateGraph** untuk workflow multi-step
        - **Conditional Edges** untuk routing dinamis
        - **Node Functions** untuk setiap langkah proses
        - **AgentState** untuk manajemen state
        """)

    with st.expander("📊 LangSmith", expanded=True):
        st.markdown("""
        - **Tracing** otomatis setiap eksekusi chain
        - **Monitoring** latensi dan token usage
        - **Evaluasi** kualitas output LLM
        - **Project Dashboard** untuk visualisasi
        """)

    st.divider()
    st.info(f"🔍 Semua trace tersimpan di LangSmith project:\n**{LANGCHAIN_PROJECT}**")

# ============================================================
# Tab Layout
# ============================================================
tab1, tab2, tab3 = st.tabs(["💬 Chat Agent (LangGraph)", "📝 Summarizer (LangChain)", "🔍 Info Arsitektur"])

# ---- TAB 1: LangGraph Chat Agent ----
with tab1:
    st.subheader("💬 Chat Agent dengan LangGraph")
    st.markdown("""
    Agent ini menggunakan **LangGraph** untuk routing cerdas:
    - Deteksi otomatis jenis tugas (QA / Summarize / Chat)
    - Multi-step workflow dengan state management
    """)

    if "messages" not in st.session_state:
        st.session_state.messages    = []
        st.session_state.chat_history = []

    # Tampilkan riwayat chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "task_type" in msg:
                st.caption(f"🏷️ Terdeteksi sebagai: **{msg['task_type'].upper()}**")

    # Input chat
    if prompt := st.chat_input("Ketik pertanyaan atau teks untuk diproses..."):
        # Tampilkan pesan user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Proses dengan LangGraph agent
        with st.chat_message("assistant"):
            with st.spinner("🕸️ LangGraph sedang memproses..."):
                try:
                    response = run_agent(prompt, st.session_state.chat_history)
                    result   = response["result"]
                    task     = response["task_type"]

                    # Hapus prefix tag dari hasil
                    clean_result = result.split("] ", 1)[-1] if "] " in result else result

                    st.write(clean_result)
                    st.caption(f"🏷️ Terdeteksi sebagai: **{task.upper()}** | ✅ Trace tersimpan di LangSmith")

                    st.session_state.messages.append({
                        "role":      "assistant",
                        "content":   clean_result,
                        "task_type": task
                    })
                    st.session_state.chat_history = response["messages"]

                except Exception as e:
                    st.error(f"Error: {e}\n\nPastikan GROQ_API_KEY sudah diset di file \".env\"")

    if st.button("🗑️ Hapus Riwayat"):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.rerun()

# ---- TAB 2: Summarizer ----
with tab2:
    st.subheader("📝 Text Summarizer dengan LangChain")
    st.markdown("Menggunakan **Map-Reduce Chain** dari LangChain untuk meringkas teks panjang.")

    col1, col2 = st.columns([1, 1])

    with col1:
        input_text = st.text_area(
            "Masukkan teks yang ingin diringkas:",
            height=300,
            placeholder="Paste teks panjang di sini..."
        )
        chain_type = st.selectbox("Metode Chain:", ["map_reduce", "stuff", "refine"])

        if st.button("📝 Ringkas Sekarang", type="primary"):
            if input_text.strip():
                with st.spinner("🔗 LangChain Map-Reduce sedang berjalan..."):
                    try:
                        summary = run_summarize(input_text, chain_type)
                        st.session_state.summary = summary
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Masukkan teks terlebih dahulu!")

    with col2:
        if "summary" in st.session_state:
            st.subheader("📋 Hasil Ringkasan:")
            st.success(st.session_state.summary)
            st.caption("✅ Trace tersimpan di LangSmith")

# ---- TAB 3: Arsitektur ----
with tab3:
    st.subheader("🔍 Arsitektur Sistem")

    st.code("""
┌─────────────────────────────────────────────┐
│           SISTEM NLP/LLM PRIBADI            │
│                                             │
│  ┌─────────────┐   ┌─────────────────────┐  │
│  │  LangSmith  │◄──│   Semua Chain &     │  │
│  │  (Tracing)  │   │   Graph Nodes       │  │
│  └─────────────┘   └─────────────────────┘  │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │          LangGraph Workflow          │   │
│  │                                      │   │
│  │  START → classify_task               │   │
│  │              ↓                       │   │
│  │     ┌────────┴────────┐              │   │
│  │     ↓        ↓        ↓              │   │
│  │  process  process  process           │   │
│  │    _qa   _summ    _chat              │   │
│  │     └────────┬────────┘              │   │
│  │              ↓                       │   │
│  │           finalize → END             │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │        LangChain Components          │   │
│  │  • RetrievalQA Chain + FAISS         │   │
│  │  • Map-Reduce Summarization          │   │
│  │  • PromptTemplate & ChatOpenAI       │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
    """, language="text")

    st.markdown("""
    ### Alur Data:
    1. **User Input** → masuk ke LangGraph StateGraph
    2. **classify_task** → LLM menentukan jenis tugas
    3. **Conditional Routing** → diarahkan ke node yang tepat
    4. **Process Node** → menggunakan LangChain chain yang sesuai
    5. **finalize** → simpan ke riwayat state
    6. **LangSmith** → otomatis trace semua langkah
    """)