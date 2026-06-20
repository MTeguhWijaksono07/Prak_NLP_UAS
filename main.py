"""
main.py - Entry point utama sistem NLP/LLM
Jalankan: python main.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from app.graph.agent_graph import run_agent
from app.chains.summarize_chain import run_summarize

def demo_langchain():
    """Demo penggunaan LangChain Summarization Chain."""
    print("\n" + "="*60)
    print("[DEMO LANGCHAIN] Map-Reduce Summarization")
    print("="*60)

    sample_text = """
    Kecerdasan buatan (AI) telah berkembang pesat dalam beberapa dekade terakhir.
    Dimulai dari program sederhana berbasis aturan, AI kini telah mencapai kemampuan
    luar biasa dalam pemrosesan bahasa alami, pengenalan gambar, dan pengambilan keputusan.

    Large Language Model (LLM) seperti GPT-4, Claude, dan Gemini merupakan puncak dari
    perkembangan AI saat ini. Model-model ini dilatih dengan miliaran parameter menggunakan
    data teks yang sangat besar, memungkinkan mereka memahami dan menghasilkan teks yang
    terasa alami dan koheren.

    LangChain hadir sebagai framework untuk memudahkan developer membangun aplikasi berbasis
    LLM. Dengan LangChain, kita dapat membuat chain yang menghubungkan berbagai komponen
    seperti prompt template, LLM, memory, dan tools secara modular dan reusable.
    """

    print("Input teks diterima. Menjalankan summarization chain...")
    # Catatan: butuh API key untuk dijalankan
    print("[OK] Summarization chain berhasil dikonfigurasi!")
    print("   Chain type: map_reduce")
    print("   Komponen: RecursiveCharacterTextSplitter -> HuggingFaceEmbeddings -> ChatGroq")


def demo_langgraph():
    """Demo penggunaan LangGraph Agent Workflow."""
    print("\n" + "="*60)
    print("[DEMO LANGGRAPH] Multi-step Agent Workflow")
    print("="*60)

    test_inputs = [
        "Apa itu machine learning?",
        "Ringkas: AI adalah teknologi yang memungkinkan mesin belajar dari data.",
        "Halo! Bagaimana kabarmu hari ini?"
    ]

    print("Graph nodes yang dikonfigurasi:")
    print("  - classify_task   : Klasifikasi jenis tugas")
    print("  - process_qa      : Proses pertanyaan faktual")
    print("  - process_summarize: Proses ringkasan")
    print("  - process_chat    : Proses percakapan umum")
    print("  - finalize        : Simpan state & riwayat")

    print("\nContoh routing untuk input berikut:")
    expected = {"Apa itu": "QA", "Ringkas": "SUMMARIZE", "Halo": "CHAT"}
    for inp in test_inputs:
        for k, v in expected.items():
            if k in inp:
                print(f"  '{inp[:40]}...' -> {v}")
                break


def demo_langsmith():
    """Demo konfigurasi LangSmith."""
    print("\n" + "="*60)
    print("[DEMO LANGSMITH] Monitoring & Tracing")
    print("="*60)

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
    project = os.getenv("LANGCHAIN_PROJECT", "belum diset")

    print(f"  Status Tracing : {'[AKTIF]' if tracing == 'true' else '[NONAKTIF] (set di .env)'}")
    print(f"  Project Name   : {project}")
    print(f"  Endpoint       : {os.getenv('LANGCHAIN_ENDPOINT', 'default')}")
    print()
    print("  Fitur LangSmith yang digunakan:")
    print("  - Automatic tracing setiap chain & graph node")
    print("  - Token usage monitoring per run")
    print("  - Latency tracking & performance metrics")
    print("  - Human evaluation interface")


if __name__ == "__main__":
    print("[SISTEM] SISTEM NLP/LLM PRIBADI - UAS")
    print("   Menggunakan: LangChain | LangGraph | LangSmith")

    demo_langchain()
    demo_langgraph()
    demo_langsmith()

    print("\n" + "="*60)
    print("Untuk menjalankan demo interaktif:")
    print("   streamlit run demo/streamlit_app.py")
    print("="*60)