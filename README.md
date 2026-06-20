<img width="1365" height="767" alt="image" src="https://github.com/user-attachments/assets/0aca2d25-c1b9-45ba-bfb3-e6efa5bae70c" /># 🤖 Sistem NLP/LLM Pribadi - UAS

Proyek UAS berbasis sistem **NLP/LLM** yang mengintegrasikan **LangChain**, **LangGraph**, dan **LangSmith** dalam satu aplikasi terpadu.

## 📚 Library yang Digunakan

| Library | Fungsi | Fitur yang Dipakai |
|---------|--------|-------------------|
| **LangChain** | Framework utama LLM | RetrievalQA Chain, Map-Reduce Summarization, PromptTemplate, FAISS VectorStore |
| **LangGraph** | Multi-step workflow | StateGraph, Conditional Edges, Node-based routing, State Management |
| **LangSmith** | Monitoring & Evaluasi | Auto-tracing, Token monitoring, Latency tracking, Project dashboard |

## 🏗️ Arsitektur Sistem

```
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
│  │  START → classify_task               │   │
│  │        ↓        ↓        ↓           │   │
│  │    process_qa  summ    chat           │   │
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
```
## 📁 Aplikasi
<img width="1365" height="767" alt="image" src="https://github.com/user-attachments/assets/cc010c94-6e1e-4375-9852-81539f5e5e34" />

## 📁 Struktur Proyek

```
nlp-uas-system/
├── main.py                      # Entry point & demo CLI
├── requirements.txt             # Dependencies
├── .env.example                 # Template environment variables
├── README.md
│
├── app/
│   ├── config.py                # Konfigurasi global & LangSmith
│   ├── chains/
│   │   ├── qa_chain.py          # LangChain: RetrievalQA
│   │   └── summarize_chain.py   # LangChain: Map-Reduce Summarization
│   ├── graph/
│   │   └── agent_graph.py       # LangGraph: StateGraph workflow
│   └── utils/
│
└── demo/
    └── streamlit_app.py         # Demo UI interaktif
```

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/username/nlp-uas-system.git
cd nlp-uas-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
```bash
cp .env.example .env
# Edit .env dan isi API keys:
# - OPENAI_API_KEY
# - LANGCHAIN_API_KEY (dari smith.langchain.com)
```

### 4. Jalankan Demo CLI
```bash
python main.py
```

### 5. Jalankan Aplikasi Web (Streamlit)
```bash
streamlit run demo/streamlit_app.py
```

## 🔧 Penjelasan Setiap Library

### 🔗 LangChain
LangChain digunakan sebagai fondasi utama untuk membangun pipeline LLM:
- **`RetrievalQA`** — Menjawab pertanyaan berdasarkan dokumen lokal menggunakan FAISS vector store
- **`load_summarize_chain`** — Map-reduce summarization untuk dokumen panjang
- **`PromptTemplate`** — Custom prompt untuk mengontrol output LLM
- **`RecursiveCharacterTextSplitter`** — Memecah dokumen panjang menjadi chunk

### 🕸️ LangGraph
LangGraph mengelola alur kerja multi-langkah berbasis graph:
- **`StateGraph`** — Mendefinisikan workflow sebagai directed graph
- **`AgentState`** — TypedDict untuk menyimpan state antar node
- **`Conditional Edges`** — Routing dinamis berdasarkan hasil klasifikasi
- Node: `classify_task → process_* → finalize`

### 📊 LangSmith
LangSmith memberikan visibilitas penuh ke dalam sistem:
- **Auto-tracing** — Setiap chain dan graph node otomatis di-trace
- **Token monitoring** — Pantau penggunaan token per request
- **Latency tracking** — Ukur performa setiap komponen
- Aktifkan dengan set `LANGCHAIN_TRACING_V2=true`

## 👤 Informasi Mahasiswa

| Item | Detail |
|------|--------|
| Nama | [Nama Mahasiswa] |
| NIM | [NIM] |
| Mata Kuliah | Pemrosesan Bahasa Alami / NLP |
| Jenis Tugas | UAS — Sistem NLP/LLM Pribadi |
