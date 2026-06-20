"""
qa_chain.py - Question & Answer Chain menggunakan LangChain
Fitur: RetrievalQA dengan FAISS vector store untuk dokumen lokal
"""

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from app.config import LLM_MODEL, TEMPERATURE, MAX_TOKENS, GROQ_API_KEY, EMBED_MODEL


QA_PROMPT_TEMPLATE = """Gunakan konteks berikut untuk menjawab pertanyaan.
Jika jawabannya tidak ada dalam konteks, katakan bahwa kamu tidak tahu.

Konteks:
{context}

Pertanyaan: {question}

Jawaban (dalam Bahasa Indonesia):"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def build_qa_chain(documents: list[str]) -> RetrievalQA:
    """
    Membangun RetrievalQA chain dari daftar dokumen teks.

    Alur kerja LangChain:
    1. Split dokumen menjadi chunk kecil
    2. Embed setiap chunk menggunakan HuggingFace Embeddings (lokal)
    3. Simpan embedding ke FAISS vector store
    4. Buat RetrievalQA chain yang mengambil konteks relevan
    """
    # Step 1: Buat Document objects
    docs = [Document(page_content=text) for text in documents]

    # Step 2: Split teks menjadi chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(docs)

    # Step 3: Buat embeddings dan vector store (FAISS)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Step 4: Buat LLM dan RetrievalQA chain
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        groq_api_key=GROQ_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    return qa_chain


def run_qa(chain: RetrievalQA, question: str) -> dict:
    """Jalankan QA chain dan kembalikan jawaban + sumber dokumen."""
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result.get("source_documents", [])]
    }