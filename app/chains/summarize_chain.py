"""
summarize_chain.py - Summarization Chain menggunakan LangChain
Fitur: Map-reduce summarization untuk dokumen panjang
"""

from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from app.config import LLM_MODEL, TEMPERATURE, GROQ_API_KEY


MAP_PROMPT_TEMPLATE = """Buat ringkasan singkat dari teks berikut dalam Bahasa Indonesia:

"{text}"

RINGKASAN SINGKAT:"""

COMBINE_PROMPT_TEMPLATE = """Gabungkan ringkasan-ringkasan berikut menjadi satu
ringkasan komprehensif dalam Bahasa Indonesia. Sertakan poin-poin utama.

"{text}"

RINGKASAN KOMPREHENSIF:"""

MAP_PROMPT     = PromptTemplate(template=MAP_PROMPT_TEMPLATE,     input_variables=["text"])
COMBINE_PROMPT = PromptTemplate(template=COMBINE_PROMPT_TEMPLATE, input_variables=["text"])


def build_summarize_chain(chain_type: str = "map_reduce"):
    """
    Membangun summarization chain.

    Alur kerja LangChain:
    - 'stuff'      : cocok untuk dokumen pendek (langsung ke LLM)
    - 'map_reduce' : cocok untuk dokumen panjang (ringkas per chunk, lalu gabung)
    - 'refine'     : iteratif, semakin halus setiap iterasi
    """
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        groq_api_key=GROQ_API_KEY
    )

    if chain_type == "map_reduce":
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=MAP_PROMPT,
            combine_prompt=COMBINE_PROMPT,
            verbose=False
        )
    else:
        chain = load_summarize_chain(llm, chain_type=chain_type)

    return chain


def run_summarize(text: str, chain_type: str = "map_reduce") -> str:
    """Ringkas teks panjang menggunakan LangChain summarization chain."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs     = splitter.create_documents([text])
    chain    = build_summarize_chain(chain_type)
    result   = chain.invoke({"input_documents": docs})
    return result["output_text"]