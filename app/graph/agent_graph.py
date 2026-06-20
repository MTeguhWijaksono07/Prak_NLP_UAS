"""
agent_graph.py - Multi-step Agent Workflow menggunakan LangGraph
Fitur: Graph-based workflow dengan state management untuk routing cerdas
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import LLM_MODEL, TEMPERATURE, GROQ_API_KEY, SYSTEM_PROMPT
import operator


# ============================================================
# State Definition
# LangGraph memerlukan TypedDict untuk mendefinisikan state graph
# ============================================================
class AgentState(TypedDict):
    messages:  Annotated[list, operator.add]   # riwayat percakapan
    user_input: str                            # input pengguna
    task_type:  str                            # jenis tugas yang terdeteksi
    result:     str                            # hasil akhir
    iteration:  int                            # jumlah iterasi


# ============================================================
# Node Functions
# Setiap node adalah fungsi Python yang menerima & mengembalikan state
# ============================================================

def classify_task(state: AgentState) -> AgentState:
    """
    Node 1: Klasifikasi jenis tugas berdasarkan input pengguna.
    Menentukan apakah user ingin: QA, Summarize, atau Chat biasa.
    """
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

    classification_prompt = f"""Klasifikasikan pertanyaan/tugas berikut ke salah satu kategori:
- "qa"        : jika user bertanya tentang fakta atau informasi spesifik
- "summarize" : jika user ingin meringkas teks
- "chat"      : jika user ingin percakapan umum atau diskusi

Input: "{state['user_input']}"

Jawab HANYA dengan satu kata: qa, summarize, atau chat"""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    task_type = response.content.strip().lower()

    if task_type not in ["qa", "summarize", "chat"]:
        task_type = "chat"

    return {**state, "task_type": task_type}


def process_qa(state: AgentState) -> AgentState:
    """Node 2a: Proses pertanyaan berbasis fakta."""
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        groq_api_key=GROQ_API_KEY
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\nJawab pertanyaan berikut secara faktual dan akurat."),
        HumanMessage(content=state["user_input"])
    ]
    response = llm.invoke(messages)
    return {**state, "result": f"[QA] {response.content}"}


def process_summarize(state: AgentState) -> AgentState:
    """Node 2b: Proses ringkasan teks."""
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
        groq_api_key=GROQ_API_KEY
    )

    messages = [
        SystemMessage(content="Kamu adalah ahli meringkas teks. Buat ringkasan yang padat dan informatif."),
        HumanMessage(content=f"Ringkas teks berikut:\n\n{state['user_input']}")
    ]
    response = llm.invoke(messages)
    return {**state, "result": f"[RINGKASAN] {response.content}"}


def process_chat(state: AgentState) -> AgentState:
    """Node 2c: Proses percakapan umum."""
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        groq_api_key=GROQ_API_KEY
    )

    history  = state.get("messages", [])
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=state["user_input"])]
    response = llm.invoke(messages)
    return {**state, "result": f"[CHAT] {response.content}"}


def finalize(state: AgentState) -> AgentState:
    """Node 3: Finalisasi dan simpan ke riwayat percakapan."""
    new_messages = [
        HumanMessage(content=state["user_input"]),
        HumanMessage(content=state["result"])   # simpan hasil ke history
    ]
    return {
        **state,
        "messages":  state.get("messages", []) + new_messages,
        "iteration": state.get("iteration", 0) + 1
    }


# ============================================================
# Routing Function
# LangGraph menggunakan conditional edges untuk routing dinamis
# ============================================================
def route_by_task(state: AgentState) -> Literal["process_qa", "process_summarize", "process_chat"]:
    """Router: Arahkan ke node yang sesuai berdasarkan task_type."""
    task_map = {
        "qa":        "process_qa",
        "summarize": "process_summarize",
        "chat":      "process_chat"
    }
    return task_map.get(state["task_type"], "process_chat")


# ============================================================
# Build Graph
# ============================================================
def build_agent_graph() -> StateGraph:
    """
    Membangun LangGraph workflow.

    Alur graph:
    START → classify_task → [route] → process_* → finalize → END
    """
    graph = StateGraph(AgentState)

    # Tambahkan nodes
    graph.add_node("classify_task",      classify_task)
    graph.add_node("process_qa",         process_qa)
    graph.add_node("process_summarize",  process_summarize)
    graph.add_node("process_chat",       process_chat)
    graph.add_node("finalize",           finalize)

    # Set entry point
    graph.set_entry_point("classify_task")

    # Conditional edge: routing berdasarkan task type
    graph.add_conditional_edges(
        "classify_task",
        route_by_task,
        {
            "process_qa":        "process_qa",
            "process_summarize": "process_summarize",
            "process_chat":      "process_chat"
        }
    )

    # Semua proses mengarah ke finalize, lalu END
    graph.add_edge("process_qa",        "finalize")
    graph.add_edge("process_summarize", "finalize")
    graph.add_edge("process_chat",      "finalize")
    graph.add_edge("finalize",          END)

    return graph.compile()


def run_agent(user_input: str, history: list = None) -> dict:
    """Jalankan agent graph dengan input pengguna."""
    app = build_agent_graph()

    initial_state: AgentState = {
        "messages":   history or [],
        "user_input": user_input,
        "task_type":  "",
        "result":     "",
        "iteration":  0
    }

    final_state = app.invoke(initial_state)
    return {
        "result":    final_state["result"],
        "task_type": final_state["task_type"],
        "messages":  final_state["messages"]
    }