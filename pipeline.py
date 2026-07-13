"""
pipeline.py — LangGraph state machine, short-term memory, and CLI entry point.
Run this file to start PoBot.
"""
import os
from typing import TypedDict, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from retrieval import hybrid_retrieve, rerank_chunks
from prompts import CRAG_PROMPT, GENERATE_PROMPT, SUMMARIZE_PROMPT, ROUTE_PROMPT, EXPAND_PROMPT

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)

MAX_RETRIES = 2
MAX_MEMORY_MESSAGES = 10
SUMMARY_WORD_LIMIT = 600


# ============================================================
# STRUCTURED OUTPUT SCHEMAS
# ============================================================
class RouteDecision(BaseModel):
    route: Literal["RAG", "CHITCHAT"] = Field(description="How this message should be handled")

class CragVerdict(BaseModel):
    verdict: Literal["USABLE", "PARTIALLY_USABLE", "NOT_USABLE"]
    rewritten_query: str = Field(
        description="Required when verdict is NOT_USABLE. Empty string otherwise."
    )

router_llm = llm.with_structured_output(RouteDecision)
crag_llm = llm.with_structured_output(CragVerdict)


# ============================================================
# SHORT-TERM MEMORY — accessible from every node
# ============================================================
class ConversationMemory:
    def __init__(self):
        self.messages = []   # [{"role": "user"|"assistant", "content": str}, ...]
        self.summary = ""    # rolling summary of evicted messages, capped ~600 words

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > MAX_MEMORY_MESSAGES:
            evicted = self.messages.pop(0)
            self._update_summary(evicted)

    def _update_summary(self, evicted_message: dict):
        conversation_text = f"{evicted_message['role']}: {evicted_message['content']}"
        existing_note = (f"Existing summary to merge with (keep total under "
                          f"{SUMMARY_WORD_LIMIT} words):\n{self.summary}" if self.summary else "")
        prompt = SUMMARIZE_PROMPT.format(
            existing_summary_note=existing_note, conversation_text=conversation_text
        )
        result = llm.invoke(prompt)
        self.summary = result.content.strip()

    def get_full_context(self) -> str:
        """Summary + all retained raw messages — used everywhere memory is needed."""
        parts = []
        if self.summary:
            parts.append(f"[Earlier conversation summary]\n{self.summary}")
        if self.messages:
            parts.append("[Recent messages]\n" +
                          "\n".join(f"{m['role']}: {m['content']}" for m in self.messages))
        return "\n\n".join(parts) if parts else "None yet — this is the first message."


memory = ConversationMemory()


# ============================================================
# GRAPH STATE
# ============================================================
class RAGState(TypedDict):
    query: str
    expanded_query: str
    retrieved_chunks: list
    response: str
    crag_retry_count: int
    crag_verdict: str
    route: str


# ============================================================
# NODES
# ============================================================
def route_node(state: RAGState) -> RAGState:
    prompt = ROUTE_PROMPT.format(history_context=memory.get_full_context(), query=state["query"])
    try:
        decision = router_llm.invoke(prompt)
        state["route"] = decision.route
    except Exception as e:
        print(f"WARNING: routing failed ({e}), defaulting to RAG.")
        state["route"] = "RAG"   # safe default — RAG path still has its own fallback if wrong
    return state


def route_router(state: RAGState) -> str:
    return "expand" if state["route"] == "RAG" else "chitchat"


def chitchat_node(state: RAGState) -> RAGState:
    """Handles greetings/small talk — still has full memory access."""
    prompt = f"""You are PoBot, a friendly assistant for Hong Kong labor \
regulations. Respond naturally and briefly to this message. You may reference \
the conversation below if relevant.

Conversation so far:
{memory.get_full_context()}

User's message: "{state['query']}"

Response:"""
    result = llm.invoke(prompt)
    state["response"] = result.content.strip()
    return state


def expand_node(state: RAGState) -> RAGState:
    """Rewrites the query into a standalone, retrieval-optimized query,
    using full conversation history to catch related-topic continuity
    (e.g. maternity leave -> sick leave follow-up)."""
    prompt = EXPAND_PROMPT.format(history_context=memory.get_full_context(), query=state["query"])
    result = llm.invoke(prompt)
    rewritten = result.content.strip()
    state["expanded_query"] = rewritten if rewritten else state["query"]
    return state


def retrieve_node(state: RAGState) -> RAGState:
    query_to_use = state["expanded_query"] or state["query"]
    candidates = hybrid_retrieve(query_to_use, top_k=12, alpha=0.5)
    state["retrieved_chunks"] = rerank_chunks(query_to_use, candidates, top_n=4)
    return state

def crag_node(state: RAGState) -> RAGState:
    chunks = state["retrieved_chunks"]
    if not chunks:
        state["crag_verdict"] = "not_relevant"
        state["crag_retry_count"] += 1
        return state

    topic_hint = ", ".join(sorted({t for c in chunks for t in c.get("topic_tag", [])}))
    context = "\n\n---\n\n".join(
        f"[{c['heading_text']}] {c['content'][:800]}" for c in chunks
    )
    prompt = CRAG_PROMPT.format(
        original_query=state["query"],
        context=context,
        topic_tags=topic_hint or "none detected",
    )

    try:
        verdict = crag_llm.invoke(prompt)
    except Exception as e:
        print(f"WARNING: CRAG grading failed ({e}), treating as not relevant.")
        state["crag_verdict"] = "not_relevant"
        state["crag_retry_count"] += 1
        return state

    if verdict.verdict in ("USABLE", "PARTIALLY_USABLE"):
        state["crag_verdict"] = "relevant"
        return state

    # NOT_USABLE — guarantee the retry actually changes something
    state["crag_verdict"] = "not_relevant"
    state["crag_retry_count"] += 1

    current_query = state.get("expanded_query", "") or state["query"]
    new_query = (verdict.rewritten_query or "").strip()

    if new_query and new_query.lower() != current_query.lower():
        state["expanded_query"] = new_query
    else:
        # Model gave nothing useful or repeated the same query —
        # fall back to the raw original question instead of looping
        # on an already-failed expanded/rewritten version.
        state["expanded_query"] = state["query"]

    return state


def crag_router(state: RAGState) -> str:
    if state["crag_verdict"] == "relevant":
        return "generate"
    if state["crag_retry_count"] < MAX_RETRIES:
        return "retrieve"
    return "fallback"


def generate_node(state: RAGState) -> RAGState:
    if not state["retrieved_chunks"]:
        state["response"] = ("I don't have reliable information on this. "
                              "Please consult the Hong Kong Labour Department directly.")
        return state
    context = "\n\n---\n\n".join(
        f"[{c['heading_text']}] {c['content']}" for c in state["retrieved_chunks"]
    )
    prompt = GENERATE_PROMPT.format(
        memory_summary=memory.get_full_context(), context=context, question=state["query"]
    )
    result = llm.invoke(prompt)
    state["response"] = result.content.strip()
    return state


def fallback_node(state: RAGState) -> RAGState:
    state["response"] = ("I don't have reliable information to answer this confidently. "
                          "Please consult the Hong Kong Labour Department directly, or seek "
                          "advice from a labor rights NGO.")
    return state


# ============================================================
# GRAPH ASSEMBLY
# ============================================================
graph = StateGraph(RAGState)
graph.add_node("route", route_node)
graph.add_node("chitchat", chitchat_node)
graph.add_node("expand", expand_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("crag", crag_node)
graph.add_node("generate", generate_node)
graph.add_node("fallback", fallback_node)

graph.add_edge(START, "route")
graph.add_conditional_edges("route", route_router, {"expand": "expand", "chitchat": "chitchat"})
graph.add_edge("expand", "retrieve")
graph.add_edge("retrieve", "crag")
graph.add_conditional_edges("crag", crag_router,
    {"generate": "generate", "retrieve": "retrieve", "fallback": "fallback"})
graph.add_edge("generate", END)
graph.add_edge("fallback", END)
graph.add_edge("chitchat", END)

app = graph.compile()


# ============================================================
# CLI
# ============================================================
def run_cli():
    print("PoBot — Hong Kong Labor Regulations Assistant")
    print("Type your question, or 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if not query:
            continue

        memory.add_message("user", query)

        state: RAGState = {
            "query": query, "expanded_query": "", "retrieved_chunks": [],
            "response": "", "crag_retry_count": 0, "crag_verdict": "", "route": "",
        }
        result = app.invoke(state)
        print(f"\nPoBot: {result['response']}\n")

        memory.add_message("assistant", result["response"])


if __name__ == "__main__":
    run_cli()