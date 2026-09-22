"""
The team chatbot, as a small LangGraph graph. Stateless by design: the
caller (victoris-backend) supplies whatever conversation history it wants
remembered on each call -- see
docs/specs/2026-09-22-coderefine-judging-platform.md, POST /chat.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import re

import json

from src.chatbot.state import ChatState
from src.chatbot.prompts import CHATBOT_SYSTEM_PROMPT
from src.ingestion.retriever import load_retriever
from src.agent.llm import get_llm

_retriever = None

MAX_CHARS_PER_CHUNK = 800

_PROCEDURAL_KEYWORDS = [
    "regist", "sign up", "sign-up", "signup", "deadline", "apply",
    "application", "submit", "submission process", "contact form",
    "portal", "fee", "enroll", "how to join",
    # Arabic equivalents, since this chatbot has been tested bilingually
    "تسجيل", "التسجيل", "التقديم", "الاشتراك", "موعد نهائي",
]

_GENERAL_SECTION_PATTERN = re.compile(
    r"\*\*General (?:guidance|explanation)[^\n]*\*\*.*?(?=\n\*\*|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _strip_speculative_procedure_sections(answer: str) -> str:
    def maybe_strip(match):
        block = match.group(0)
        if any(keyword.lower() in block.lower() for keyword in _PROCEDURAL_KEYWORDS):
            return ""  # drop the whole speculative block
        return block  # genuine concept explanations pass through untouched

    return _GENERAL_SECTION_PATTERN.sub(maybe_strip, answer).strip()


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = load_retriever()
    return _retriever


def _grade_relevance(question: str, docs: list) -> list:
    """
    Retrieval finds chunks that are SIMILAR in meaning to the question
    that's not the same as chunks that actually ANSWER it. A compound or
    oddly-phrased question can retrieve chunks that are topically related
    but don't directly address what was asked. This asks the LLM to check
    each retrieved chunk against the actual question before any of them
    get used to answer filtering out anything that doesn't genuinely
    help, instead of assuming "retrieved" means "relevant."
    """
    if not docs:
        return []

    numbered = "\n\n".join(
        f"[{i}] Section: {d.metadata['section']}\n{d.page_content[:MAX_CHARS_PER_CHUNK]}"
        for i, d in enumerate(docs)
    )
    prompt = f"""Question: {question}

Below are numbered sections retrieved for this question. For EACH one,
decide: could this section plausibly help answer the question, even if
the wording is different from how the question was phrased? Be
INCLUSIVE, not strict a paraphrase, a synonym, or a section that
covers part of the answer should be marked relevant. Only exclude a
section if it is genuinely about something else entirely.

{numbered}

Return ONLY a JSON list of the indices (integers) of sections that could
plausibly help. Return an empty list [] only if NONE of them are
remotely related to the question. Example: [0, 2]"""

    llm = get_llm(temperature=0.0)  # deterministic for a grading/filtering task
    response = llm.invoke(prompt)
    try:
        relevant_indices = json.loads(response.content)
        return [docs[i] for i in relevant_indices if isinstance(i, int) and 0 <= i < len(docs)]
    except (json.JSONDecodeError, TypeError):
        # If grading itself fails to parse, fail toward including everything
        # retrieved rather than silently answering from nothing.
        return docs


def chat_node(state: ChatState) -> dict:
    """
    The only node in this graph. Runs once per call: retrieves relevant
    rules for the LATEST message, but sends the FULL message list (caller-
    supplied history + the new question) to the LLM that combination is
    what lets it answer follow-ups correctly while still grounding answers
    in retrieved rules.
    """
    latest_question = state["messages"][-1].content

    retriever = _get_retriever()
    docs = retriever.invoke(latest_question)
    relevant_docs = _grade_relevance(latest_question, docs)

    if not relevant_docs and docs:
        relevant_docs = docs[:1]

    if relevant_docs:
        context = "\n\n".join(
            f"[{d.metadata['section']}]\n{d.page_content[:MAX_CHARS_PER_CHUNK]}" for d in relevant_docs
        )
    else:
        context = ""

    system_message = SystemMessage(content=CHATBOT_SYSTEM_PROMPT.format(context=context))
    llm = get_llm(temperature=0.2)

    response = llm.invoke([system_message] + state["messages"])
    response.content = _strip_speculative_procedure_sections(response.content)
    return {"messages": [response]}


def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    # No checkpointer: the caller supplies whatever history it wants
    # remembered on every call, instead of this service remembering it.
    return graph.compile()


_chat_app = None


def _get_chat_app():
    global _chat_app
    if _chat_app is None:
        _chat_app = build_chat_graph()
    return _chat_app


def ask(question: str, history: list[dict] | None = None) -> str:
    """
    history: prior turns as [{"role": "user" | "assistant", "content": str}, ...],
    oldest first -- exactly what POST /chat's request body carries. No
    thread_id: this call is stateless, all context comes from history.
    """
    messages = []
    for turn in history or []:
        message_cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(message_cls(content=turn["content"]))
    messages.append(HumanMessage(content=question))

    app = _get_chat_app()
    result = app.invoke({"messages": messages})
    return result["messages"][-1].content
