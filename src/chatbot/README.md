# src/chatbot/

The team support chatbot. Answers questions using only the rules document (via `src/ingestion/`) no connection to the grading agent at all. Stateless: every call takes the full conversation history it needs as a parameter, and remembers nothing on its own.

## Files

**`prompts.py`** the system prompt, and it's doing more work than it looks like. Draws a hard line between three things: facts stated in the actual rules (must be quoted accurately, never embellished), general concepts explained from common knowledge (allowed, but must be clearly labeled as not an official rule), and this organization's real procedures like registration or deadlines (never guessed at, even when labeled if it's not in the rules doc, the answer is "not specified, contact the organizers"). Includes worked examples of right vs. wrong answers, since plain instructions alone weren't reliably followed in testing.

**`state.py`** defines the per-call message shape. One field, `messages`, using LangGraph's `add_messages` so new messages get appended within a single graph invocation instead of replacing it.

**`graph.py`**: the actual chatbot logic, in order: retrieve candidate chunks for the latest question, grade each one for genuine relevance to the question (a separate LLM call, since a chunk being retrieved because it's topically similar isn't the same as it actually answering what was asked), then answer using only chunks that passed grading. If grading rejects everything, the top raw retrieval result is used as a fallback rather than leaving the answer with nothing. `ask(question, history)` builds the message list from the caller-supplied `history` (oldest first) plus the new question, and sends the whole thing to the LLM alongside the retrieved context. Also caps how much of any single retrieved chunk gets used, so one oversized chunk can't blow through the LLM's rate limit by itself.

**`chatbot.py`** a thin wrapper exposing `answer_question(question, history=None)`, which is what everything else (the CLI, the API) actually calls. Also runnable directly for a quick manual test.

## No memory of its own

`ask()`/`answer_question()` take `history` as a plain argument -- a list of `{"role": "user" | "assistant", "content": str}` dicts, oldest first. There is no `thread_id` and nothing is persisted here: the caller (victoris-backend) is responsible for storing every message and passing back whatever it wants remembered on the next call.
