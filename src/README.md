# src/, top level files

`main.py`: the local dev CLI. `chat` is the only subcommand -- an interactive REPL over the same stateless `answer_question(question, history)` that `POST /chat` calls, keeping its own `history` list and passing it back on every turn.

`visualize_graphs.py`: generates PNG image files of the LangGraph graphs in this project (the grading graph, the chatbot graph). Uses LangGraph's built-in `draw_mermaid_png()`, which needs internet access but no local install. Saves images into `logs/`.

`debug_retrieval.py`: a small diagnostic tool. Shows exactly which knowledge-base chunks get retrieved for a given question, with no LLM involved. Useful when the chatbot seems to be missing something that should be in the rules document, since this tells you whether it's a retrieval problem or something else.
