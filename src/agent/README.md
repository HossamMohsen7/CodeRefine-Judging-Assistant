# src/agent/

The grading agent. Reads a submitted repo and produces a scored, evidence-cited draft -- always through the same three-step pipeline, for every trial.

## Files

`rubric.py`: the five real judging criteria and their weights (15/20/20/25/20). No functions, just data. Bonus is scored separately by a judge in victoris-dashboard, never by this agent.

`llm.py`: creates the Groq clients. `get_llm()` is the text client for grading and chat. `get_vision_llm()` is a separate Qwen vision client used only to make factual observations about committed image/PDF diagrams. Both use `GROQ_API_KEY`.

`state.py`: `GradingState`, the shape of data that flows through the pipeline -- what's known at the start (`repo_url`), what `gather` fills in (`raw_notes`, and `repo_error` if the repo couldn't be reached at all), what `format` fills in (`draft_scorecard`), what `verify` fills in (`final_scorecard`, `verification_notes`). No logic in this file, just the data shape.

`nodes.py`: the actual reasoning logic, as three functions:
- `gather_node`, reads the repo's README, Deep Dives file, BOTE file, and architecture diagram, and produces raw observations (no scoring). Committed PNG/JPG/WebP diagrams and the first five PDF pages add factual vision evidence; the Excalidraw parser handles `.excalidraw` files. Includes prompt-injection defenses, since this content is written by the team being graded and shouldn't be trusted as instructions. Sets `repo_error` in its returned dict if the repo itself couldn't be reached.
- `format_node`, turns those observations into a scored draft, citing evidence by ID rather than copied text so verification doesn't break on paraphrasing.
- `verify_node`, a plain code check (no LLM call) that confirms cited evidence actually exists, and flags any justification that looks like it invented a scoring threshold not in the rubric.

`graph.py`: wires the three functions above into one runnable LangGraph flow. `grade_repo(repo_url)` is the one function `POST /grade` calls -- runs the pipeline, returns the full final state. `format_grade_response(final_scorecard, verification_notes)` is a separate pure function that reshapes that internal state into `POST /grade`'s exact `{score, feedback, evidence}` response.

## Nothing in this folder runs on its own

Everything here gets called from `src/api/main.py` (or `src/visualize_graphs.py` for diagram generation). There's no reason to run any of these files directly, except `state.py`, `rubric.py`, and `llm.py`, which aren't runnable at all since they define things other files use.
