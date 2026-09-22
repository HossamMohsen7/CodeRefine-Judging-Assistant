"""
Thin backward-compatible entry point. The real logic lives in graph.py.
"""

from src.chatbot.graph import ask


def answer_question(question: str, history: list[dict] | None = None) -> str:
    return ask(question, history=history)


if __name__ == "__main__":
    # Loading .env here specifically, because running this file directly
    # (poetry run python -m src.chatbot.chatbot) skips main.py entirely --
    # and main.py is normally what loads the .env file. Any file with its
    # own standalone test block needs to handle this itself.
    from dotenv import load_dotenv
    load_dotenv()

    # Quick manual test -- two related questions, second one is a follow-up
    # that only makes sense when the caller passes the first turn back in
    # as history (this service no longer remembers it on its own).
    first_question = "How many members can be on a team?"
    first_answer = answer_question(first_question)
    print(f"Q: {first_question}")
    print(f"A: {first_answer}\n")

    second_question = "And what age do they need to be?"
    history = [
        {"role": "user", "content": first_question},
        {"role": "assistant", "content": first_answer},
    ]
    print(f"Q: {second_question}")
    print(f"A: {answer_question(second_question, history=history)}")
