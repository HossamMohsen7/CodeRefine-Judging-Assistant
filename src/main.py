"""
The single entry point for running this project locally.

Usage:
    poetry run python -m src.main chat
"""

import argparse

from dotenv import load_dotenv

load_dotenv()  # reads .env so GROQ_API_KEY / GITHUB_TOKEN are available


def run_chat() -> None:
    """
    Local manual testing only -- the real chat surface is POST /chat
    (src/api/main.py). This REPL keeps its own history list and passes it
    on every call, exactly like victoris-backend will, since the chatbot
    itself remembers nothing between calls.
    """
    from src.chatbot.chatbot import answer_question

    history: list[dict] = []
    print("Team support chatbot. Type 'exit' to quit.\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        answer = answer_question(question, history=history)
        print(f"A: {answer}\n")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeRefine AI assistant -- local dev CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("chat", help="Start the team support chatbot")

    args = parser.parse_args()
    if args.command == "chat":
        run_chat()


if __name__ == "__main__":
    main()
