"""
Produces:
    logs/grading_graph.png
    logs/chatbot_graph.png
"""

from pathlib import Path

OUTPUT_DIR = Path("logs")


def save_graph_image(compiled_graph, filename: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    png_bytes = compiled_graph.get_graph().draw_mermaid_png()
    path.write_bytes(png_bytes)
    print(f"Saved {path}")


def main():
    from src.agent.graph import build_grading_graph
    from src.chatbot.graph import build_chat_graph

    save_graph_image(build_grading_graph(), "grading_graph.png")
    save_graph_image(build_chat_graph(), "chatbot_graph.png")


if __name__ == "__main__":
    main()