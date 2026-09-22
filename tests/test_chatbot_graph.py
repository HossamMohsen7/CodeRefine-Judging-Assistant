from langchain_core.messages import AIMessage, HumanMessage

from src.chatbot import graph


class _FakeDoc:
    def __init__(self, section, content):
        self.metadata = {"section": section}
        self.page_content = content


class _FakeRetriever:
    def invoke(self, _question):
        return [_FakeDoc("Team Size", "Teams must have 3 to 5 members.")]


class _FakeRelevanceLlm:
    def invoke(self, _prompt):
        class _Response:
            content = "[0]"
        return _Response()


class _FakeChatLlm:
    def __init__(self):
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return AIMessage(content="Teams must have 3 to 5 members.")


def test_ask_sends_supplied_history_to_the_llm(monkeypatch):
    fake_chat_llm = _FakeChatLlm()
    llms = iter([_FakeRelevanceLlm(), fake_chat_llm])

    monkeypatch.setattr(graph, "_get_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(graph, "get_llm", lambda **_kwargs: next(llms))

    history = [
        {"role": "user", "content": "What is CodeRefine?"},
        {"role": "assistant", "content": "A system-design track."},
    ]

    answer = graph.ask("How many members per team?", history=history)

    assert answer == "Teams must have 3 to 5 members."
    sent_human_messages = [m for m in fake_chat_llm.received_messages if isinstance(m, HumanMessage)]
    assert [m.content for m in sent_human_messages] == [
        "What is CodeRefine?", "How many members per team?",
    ]
