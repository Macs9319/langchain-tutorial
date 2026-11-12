"""
Tutorial 04 — Conversation memory
=================================

Chat models are *stateless*: every call starts from zero. The model only
"remembers" earlier turns because you send them again with each request.
This tutorial shows how to keep and resend that history.

What you'll learn:
  1. See that the model forgets without history
  2. Keep history yourself with a list + MessagesPlaceholder
  3. Keep a separate history per user with a session store
  4. Keep long conversations bounded with trim_messages
  5. Put it together in an interactive chatbot (--chat)

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 04_conversation_memory.py          # run the demos
  uv run 04_conversation_memory.py --chat   # chat interactively (type 'quit' to exit)
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)

# MessagesPlaceholder("history") is a slot in the template that accepts a
# whole list of messages: the earlier turns of the conversation.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly assistant. Keep answers short."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
chain = prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Step 1: Without history, the model forgets
# ---------------------------------------------------------------------------
def no_memory() -> None:
    print("=== Step 1: no memory ===")
    print("you: My name is Sam and I love hiking.")
    print("ai :", chain.invoke({"history": [], "input": "My name is Sam and I love hiking."}))
    print("you: What's my name?")
    print("ai :", chain.invoke({"history": [], "input": "What's my name?"}), "\n")


# ---------------------------------------------------------------------------
# Step 2: Keep history yourself
# ---------------------------------------------------------------------------
# After each turn, append the user's message and the reply to a list, then
# pass the list back in next time. That is all "memory" is.
def manual_memory() -> None:
    print("=== Step 2: manual history list ===")
    history: list[BaseMessage] = []
    for text in ["My name is Sam and I love hiking.",
                 "What's my name?",
                 "Suggest a weekend activity for me."]:
        reply = chain.invoke({"history": history, "input": text})
        history += [HumanMessage(text), AIMessage(reply)]
        print(f"you: {text}\nai : {reply}")
    print(f"(history now holds {len(history)} messages)\n")


# ---------------------------------------------------------------------------
# Step 3: A separate history per session
# ---------------------------------------------------------------------------
# A real app talks to many users at once, each with their own conversation.
# Store one history list per session ID, load it before each call and save
# the new turn after. Swap the dict for a database (Redis, SQL, ...) to make
# memory survive restarts.
store: dict[str, list[BaseMessage]] = {}


def say(session_id: str, text: str) -> str:
    history = store.setdefault(session_id, [])
    reply = chain.invoke({"history": history, "input": text})
    history += [HumanMessage(text), AIMessage(reply)]
    return reply


def session_memory() -> None:
    print("=== Step 3: separate sessions ===")
    say("alice", "Hi, I'm Alice. My favourite colour is green.")
    say("bob", "Hi, I'm Bob. My favourite colour is orange.")
    print("alice asks:", say("alice", "What's my name and favourite colour?"))
    print("bob asks  :", say("bob", "What's my name and favourite colour?"))
    print(f"(alice's session holds {len(store['alice'])} messages)\n")


# ---------------------------------------------------------------------------
# Step 4: Trim long histories
# ---------------------------------------------------------------------------
# History grows every turn, and so does the cost of each request. Eventually
# it would exceed the model's context window. trim_messages keeps only the
# most recent messages that fit within a token budget.
trimmer = trim_messages(
    max_tokens=60,                # tiny on purpose so the effect is visible
    strategy="last",              # keep the most recent messages
    token_counter="approximate",  # fast estimate; pass `llm` for exact counts
    start_on="human",             # never start the history on an AI reply
)


def trimmed_memory() -> None:
    print("=== Step 4: trimming ===")
    history: list[BaseMessage] = [
        HumanMessage("My name is Sam."),
        AIMessage("Nice to meet you, Sam!"),
        HumanMessage("I live in Lisbon."),
        AIMessage("Lisbon is a lovely city."),
        HumanMessage("I have a dog called Pixel."),
        AIMessage("Pixel is a great name for a dog!"),
    ]
    kept = trimmer.invoke(history)
    print(f"kept {len(kept)} of {len(history)} messages:")
    for message in kept:
        print(f"  [{message.type}] {message.content}")
    # The oldest turns were dropped, so the model can no longer see the name.
    print("ai :", chain.invoke({"history": kept, "input": "What's my name and my dog's name?"}), "\n")


# ---------------------------------------------------------------------------
# Step 5: An interactive chatbot
# ---------------------------------------------------------------------------
# Steps 2-4 combined: keep the full history, send a trimmed copy each turn,
# and stream the reply. The conversation can run forever without the
# request growing without limit.
def interactive_chat() -> None:
    history: list[BaseMessage] = []
    print("Chat started. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        recent = trim_messages(history, max_tokens=2000, strategy="last",
                               token_counter="approximate", start_on="human")
        print("ai : ", end="", flush=True)
        reply = ""
        for piece in chain.stream({"history": recent, "input": text}):
            print(piece, end="", flush=True)
            reply += piece
        print()
        history += [HumanMessage(text), AIMessage(reply)]


if __name__ == "__main__":
    if "--chat" in sys.argv:
        interactive_chat()
    else:
        no_memory()
        manual_memory()
        session_memory()
        trimmed_memory()
