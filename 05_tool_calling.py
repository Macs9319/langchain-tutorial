"""
Tutorial 05 — Tool calling
==========================

A model can only produce text. It can't check today's date, do exact maths
or look up your data. With *tool calling* you describe Python functions to
the model; when it needs one, it replies with a request ("call get_weather
with city='Lisbon'"), your code runs the function, and you send the result
back so the model can finish its answer.

What you'll learn:
  1. Turn Python functions into tools with the @tool decorator
  2. Give tools to the model with .bind_tools()
  3. Read the model's tool requests from response.tool_calls
  4. Run the tools and send results back as ToolMessages
  5. Loop until the model has a final answer (a minimal "agent")

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 05_tool_calling.py
  uv run 05_tool_calling.py "Is it warmer in Oslo or Cairo, and by how many degrees?"
"""

import os
import sys
from datetime import date

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Step 1: Define tools
# ---------------------------------------------------------------------------
# @tool reads the function name, type hints and docstring and turns them into
# a schema the model can see. The docstring is the model's only guide to
# *when* and *how* to use the tool, so write it carefully.

# Fake data so the tutorial works offline from any real weather service.
FAKE_WEATHER = {"lisbon": 22, "oslo": 4, "cairo": 31, "tokyo": 17, "london": 11}


@tool
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for a city."""
    temp = FAKE_WEATHER.get(city.strip().lower())
    if temp is None:
        return f"No weather data for {city}. Known cities: {', '.join(FAKE_WEATHER)}."
    return f"{temp}°C in {city}"


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(31 - 4) * 1.8'.

    Supports + - * / and parentheses. Use this for any maths instead of
    working it out yourself.
    """
    allowed = set("0123456789+-*/(). ")
    # Reject ** too: something like 9**9**9**9 would hang the program.
    if not set(expression) <= allowed or "**" in expression:
        return "Error: only numbers, + - * / and parentheses are allowed."
    try:
        # Safe here because of the character whitelist above. Never eval()
        # arbitrary model output without restricting it like this.
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as error:  # tell the model what went wrong so it can retry
        return f"Error: {error}"


@tool
def today() -> str:
    """Return today's date in ISO format (YYYY-MM-DD)."""
    return date.today().isoformat()


TOOLS = [get_weather, calculate, today]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def show_tools() -> None:
    print("=== Step 1: tools the model will see ===")
    for t in TOOLS:
        print(f"- {t.name}{t.args}: {t.description.splitlines()[0]}")
    # Tools are Runnables, so you can test them directly without a model:
    print("direct test:", get_weather.invoke({"city": "Lisbon"}), "\n")


# ---------------------------------------------------------------------------
# Step 2 & 3: Bind tools and inspect the model's request
# ---------------------------------------------------------------------------
llm_with_tools = llm.bind_tools(TOOLS)


def single_tool_request() -> None:
    print("=== Steps 2-3: what the model asks for ===")
    response = llm_with_tools.invoke("What's the weather in Tokyo?")
    # The model doesn't run anything. It returns tool_calls for us to run.
    print("text      :", repr(response.text))
    for call in response.tool_calls:
        print("tool call :", call["name"], call["args"], f"(id={call['id']})")
    print()


# ---------------------------------------------------------------------------
# Step 4 & 5: Run tools, send results back, repeat
# ---------------------------------------------------------------------------
# Invoking a tool with the whole tool_call dict returns a ToolMessage that
# carries the matching tool_call_id, so the model knows which result is which.
def run_agent(question: str, max_steps: int = 5, verbose: bool = True) -> str:
    messages: list[BaseMessage] = [
        SystemMessage("You are a helpful assistant. Use the tools when they help."),
        HumanMessage(question),
    ]
    for _ in range(max_steps):  # a cap so a confused model can't loop forever
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:  # no more tool requests -> final answer
            return response.text

        for call in response.tool_calls:
            result = TOOLS_BY_NAME[call["name"]].invoke(call)  # -> ToolMessage
            messages.append(result)
            if verbose:
                print(f"  -> {call['name']}({call['args']}) = {result.content}")

    return "Stopped: too many tool calls without a final answer."


def agent_demo(question: str) -> None:
    print("=== Steps 4-5: the tool loop ===")
    print("question:", question)
    answer = run_agent(question)
    print("answer  :", answer, "\n")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "What's today's date, and how much warmer is Cairo than Oslo "
        "right now in Fahrenheit degrees?"
    )

    show_tools()
    single_tool_request()
    agent_demo(question)
