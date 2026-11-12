"""
Tutorial 14 — Tools from MCP servers
====================================

So far every tool was a Python function inside the tutorial itself. MCP
(Model Context Protocol) is an open standard that lets tools live in
separate *servers* that any AI app can connect to. Thousands of MCP servers
already exist (GitHub, Slack, databases, file systems, browsers...), so
instead of writing an integration yourself, you connect to a server.

This tutorial connects to mcp_servers/pedalwise_server.py, a small server
in this repo, so it works offline from any third-party service.

What you'll learn:
  1. Connect to an MCP server with MultiServerMCPClient
  2. Discover its tools, and call one directly
  3. Read its resources and fill in its prompts
  4. Give MCP tools to an agent built with create_agent (Tutorial 13)
  5. Keep one session open for many calls, and what to watch for security-wise

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).
  Steps 1-3 and 5's session demo make no OpenAI calls.

Run:
  uv run 14_mcp_tools.py
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

load_dotenv()

HERE = Path(__file__).parent

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)

# ---------------------------------------------------------------------------
# Step 1: Describe how to reach each server
# ---------------------------------------------------------------------------
# "stdio" means: start the server as a subprocess and talk over its
# stdin/stdout. sys.executable is the Python running this script, so the
# server gets the same installed packages.
#
# Other servers are added the same way. For example (not run here):
#   "files": {"transport": "stdio", "command": "npx",
#             "args": ["-y", "@modelcontextprotocol/server-filesystem", "/some/folder"]},
#   "remote": {"transport": "streamable_http", "url": "https://example.com/mcp",
#              "headers": {"Authorization": "Bearer ..."}},
SERVERS = {
    "pedalwise": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(HERE / "mcp_servers" / "pedalwise_server.py")],
    },
}

client = MultiServerMCPClient(SERVERS)


def text_of(result) -> str:
    """MCP tools return a list of content blocks, not a plain string, because
    a result can mix text, images and other media. Join the text parts."""
    if isinstance(result, str):
        return result
    return " ".join(block["text"] for block in result if block.get("type") == "text")


# ---------------------------------------------------------------------------
# Step 2: Discover and call tools
# ---------------------------------------------------------------------------
# get_tools() asks every server what it offers and wraps each tool as a
# normal LangChain tool. MCP is async, so these tools are called with await
# (tool.ainvoke, agent.ainvoke).
async def discover_tools() -> list:
    print("=== Step 2: tools discovered from the server ===")
    tools = await client.get_tools()
    for t in tools:
        print(f"- {t.name}{list(t.args)}: {t.description.splitlines()[0]}")

    check_stock = next(t for t in tools if t.name == "check_stock")
    result = await check_stock.ainvoke({"item": "Trail 500", "store": "Porto"})
    print("raw result :", result)
    print("as text    :", text_of(result))
    print()
    return tools


# ---------------------------------------------------------------------------
# Step 3: Resources and prompts
# ---------------------------------------------------------------------------
# Resources are data your *code* fetches (the model doesn't call them like
# tools). They come back as Blobs: raw data plus a mime type and source URI.
# Prompts come back as ready-to-send LangChain messages.
async def resources_and_prompts() -> None:
    print("=== Step 3: resources and prompts ===")
    hours = await client.get_resources("pedalwise", uris="pedalwise://opening-hours")
    print("opening hours:", hours[0].as_string())

    warranty = await client.get_resources("pedalwise", uris="pedalwise://policies/warranty")
    print("warranty doc :", warranty[0].as_string().splitlines()[0], "...")

    messages = await client.get_prompt("pedalwise", "apology_email",
                                       arguments={"customer_name": "Sam",
                                                  "problem": "a late delivery"})
    print("prompt       :", messages[0].content, "\n")
    # You could now send `messages` to a model: await llm.ainvoke(messages)


# ---------------------------------------------------------------------------
# Step 4: MCP tools in an agent
# ---------------------------------------------------------------------------
# To the agent, MCP tools are ordinary tools: it sees their names and
# descriptions and calls them as needed.
async def agent_demo(tools: list) -> None:
    print("=== Step 4: an agent using MCP tools ===")
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt="You are Pedalwise Bikes support. Use the tools. "
                      "Answer in one or two short sentences.",
    )
    for question in ["Where is my order A1003?",
                     "Is the City e-bike in stock in Lisbon or Porto?"]:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        used = [c["name"] for m in result["messages"] for c in getattr(m, "tool_calls", [])]
        print(f"customer : {question}\nassistant: {result['messages'][-1].text}\n"
              f"  (tools used: {', '.join(used) or 'none'})")
    print()


# ---------------------------------------------------------------------------
# Step 5: One session for many calls
# ---------------------------------------------------------------------------
# Tools from client.get_tools() open a fresh connection for every call. For a
# stdio server that means starting a new process each time: simple, but
# slow, and the server can't remember anything between calls. Opening a
# session keeps one connection (one process) for everything inside the block.
async def session_demo() -> None:
    print("=== Step 5: a persistent session ===")
    async with client.session("pedalwise") as session:
        tools = await load_mcp_tools(session)  # these tools reuse `session`
        get_order = next(t for t in tools if t.name == "get_order")
        for order_id in ["A1001", "A1002", "A1003"]:
            print(" ", text_of(await get_order.ainvoke({"order_id": order_id})))
    print()


# A word on safety. An MCP server is code you choose to trust:
#  - A stdio server runs on your computer with your permissions.
#  - A remote server receives whatever the model sends to its tools.
#  - Tool descriptions are written by the server's author and go straight
#    into your model's prompt, so a malicious server could try to steer it.
# Use servers from sources you trust, give them only the access they need,
# and combine sensitive tools with human approval (Tutorials 08 and 13).


async def main() -> None:
    tools = await discover_tools()
    await resources_and_prompts()
    await agent_demo(tools)
    await session_demo()


if __name__ == "__main__":
    asyncio.run(main())
