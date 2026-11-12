# LangChain Tutorial Series

Short, runnable Python scripts that go from a first chat call to a
multi-agent system with automated evaluation. Each tutorial builds on the
previous ones, and every file explains itself in comments, so you can read
them top to bottom.

All tutorials use OpenAI models through [LangChain](https://python.langchain.com)
and [LangGraph](https://langchain-ai.github.io/langgraph/).

## Setup

You need [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
uv sync                      # install dependencies into .venv
cp .env.example .env         # then put your key after OPENAI_API_KEY=
```

Optional settings in `.env`:

| Variable | Default | Used by |
|---|---|---|
| `OPENAI_MODEL` | `gpt-5-mini` | all tutorials |
| `JUDGE_MODEL` | same as `OPENAI_MODEL` | 10 (the grading model) |
| `FALLBACK_MODEL` | `gpt-5-nano` | 16 (the backup model) |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` | off | all (see Tutorial 10) |

## The tutorials

| # | File | You learn | Run |
|---|---|---|---|
| 01 | [`01_chat_basics.py`](01_chat_basics.py) | Create a chat model, send prompts and message lists, read token usage, stream replies | `uv run 01_chat_basics.py "your question"` |
| 02 | [`02_prompts_and_chains.py`](02_prompts_and_chains.py) | Prompt templates with `{variables}`, output parsers, chains with `\|`, `.batch()` | `uv run 02_prompts_and_chains.py "black holes"` |
| 03 | [`03_structured_output.py`](03_structured_output.py) | Get typed Python objects back with Pydantic and `.with_structured_output()` | `uv run 03_structured_output.py "Pad Thai"` |
| 04 | [`04_conversation_memory.py`](04_conversation_memory.py) | Why models forget, keeping history, per-user sessions, trimming long chats | `uv run 04_conversation_memory.py --chat` |
| 05 | [`05_tool_calling.py`](05_tool_calling.py) | Let the model call your Python functions; write the agent loop by hand | `uv run 05_tool_calling.py` |
| 06 | [`06_rag.py`](06_rag.py) | Answer from your own documents: split, embed, retrieve, cite sources | `uv run 06_rag.py "How long is the battery warranty?"` |
| 07 | [`07_langgraph_agent.py`](07_langgraph_agent.py) | Rebuild the agent as a LangGraph graph; memory that survives restarts (SQLite) | `uv run 07_langgraph_agent.py --chat` |
| 08 | [`08_human_in_the_loop.py`](08_human_in_the_loop.py) | Pause the agent for a person to approve, edit or reject an action | `uv run 08_human_in_the_loop.py --chat` |
| 09 | [`09_multi_agent.py`](09_multi_agent.py) | A supervisor that routes work to specialist agents | `uv run 09_multi_agent.py` |
| 10 | [`10_evaluation.py`](10_evaluation.py) | Test sets, rule checks, LLM-as-judge, cost tracking, comparing variants | `uv run 10_evaluation.py` |
| 11 | [`11_async_streaming.py`](11_async_streaming.py) | Async calls with `asyncio.gather`, `astream_events`, streaming an agent's tokens, tool calls and progress | `uv run 11_async_streaming.py` |
| 12 | [`12_fastapi_server.py`](12_fastapi_server.py) | Serve the agent with FastAPI: JSON and streaming (SSE) endpoints, per-thread memory, a browser chat page | `uv run 12_fastapi_server.py`, then open http://127.0.0.1:8001 |
| 13 | [`13_create_agent_middleware.py`](13_create_agent_middleware.py) | The high-level `create_agent()`, typed agent results, built-in middleware (PII redaction, call limits, summarization, human approval) and writing your own | `uv run 13_create_agent_middleware.py` |
| 14 | [`14_mcp_tools.py`](14_mcp_tools.py) | Connect to MCP servers: discover tools, read resources and prompts, give MCP tools to an agent. Includes a small server in [`mcp_servers/`](mcp_servers/) | `uv run 14_mcp_tools.py` |
| 15 | [`15_long_term_memory.py`](15_long_term_memory.py) | Remember users across conversations with a store: memory tools, semantic search, automatic recall, SQLite persistence, letting users see and delete memories | `uv run 15_long_term_memory.py --chat --user sam` |
| 16 | [`16_reliability.py`](16_reliability.py) | Survive failures and control cost: retryable errors, `.with_retry()`, fallback models, rate limiting, caching, retry/fallback middleware for agents | `uv run 16_reliability.py` |

Run any tutorial without arguments to see its built-in demos. Scripts with
`--chat` also have an interactive mode; type `quit` to leave it.

## How they connect

```
01 chat ─► 02 chains ─► 03 structured output ─────────────┐
              │                                           │
              ├─► 04 memory ──────┐                       │
              ├─► 05 tools ───────┼─► 07 LangGraph ─► 08 human approval
              └─► 06 RAG ─────────┘        │              │
                    │                      └─► 09 multi-agent (uses 03 for routing)
                    └──────────────────────────► 10 evaluation
```

## Sample data

Tutorials 06–10 use the documents in [`data/`](data/), which describe
**Pedalwise Bikes**, a made-up bicycle shop: its returns policy, warranty and
servicing prices. Because the shop is fictional, the model can't answer from
training data, which makes it obvious when retrieval works. Tutorials 08 and
09 also use a small fake order database defined in the scripts.

## Files the tutorials create

- `memory.sqlite*`: the saved conversation from Tutorial 07's `--chat`. Delete
  it, or run with `--chat --new`, to start over.
- `long_term_memory.sqlite*`: facts Tutorial 15's `--chat` remembers about each
  user. List them with `--show --user NAME`; delete the file to forget everything.

They're listed in `.gitignore` along with `.env`, so neither they nor your API
key get committed.

## Cost

Every run calls the OpenAI API and is billed to your account. Tutorials 01–09
make a handful of calls each. Tutorial 10 makes roughly 50 (8 questions × 3
variants, plus grading each answer). Tutorials 06, 07, 09 and 10 also call the
embeddings API once per run to index the three small sample documents.
