# LangChain Tutorial Series

Short, runnable Python scripts that go from a first chat call to a
multi-agent system with automated evaluation. Each tutorial builds on the
previous ones, and every file explains itself in comments, so you can read
them top to bottom.

The tutorials use [LangChain](https://python.langchain.com) and
[LangGraph](https://langchain-ai.github.io/langgraph/), with OpenAI models by
default. Set `CHAT_MODEL` to run every tutorial on another provider, such as
DeepSeek (Tutorial 23).

## Setup

You need [uv](https://docs.astral.sh/uv/) and an OpenAI API key, or a DeepSeek
key with `CHAT_MODEL=deepseek:deepseek-flash`. With DeepSeek alone, everything
runs except what needs OpenAI's embeddings or PDF reading:

- Tutorials 06, 07, 09, 10, 15 and 19 embed documents with OpenAI, so they
  need an OpenAI key whatever `CHAT_MODEL` says (DeepSeek has no embeddings API).
- Tutorial 17 skips its PDF steps on models that can't read PDFs.
- The capstone (20-22) can use `EMBEDDINGS=none` for keyword-only search.

```bash
uv sync                      # install dependencies into .venv
cp .env.example .env         # then put your key after OPENAI_API_KEY=
```

Optional settings in `.env`:

| Variable | Default | Used by |
|---|---|---|
| `CHAT_MODEL` | `OPENAI_MODEL` | all tutorials (`provider:model`, e.g. `deepseek:deepseek-flash`) |
| `OPENAI_MODEL` | `gpt-5-mini` | all tutorials, when `CHAT_MODEL` isn't set |
| `JUDGE_MODEL` | same as `CHAT_MODEL` | 10 (the grading model; `provider:model`), 25 (default `openai:gpt-5-mini`) |
| `FALLBACK_MODEL` | `gpt-5-nano` | 16, 20-22 (the backup model; `provider:model`, or `none`) |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` | off | all (see Tutorial 10) |
| `PEDALWISE_DEMO_PASSWORD` | none | 20-27 (creates the demo accounts `sam@`, `dana@` and `staff@example.com` with this password, 12+ characters) |
| `STAFF_TOKEN` | none | 20-27 (optional token for scripts calling the staff API; 24+ characters in production) |
| `COOKIE_SECURE` | on in production | 27 (`false` for a local stack on plain http) |
| `PEDALWISE_ENV` | `development` | 21 (`production`: strict checks, JSON logs, no `/docs`) |
| `HOST`, `PORT`, `LOG_LEVEL` | `127.0.0.1`, `8001`, `INFO` | 20, 21 |
| `PEDALWISE_DB`, `PEDALWISE_INDEX_DIR` | files in this folder | 20, 21 (where data is saved) |
| `DATABASE_URL` | none (SQLite at `PEDALWISE_DB`) | 20-26 (`postgresql://...`: a database shared by several copies) |
| `POSTGRES_PASSWORD` | `pedalwise-local` | 26 (`compose.scale.yaml`'s Postgres; set your own for anything real) |
| `DEEPSEEK_API_KEY` | none | any tutorial, with a DeepSeek `CHAT_MODEL` |
| `EMBEDDINGS` | `openai` | 20-22 (`none`: keyword search only, for providers without embeddings) |
| `DEEPSEEK_THINKING` | `false` | all (DeepSeek reasons before answering; slower, more tokens) |
| `TELEMETRY_RETENTION_DAYS` | `30` | 22 (how long runs, with their questions and answers, are kept) |

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
| 17 | [`17_images_and_pdfs.py`](17_images_and_pdfs.py) | Send images and PDFs to the model: describe photos, extract receipt data, question a PDF, give an agent an uploaded photo | `uv run 17_images_and_pdfs.py` |
| 18 | [`18_guardrails.py`](18_guardrails.py) | Defend against prompt injection in layers: rules enforced in code, marking untrusted text, screening inputs and tool results, checking outputs | `uv run 18_guardrails.py` |
| 19 | [`19_advanced_rag.py`](19_advanced_rag.py) | Better retrieval: heading-aware chunks, a saved index, hybrid keyword + vector search, query rewriting, re-ranking, measuring hit rate | `uv run 19_advanced_rag.py` |
| 20 | [`20_capstone.py`](20_capstone.py) | **Capstone:** a complete support app in the [`pedalwise/`](pedalwise/) package: streaming chat, the customer's own orders, hybrid search, staff-approved refunds, saved conversations, and offline tests in [`tests/`](tests/) | set `PEDALWISE_DEMO_PASSWORD`, `uv run 20_capstone.py`, then log in at http://127.0.0.1:8001 and `/staff`; test with `uv run pytest` |
| 21 | [`21_deployment.py`](21_deployment.py) | Deploy the capstone: settings from the environment, failing fast on bad config, a [`Dockerfile`](Dockerfile) and [`compose.yaml`](compose.yaml), state on a volume, health checks, JSON logs, graceful shutdown, smoke tests, and [CI](.github/workflows/ci.yml) | `docker compose up --build`, then `uv run 21_deployment.py --check http://127.0.0.1:8001` |
| 22 | [`22_observability.py`](22_observability.py) | See what the deployed agent does: run ids, a callback that counts model calls, tokens and tools, 👍/👎 feedback tied to each answer, staff metrics (p50/p95, error rate), optional LangSmith traces, and turning complaints into eval cases | `uv run 22_observability.py`; `--report` for the app's numbers |
| 23 | [`23_model_providers.py`](23_model_providers.py) | Add DeepSeek: `init_chat_model` and `provider:model` settings, comparing providers on answers, structured output and tool calls, DeepSeek's thinking mode (and sending its reasoning back), keyword-only search without embeddings, falling back across providers | `uv run 23_model_providers.py` |
| 24 | [`24_text_to_sql.py`](24_text_to_sql.py) | Answer questions from a database: an agent with schema and query tools, guardrails enforced by SQLite itself (read-only, an authorizer for tables, columns and functions, a time limit, a row cap), self-correcting queries, and evaluating by comparing result rows | `uv run 24_text_to_sql.py`; `--guardrails` (no API calls), `--eval` |
| 25 | [`25_model_selection.py`](25_model_selection.py) | Choose the capstone's model on evidence: agent test cases that check behaviour, critical versus quality cases, repeated runs, cost per 1,000 messages from measured tokens, a speed budget, a written-down decision rule, harder multi-turn cases, importing Tutorial 22's thumbs-down cases, and checking the grader before blaming the model | `uv run 25_model_selection.py` |
| 26 | [`26_scaling.py`](26_scaling.py) | Run several copies behind a load balancer: find state kept in one process (and the refund bug it hid), move it to Postgres behind a small SQLite/Postgres layer, let a primary key stop double refunds, start copies safely with a lock, Caddy in [`compose.scale.yaml`](compose.scale.yaml), and checking it from outside | `docker compose -f compose.scale.yaml up --build -d`, then `uv run 26_scaling.py` |
| 27 | [`27_auth.py`](27_auth.py) | Real logins: Argon2id password hashes, server-side sessions in HttpOnly/SameSite/Secure cookies, CSRF tokens, the customer taken from the session (never the request), throttled guessing without revealing which emails exist, staff accounts and an audit log | `uv run 27_auth.py` (no API calls); `--add-user`, `--audit` |

Run any tutorial without arguments to see its built-in demos. Scripts with
`--chat` also have an interactive mode; type `quit` to leave it.

## How they connect

```
Foundations
01 chat ─► 02 chains ─► 03 structured output ─────────────┐
              │                                           │
              ├─► 04 memory ──────┐                       │
              ├─► 05 tools ───────┼─► 07 LangGraph ─► 08 human approval
              └─► 06 RAG ─────────┘        │              │
                    │                      └─► 09 multi-agent (uses 03 for routing)
                    └──────────────────────────► 10 evaluation

Production
07 + 08 ─► 11 streaming ─► 12 FastAPI server
07 + 08 ─► 13 create_agent + middleware ─┬─► 14 MCP tools
                                         ├─► 15 long-term memory
                                         ├─► 16 reliability (retry, fallback)
                                         └─► 18 guardrails
03 ─────► 17 images and PDFs
05 + 13 + 18 ─► 24 text-to-SQL (guardrails in the database)
06 + 10 ─► 19 advanced RAG (hybrid search, rewrite, rerank)

Capstone
11 + 12 + 13 + 16 + 19 ─► 20 Pedalwise support app (with tests) ─► 21 deployment (Docker, CI) ─► 26 scaling out (Postgres, load balancer)
18 + 21 ─────────────────► 27 real logins (sessions, CSRF, audit log)
                                                                    ├─► 22 observability ─► back to 10
                                                                    └─► 23 model providers (DeepSeek) ─► 25 choosing a model (with 10)
```

## Sample data

Tutorials 06–10 use the documents in [`data/`](data/), which describe
**Pedalwise Bikes**, a made-up bicycle shop: its returns policy, warranty and
servicing prices. Because the shop is fictional, the model can't answer from
training data, which makes it obvious when retrieval works. Tutorials 08 and
09 also use a small fake order database defined in the scripts.

Tutorial 17 adds two made-up documents for the same shop: `receipt.png`, a
photo-style till receipt, and `warranty_card.pdf`, a one-page warranty card.
Tutorial 19 adds `catalog.md`, a product catalog with part numbers like
`PW-BRK-220`, which tutorials 06–10 also pick up.

## Files the tutorials create

- `memory.sqlite*`: the saved conversation from Tutorial 07's `--chat`. Delete
  it, or run with `--chat --new`, to start over.
- `long_term_memory.sqlite*`: facts Tutorial 15's `--chat` remembers about each
  user. List them with `--show --user NAME`; delete the file to forget everything.
- `pedalwise.sqlite*`: the capstone's saved conversations. Delete it to start fresh.
- `.rag_index/`: the saved search index from Tutorials 19 and 20, so documents
  are embedded once. It rebuilds itself when the documents change; safe to delete.
- `sales.sqlite`: Tutorial 24's made-up sales database. It's rebuilt the same
  every time, so delete it freely (or run with `--rebuild`).
- The `pedalwise-pg` Docker volume: the Postgres database of Tutorial 26's
  scaled-out stack. `docker compose -f compose.scale.yaml down -v` deletes it.
- The `pedalwise-data` Docker volume: the container's conversations, approval
  queue and search index (Tutorial 21). `docker compose down -v` deletes it.

They're listed in `.gitignore` along with `.env`, so neither they nor your API
key get committed.

## Cost

Every run calls the OpenAI API and is billed to your account. Tutorials 01–09
make a handful of calls each. Tutorial 10 makes roughly 50 (8 questions × 3
variants, plus grading each answer). Tutorials 06, 07, 09 and 10 also call the
embeddings API once per run to index the three small sample documents.
Tutorial 26's check makes about 4 model calls; the rest of it is free.
Tutorial 25 makes about 650 calls with its defaults (4 models x 18 cases x 3
runs, on cheap models; about a dollar); `--repeats 1` cuts that to a third.
Tutorial 24's demo and each `--eval` question make a few calls each (the agent
reads the schema first); `--guardrails` makes none. Tutorial 23 makes 3 calls per model you have a key for, plus 1 for the
fallback demo; DeepSeek bills its own account. Tutorial 22's demo makes about 6; its `--report` and `--export-evals` make none.
Tutorial 21's `--check` makes no model calls unless you add `--chat` (one
question); the container embeds the documents once, on its first start.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
pull request: it checks that every tutorial parses, runs the tests on SQLite
and again on Postgres, and builds the Docker image. None of it calls a model,
so it needs no API key.
