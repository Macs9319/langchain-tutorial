# Learning guide

29 tutorials is a lot to start on. This page helps you choose: a route for
what you want to build, an index of where each topic is taught, and what
each tutorial needs to run. The [README](README.md) has setup and the full
table.

Every tutorial is one file you read top to bottom; the comments are the
lesson. Expect roughly 15 to 30 minutes each to read and run.

## Start here

Whatever your goal, do these four first. They're short and everything else
builds on them.

| # | Tutorial | You'll be able to |
|---|---|---|
| 01 | [Chat basics](01_chat_basics.py) | Call a chat model, read its reply and token usage, stream it |
| 02 | [Prompts and chains](02_prompts_and_chains.py) | Template prompts and connect steps with `\|` |
| 03 | [Structured output](03_structured_output.py) | Get typed Python objects back instead of text |
| 05 | [Tool calling](05_tool_calling.py) | Let a model call your functions, and write the loop yourself |

## Routes

Pick the one closest to what you want to build. Numbers in *italics* are
optional detours.

### Answer questions from your documents (RAG)

**01 → 02 → 03 → 06 → 10 → 19**, *then 23*

[06](06_rag.py) builds retrieval from scratch: split, embed, search, cite.
[10](10_evaluation.py) shows how to measure whether answers are right, which
you need before changing anything. [19](19_advanced_rag.py) improves
retrieval with hybrid search, query rewriting and re-ranking, measured with
10's method. *[23](23_model_providers.py) runs it on another provider.*

### Build an agent

**01 → 02 → 03 → 05 → 04 → 07 → 08 → 13**, *then 09, 14, 15, 16, 18*

[05](05_tool_calling.py) is the agent loop by hand, [04](04_conversation_memory.py)
how a model keeps track of a conversation, [07](07_langgraph_agent.py)
the agent as a LangGraph graph with memory, [08](08_human_in_the_loop.py)
pauses it for a person's approval, and [13](13_create_agent_middleware.py)
is the high-level `create_agent` with middleware. The detours add several
agents ([09](09_multi_agent.py)), MCP tools ([14](14_mcp_tools.py)), memory
across conversations ([15](15_long_term_memory.py)), surviving failures
([16](16_reliability.py)) and defending against prompt injection
([18](18_guardrails.py)).

### Ship a real app

**Agent route → 11 → 12 → 20 → 21 → 27 → 22 → 28 → 29 → 26**, *with 23, 25*

[11](11_async_streaming.py) and [12](12_fastapi_server.py) put an agent
behind a streaming web API. [20](20_capstone.py) is the capstone: the
Pedalwise support app in the [`pedalwise/`](pedalwise/) package, with tests.
The rest take it to production one problem at a time:

| # | Problem it solves |
|---|---|
| [21](21_deployment.py) | Running it in Docker: config from the environment, health checks, logs, CI |
| [27](27_auth.py) | Knowing who's who: logins, sessions, CSRF, an audit log |
| [22](22_observability.py) | Knowing what it does: runs, tokens, latency, customer feedback |
| [28](28_migrations.py) | Changing tables that already hold data |
| [29](29_backups.py) | Getting the data back after a mistake or a lost disk |
| [26](26_scaling.py) | Running several copies behind a load balancer |
| *[23](23_model_providers.py)* | *Not depending on one model provider* |
| *[25](25_model_selection.py)* | *Choosing the model with evidence* |

### Measure quality

**01 → 02 → 03 → 06 → 10 → 22 → 25**

Offline tests with known answers ([10](10_evaluation.py)), feedback from
real users ([22](22_observability.py)), and both combined to pick a model on
quality, speed and cost ([25](25_model_selection.py)). The recurring lesson:
read the failures before trusting the score, because graders have bugs too.

### Questions about a database

**01 → 02 → 03 → 05 → 13 → 18 → 24**

[24](24_text_to_sql.py) lets an agent write SQL, with the guardrails in the
database connection rather than the prompt. [18](18_guardrails.py) explains
why that's the place for them.

### Images and documents

**01 → 03 → 17**

[17](17_images_and_pdfs.py): photos, receipts and PDFs as model input.

## Where is it taught?

| Topic | Tutorials |
|---|---|
| Streaming tokens | 01, 11, 12 |
| Prompt templates, chains | 02 |
| Structured output, Pydantic | 03, 13 (for agents) |
| Conversation memory | 04, 07 (saved to SQLite), 15 (across conversations) |
| Tool calling | 05, 07, 13, 14 (from MCP servers) |
| RAG: embeddings, vector search | 06, 19 (hybrid, rewriting, re-ranking) |
| LangGraph graphs, checkpointers | 07, 08, 09 |
| Human approval | 08, 13, 20 |
| Multi-agent | 09 |
| Evaluation, LLM-as-judge | 10, 22, 25 |
| Async, streaming events, SSE | 11, 12 |
| FastAPI | 12, 20 |
| Middleware (PII, limits, summaries) | 13, 20 |
| MCP | 14 |
| Retries, fallbacks, rate limits, caching | 16 |
| Images and PDFs | 17 |
| Prompt injection, guardrails | 18, 24 |
| Testing without API calls | 20, [`tests/`](tests/) |
| Docker, health checks, CI | 21 |
| Structured logs, tracing, feedback | 21, 22 |
| Other providers (DeepSeek), `init_chat_model` | 23 |
| Text-to-SQL | 24 |
| Model choice, cost per message | 25 |
| Postgres, load balancing, races | 26 |
| Logins, sessions, CSRF, password hashing | 27 |
| Schema migrations | 28 |
| Backups, restores, `pg_dump` | 29 |

## What each one needs

All tutorials need an API key in `.env`: OpenAI by default, or DeepSeek with
`CHAT_MODEL=deepseek:deepseek-flash` ([23](23_model_providers.py)).

- **An OpenAI key in any case:** 06, 07, 09, 10, 15 and 19 embed documents
  with OpenAI, which DeepSeek can't do. The capstone (20-22) and 25 can skip
  embeddings with `EMBEDDINGS=none`; 25's judge is an OpenAI model unless you
  set `JUDGE_MODEL`. 17's PDF steps need a model that reads PDFs.
- **Docker:** 21, 26, 27, 28 and 29 show Docker commands; 26 needs Docker to run.
- **A demo password:** the capstone (20-22, 26-28) needs
  `PEDALWISE_DEMO_PASSWORD` in `.env` for its demo accounts.

### Free to run

These make no model calls, so they cost nothing:

- `uv run pytest`: the capstone's tests
- `uv run 24_text_to_sql.py --guardrails`: trying to break the database
- `uv run 27_auth.py`: trying to break the logins
- `uv run 28_migrations.py`: upgrading an old database
- `uv run 29_backups.py`: losing data and getting it back
- `uv run 21_deployment.py --config` and `--check URL` (without `--chat`)
- `uv run 22_observability.py --report`

Everything else makes a few model calls per run; the
[README's cost section](README.md#cost) lists the bigger ones.
