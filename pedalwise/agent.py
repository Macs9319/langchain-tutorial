"""The agent: create_agent plus the middleware that makes it production-ready."""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    dynamic_prompt,
)
from langchain_core.language_models import BaseChatModel

from .shop import Customer

SYSTEM_PROMPT = """You are the support assistant for Pedalwise Bikes.

- Answer questions about policies, prices, parts and opening hours with
  search_policies. Only state facts the search results support; if they
  don't cover something, say you don't know.
- For orders use my_orders, for availability check_stock.
- For refunds: first check_refund_eligibility. If eligible and the customer
  wants it, call request_refund; a staff member approves it before it runs.
  If not eligible, explain why.
- Keep answers short and friendly: two or three sentences."""


@dynamic_prompt
def personalised_prompt(request: ModelRequest) -> str:
    """Add the customer's name to the system prompt (Tutorial 13)."""
    customer: Customer = request.runtime.context
    return f"{SYSTEM_PROMPT}\n\nYou are talking to {customer.name}."


def build_agent(model: BaseChatModel, fallback_model: BaseChatModel | None, tools: list,
                checkpointer):
    middleware = [
        personalised_prompt,
        # Mask card numbers people type, before they reach the model (13, 18).
        PIIMiddleware("credit_card", strategy="mask"),
        # A runaway loop is capped at 8 model calls per message (09, 13).
        ModelCallLimitMiddleware(run_limit=8),
        # Long chats: summarize old turns instead of resending them (04, 13).
        SummarizationMiddleware(model=model, trigger=("messages", 40), keep=("messages", 10)),
        # Every refund waits for a person (08, 13).
        HumanInTheLoopMiddleware(
            interrupt_on={"request_refund": {"allowed_decisions": ["approve", "reject"]}},
            interrupt_mode="batched",
        ),
    ]
    # Fallback outermost, retries inside it: primary x3, then fallback (16).
    if fallback_model is not None:
        middleware.append(ModelFallbackMiddleware(fallback_model))
    middleware.append(ModelRetryMiddleware(max_retries=2, on_failure="error"))

    return create_agent(
        model,
        tools=tools,
        middleware=middleware,
        context_schema=Customer,
        checkpointer=checkpointer,
    )
