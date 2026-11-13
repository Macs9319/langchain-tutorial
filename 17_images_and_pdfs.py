"""
Tutorial 17 — Images and PDFs as input
======================================

Until now every message was plain text. Modern models can also look at
images and read PDFs, so a customer can just send a photo of their receipt
instead of typing out the order details.

The sample files in data/ are made up:
  - receipt.png        a phone photo of a Pedalwise till receipt
  - warranty_card.pdf  a one-page warranty registration card

What you'll learn:
  1. Send an image to the model as a content block
  2. Extract structured data from an image (Tutorial 03 + vision)
  3. Ask questions about a PDF
  4. Send several files in one message and reason across them
  5. Give an agent an uploaded photo, so it reads it and uses its tools

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).
  The model must support image and PDF input; gpt-5-mini does.

Run:
  uv run 17_images_and_pdfs.py
  uv run 17_images_and_pdfs.py path/to/your_photo.jpg   # describe your own image
"""

import base64
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

DATA = Path(__file__).parent / "data"

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Step 1: An image is just another content block
# ---------------------------------------------------------------------------
# A message's content can be a list of blocks instead of a string:
#   {"type": "text",  "text": "..."}
#   {"type": "image", "base64": "...", "mime_type": "image/png"}   from a file
#   {"type": "image", "url": "https://..."}                       from the web
#   {"type": "file",  "base64": "...", "mime_type": "application/pdf",
#                     "filename": "..."}                           a PDF
# These are LangChain's standard blocks; ChatOpenAI converts them to OpenAI's
# format, and other providers' classes convert them to theirs.
def file_block(path: Path) -> dict:
    """Read a local image or PDF and wrap it as a content block."""
    mime_type = mimetypes.guess_type(path.name)[0]
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    if mime_type == "application/pdf":
        return {"type": "file", "base64": data, "mime_type": mime_type, "filename": path.name}
    if mime_type and mime_type.startswith("image/"):
        return {"type": "image", "base64": data, "mime_type": mime_type}
    raise ValueError(f"Unsupported file type: {path.name}")


def describe_image(path: Path) -> None:
    print("=== Step 1: describe an image ===")
    message = HumanMessage(content=[
        {"type": "text", "text": "Describe this image in two sentences."},
        file_block(path),
    ])
    reply = llm.invoke([message])
    print(reply.text)
    # Images cost input tokens, more for larger images. Check the real cost:
    print("input tokens:", reply.usage_metadata["input_tokens"], "\n")


# ---------------------------------------------------------------------------
# Step 2: Structured data from a photo
# ---------------------------------------------------------------------------
# Reading text in an image and returning it as typed fields replaces a whole
# OCR (text recognition) pipeline with one call.
class LineItem(BaseModel):
    description: str
    amount_eur: float


class Receipt(BaseModel):
    """Data read from a till receipt."""

    store: str = Field(description="Store name and location")
    date: str = Field(description="Purchase date as YYYY-MM-DD")
    order_id: str | None = Field(description="Order number, if printed")
    items: list[LineItem]
    total_eur: float
    payment_method: str = Field(description="How it was paid, e.g. 'Visa ending 4242'")


def extract_receipt() -> Receipt:
    print("=== Step 2: structured extraction from an image ===")
    reader = llm.with_structured_output(Receipt)
    receipt = reader.invoke([HumanMessage(content=[
        {"type": "text", "text": "Extract the receipt details. Use only what is printed."},
        file_block(DATA / "receipt.png"),
    ])])
    print(receipt.model_dump_json(indent=2))

    # Never trust numbers read from an image blindly. Check what you can:
    items_sum = round(sum(item.amount_eur for item in receipt.items), 2)
    # Compare within a cent: decimals like 956.5 can't always be stored exactly.
    status = "OK" if abs(items_sum - receipt.total_eur) < 0.01 else "MISMATCH, ask a human"
    print(f"check: items add up to {items_sum}, total says {receipt.total_eur} -> {status}\n")
    return receipt


# ---------------------------------------------------------------------------
# Step 3: Questions about a PDF
# ---------------------------------------------------------------------------
# The model receives both the PDF's text and an image of each page, so it can
# read layouts, tables and stamps too. Very long PDFs are better split and
# searched with RAG (Tutorial 06) than sent whole every time.
def ask_pdf() -> None:
    print("=== Step 3: questions about a PDF ===")
    pdf = file_block(DATA / "warranty_card.pdf")
    for question in ["What is the frame serial number?",
                     "Until when are the components covered?",
                     "Are brake pads covered?"]:
        reply = llm.invoke([HumanMessage(content=[
            {"type": "text", "text": f"Answer in one short sentence: {question}"}, pdf])])
        print(f"Q: {question}\nA: {reply.text}")
    print()


# ---------------------------------------------------------------------------
# Step 4: Several files in one message
# ---------------------------------------------------------------------------
def compare_documents() -> None:
    print("=== Step 4: reasoning across two files ===")
    reply = llm.invoke([HumanMessage(content=[
        {"type": "text", "text": (
            "The first file is a receipt, the second a warranty card. Do they "
            "describe the same purchase? Compare order number, date and bike model, "
            "then answer yes or no with a one-line reason.")},
        file_block(DATA / "receipt.png"),
        file_block(DATA / "warranty_card.pdf"),
    ])])
    print(reply.text, "\n")


# ---------------------------------------------------------------------------
# Step 5: An agent that receives a photo
# ---------------------------------------------------------------------------
# Images go into an agent exactly like into a model: as content blocks in
# the user's message. The agent reads the photo, then calls tools with what
# it found.
ORDERS = {"A1004": {"item": "Trail 500 mountain bike", "status": "delivered",
                    "first_service_booked": False}}


@tool
def get_order(order_id: str) -> str:
    """Look up an order by ID: item, status and whether the free first
    service is booked."""
    order = ORDERS.get(order_id.strip().upper())
    return str(order) if order else f"No order with ID {order_id}."


@tool
def book_first_service(order_id: str, date: str) -> str:
    """Book the free first safety check for an order on a date (YYYY-MM-DD)."""
    order = ORDERS.get(order_id.strip().upper())
    if not order:
        return f"No order with ID {order_id}."
    order["first_service_booked"] = True
    return f"Free first service booked for order {order_id} on {date}."


def agent_with_photo() -> None:
    print("=== Step 5: an agent reading an uploaded photo ===")
    agent = create_agent(
        llm,
        tools=[get_order, book_first_service],
        system_prompt="You are Pedalwise Bikes support. Customers may send photos of "
                      "receipts; read the order number from them and use the tools. "
                      "Answer in two short sentences.",
    )
    result = agent.invoke({"messages": [HumanMessage(content=[
        {"type": "text", "text": "Here's my receipt. Can you book my free first "
                                 "service for 2026-11-20?"},
        file_block(DATA / "receipt.png"),
    ])]})
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", []):
            print(f"  → {call['name']}({call['args']})")
    print("assistant:", result["messages"][-1].text)
    print("order now:", ORDERS["A1004"], "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        describe_image(Path(sys.argv[1]))
    else:
        describe_image(DATA / "receipt.png")
        extract_receipt()
        ask_pdf()
        compare_documents()
        agent_with_photo()
