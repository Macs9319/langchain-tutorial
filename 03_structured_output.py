"""
Tutorial 03 — Structured output
===============================

In Tutorial 02 the chain returned a plain string. That's fine for display,
but code usually needs specific fields: a name, a number, a list. Parsing
free text is fragile, so instead we ask the model to return data that
matches a schema, and LangChain hands it back as a Python object.

What you'll learn:
  1. Describe the output you want with a Pydantic model
  2. Bind it to the chat model with .with_structured_output()
  3. Use the structured model inside a prompt | model chain
  4. Nest models and lists for richer data
  5. Extract structured data from many texts with .batch()

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 03_structured_output.py
  uv run 03_structured_output.py "Pad Thai"
"""

import os
import sys
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Step 1: Describe the output with Pydantic
# ---------------------------------------------------------------------------
# Field descriptions are sent to the model as part of the schema, so write
# them like instructions: they directly shape what the model fills in.
class Ingredient(BaseModel):
    name: str = Field(description="Ingredient name, e.g. 'rice noodles'")
    quantity: str = Field(description="Amount with unit, e.g. '200 g' or '2 tbsp'")


class Recipe(BaseModel):
    """A recipe for a single dish."""

    dish: str = Field(description="Name of the dish")
    cuisine: str = Field(description="Cuisine or country of origin")
    difficulty: Literal["easy", "medium", "hard"]
    prep_minutes: int = Field(description="Total preparation and cooking time in minutes")
    ingredients: list[Ingredient]  # Step 4: a nested list of models
    steps: list[str] = Field(description="Short, ordered cooking steps")


# ---------------------------------------------------------------------------
# Step 2: Bind the schema to the model
# ---------------------------------------------------------------------------
# The returned object is still a chat model, but .invoke() now returns a
# Recipe instance instead of an AIMessage.
recipe_llm = llm.with_structured_output(Recipe)


def direct_call(dish: str) -> None:
    print("=== Step 2: structured model, called directly ===")
    recipe = recipe_llm.invoke(f"Give me a simple home recipe for {dish}.")
    print(type(recipe).__name__, "->", recipe.dish, f"({recipe.cuisine})")
    print(f"difficulty: {recipe.difficulty}, time: {recipe.prep_minutes} min")
    print("ingredients:")
    for item in recipe.ingredients:
        print(f"  - {item.quantity} {item.name}")
    print("steps:")
    for number, step in enumerate(recipe.steps, start=1):
        print(f"  {number}. {step}")
    print()


# ---------------------------------------------------------------------------
# Step 3: Structured output inside a chain
# ---------------------------------------------------------------------------
# No StrOutputParser needed: the structured model already returns an object.
recipe_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "You are a chef who writes recipes for {skill} cooks."),
            ("human", "Recipe for {dish}, max {max_steps} steps."),
        ]
    )
    | recipe_llm
)


def chain_call(dish: str) -> None:
    print("=== Step 3: inside a prompt | model chain ===")
    recipe = recipe_chain.invoke({"skill": "beginner", "dish": dish, "max_steps": 4})
    # Pydantic objects convert to dicts/JSON easily, e.g. to save or send.
    print(recipe.model_dump_json(indent=2), "\n")


# ---------------------------------------------------------------------------
# Step 5: Extraction from many texts with .batch()
# ---------------------------------------------------------------------------
class Review(BaseModel):
    """Information extracted from a product review."""

    product: str = Field(description="The product being reviewed")
    sentiment: Literal["positive", "neutral", "negative"]
    rating: int | None = Field(description="Star rating 1-5 if stated, otherwise null")
    complaints: list[str] = Field(description="Specific problems mentioned; empty if none")


extract_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "Extract the requested fields from the review. "
                       "Only use information stated in the text."),
            ("human", "{review}"),
        ]
    )
    | llm.with_structured_output(Review)
)

SAMPLE_REVIEWS = [
    "The X200 headphones sound amazing and the battery lasts all week. 5 stars!",
    "My AeroPress arrived with a cracked plunger and support never replied. 1/5.",
    "The desk lamp is fine. Bright enough, a bit wobbly on the base.",
]


def batch_extract() -> None:
    print("=== Step 5: batch extraction ===")
    reviews = extract_chain.batch([{"review": text} for text in SAMPLE_REVIEWS])
    for review in reviews:
        print(f"{review.product:<20} {review.sentiment:<9} "
              f"rating={review.rating}  complaints={review.complaints}")
    print()


if __name__ == "__main__":
    dish = " ".join(sys.argv[1:]) or "pancakes"
    print(f"Dish: {dish}\n")

    direct_call(dish)
    chain_call(dish)
    batch_extract()
