"""Agentic RAG pipeline for allaboutberlin.com guides."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv
from openai import OpenAI

from ingest import build_index, load_documents
from rag_helper import DEFAULT_MODEL

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

SEARCH_TOOL = {
    "type": "function",
    "name": "search",
    "description": (
        "Search the allaboutberlin.com guides database for information about "
        "living in Germany (Schufa, banking, housing, health insurance, visas, "
        "taxes, registration, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    'Search query about life in Germany, e.g. "Schufa score", '
                    '"health insurance", "apartment search in Berlin".'
                ),
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

DEFAULT_INSTRUCTIONS = """
You're a helpful assistant for expats living in or moving to Berlin, Germany.
You're given a question and your task is to answer it based on the allaboutberlin.com guides.

If you want to look up information, use the search function.
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform a search, analyze the results,
then perform more searches with related or expanded keywords.

The question has to be about life in Germany (bureaucracy, housing, banking, health insurance,
visas, taxes, registration, etc.). Off-topic questions shouldn't be answered.
If the search returns nothing relevant, it's likely an off-topic question.
If you can't answer the question using the guides, don't answer it yourself.
Only use facts from the allaboutberlin.com guides database.

At the end, ask if there are other areas of life in Germany the user wants to explore.
""".strip()


# ── Agent class ────────────────────────────────────────────────────────────────


class AgentRAG:

    def __init__(
        self,
        index,
        llm_client: OpenAI,
        instructions: str = DEFAULT_INSTRUCTIONS,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.model = model

    def _search(self, query: str, num_results: int = 5) -> list[dict]:
        return self.index.search(query, num_results=num_results)

    def _execute_tool_call(self, call) -> dict:
        args = json.loads(call.arguments)
        result = self._search(**args)
        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, indent=2),
        }

    def loop(self, question: str) -> str:
        messages: list = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": question},
        ]
        last_answer = ""
        iteration = 1

        while True:
            print(f"iteration #{iteration}...")
            has_function_calls = False

            response = self.llm_client.responses.create(
                model=self.model,
                input=messages,
                tools=[SEARCH_TOOL],
            )
            messages.extend(response.output)

            for item in response.output:
                if item.type == "function_call":
                    print(f"function_call: {item.name} {item.arguments}")
                    messages.append(self._execute_tool_call(item))
                    has_function_calls = True
                elif item.type == "message":
                    last_answer = item.content[0].text

            iteration += 1
            if not has_function_calls:
                break

        return last_answer


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG for Berlin expat guides")
    parser.add_argument("--question", "-q", required=True, help="Question to ask")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="OpenAI model to use")
    args = parser.parse_args()

    print("Loading documents...")
    docs = load_documents()
    print("Building keyword index...")
    idx = build_index(docs)
    print(f"Indexed {len(docs)} document sections.\n")

    agent = AgentRAG(index=idx, llm_client=OpenAI(), model=args.model)
    answer = agent.loop(args.question)
    print("\n" + answer)
