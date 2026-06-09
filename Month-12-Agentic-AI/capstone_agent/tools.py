"""
Agent tools: knowledge base search (RAG) and mock web search.
In production, replace mock_web_search with Tavily/SerpAPI.
"""

from dotenv import load_dotenv
load_dotenv()

import anthropic
from knowledge_base import search as kb_search

_client = anthropic.Anthropic()


def search_knowledge_base(query: str) -> str:
    """RAG lookup — returns top-3 semantically relevant knowledge base chunks."""
    results = kb_search(query, n_results=3)
    if not results:
        return "No relevant information found in knowledge base."
    return "\n\n".join(f"[KB {i+1}]: {r}" for i, r in enumerate(results))


def mock_web_search(query: str) -> str:
    """
    Simulates web search via Claude Haiku generating realistic search snippets.
    Replace with Tavily/SerpAPI for real search results.
    """
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=350,
        messages=[{
            "role": "user",
            "content": (
                f"Generate 3 realistic, factual web search result snippets for: '{query}'\n"
                "Format: numbered list, each snippet 2-3 sentences. Be technical and accurate."
            )
        }]
    )
    return response.content[0].text
