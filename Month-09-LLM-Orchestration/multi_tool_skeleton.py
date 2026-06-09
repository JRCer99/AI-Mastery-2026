"""
Multi-tool Agent Skeleton — Month 9 LLM Orchestration
Modular framework: Tool → Memory → Agent → AgentExecutor
This is the foundation for the Month 12 Agentic AI capstone.
"""

from dataclasses import dataclass
from typing import Callable, Optional
from datetime import datetime
import re


# ─── Tool ────────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]

    def run(self, input: str) -> str:
        try:
            return self.func(input)
        except Exception as e:
            return f"[Tool error] {e}"


# ─── Memory ──────────────────────────────────────────────────────────────────

class Memory:
    def __init__(self):
        self._history: list = []

    def add(self, role: str, content: str):
        self._history.append({"role": role, "content": content, "ts": datetime.now().isoformat()})

    def recent(self, n: int = 5) -> list:
        return self._history[-n:]

    def clear(self):
        self._history.clear()


# ─── Agent ───────────────────────────────────────────────────────────────────

class Agent:
    """
    ReAct-style agent skeleton.
    Observe → Think → Act loop. Swap _think() for real LLM call when API ready.
    """

    def __init__(self, tools: list, memory: Optional[Memory] = None):
        self.tools = {t.name: t for t in tools}
        self.memory = memory or Memory()

    def _think(self, query: str) -> tuple:
        """Simulated routing — replace with LLM call when API key ready."""
        q = query.lower()
        if any(w in q for w in ["time", "date", "now", "today"]):
            return "get_time", ""
        match = re.search(r'[\d][\s\d\+\-\*\/\.]+[\d]', query)
        if match:
            return "calculator", match.group()
        if any(w in q for w in ["reverse", "flip"]):
            return "string_reverser", query.split()[-1]
        if any(w in q for w in ["count", "words"]):
            return "word_counter", query
        return None, None

    def run(self, query: str) -> str:
        self.memory.add("user", query)
        tool_name, tool_input = self._think(query)

        if tool_name and tool_name in self.tools:
            observation = self.tools[tool_name].run(tool_input)
            response = f"[Thought] Use {tool_name}\n[Action] {tool_input or '(none)'}\n[Observation] {observation}"
        else:
            response = "No suitable tool found for that query."

        self.memory.add("agent", response)
        return response


# ─── Built-in Tools ──────────────────────────────────────────────────────────

DEFAULT_TOOLS = [
    Tool("get_time",       "Returns current date and time",  lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    Tool("calculator",     "Evaluates math expressions",     lambda e: str(eval(e)) if re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', e) else "Invalid"),
    Tool("string_reverser","Reverses a string",              lambda t: t[::-1]),
    Tool("word_counter",   "Counts words in a string",       lambda t: f"{len(t.split())} words"),
]


# ─── Demo ─────────────────────────────────────────────────────────────────────

def main():
    print("🤖 Multi-tool Agent Skeleton — Month 9 LLM Orchestration")
    print("=" * 58)
    print("Tools:", ", ".join(t.name for t in DEFAULT_TOOLS))
    print("(Simulated reasoning — swap _think() for real LLM call)\n")

    agent = Agent(tools=DEFAULT_TOOLS)

    queries = [
        "What time is it?",
        "What is 128 * 4 + 12?",
        "Reverse the word: hello",
        "How many words in this sentence right here?",
    ]

    for query in queries:
        print(f"Query: {query}")
        print(agent.run(query))
        print()

    print("─" * 58)
    print("Memory (last 3 turns):")
    for m in agent.memory.recent(3):
        print(f"  [{m['role']}] {m['content'][:80]}")

    print(f"\n🎉 Month 9 Project 2 Complete! - {datetime.now().strftime('%B %d, %Y')}")
    print("Skeleton ready — plug in LangGraph or real LLM for Month 12.")

if __name__ == "__main__":
    main()
