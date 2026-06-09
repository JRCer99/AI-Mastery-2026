"""
Multi-Agent Research Team — Month 12 Agentic AI
Implements the CrewAI Agent/Task/Crew interface from scratch (crewai requires Python 3.10+).
Same API as crewai — swap the classes for real crewai when upgraded to Python 3.10+.

MOCK = True  → structured mock output (no API needed)
MOCK = False → real LLM via Groq (set GROQ_API_KEY env var)
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime
from pathlib import Path
import os

MOCK = True  # Set False + export GROQ_API_KEY=... when API ready

# ─── Mock Tool ───────────────────────────────────────────────────────────────

@dataclass
class MockSearchTool:
    """Drop-in for SerperDevTool — swap when SERPER_API_KEY is available."""
    name: str = "Search"
    description: str = "Search the web for information on a topic"

    def run(self, topic: str) -> str:
        return f"""[MOCK SEARCH RESULTS for: {topic}]
• Large language models have reached human-level performance on many reasoning benchmarks as of 2026.
• Multi-agent systems outperform single LLMs on complex tasks requiring planning and tool use.
• RAG (Retrieval-Augmented Generation) has become the standard for knowledge-grounded AI applications.
• Autonomous agents now handle end-to-end workflows in coding, research, and data analysis.
• Key frameworks: LangGraph (state machines), CrewAI (role-based agents), AutoGen (conversational agents).
• Safety and alignment research is accelerating alongside capability development.
[Source: MOCK — replace with real SerperDevTool when SERPER_API_KEY is available]"""


# ─── CrewAI-compatible Agent / Task / Crew ───────────────────────────────────

@dataclass
class Agent:
    role: str
    goal: str
    backstory: str
    tools: List = field(default_factory=list)
    verbose: bool = True

    def execute(self, task_description: str, context: str = "") -> str:
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Agent: {self.role}]")
            print(f"Goal:  {self.goal}")
            print(f"Task:  {task_description}")

        result = self._mock_execute(task_description, context) if MOCK else self._llm_execute(task_description, context)

        if self.verbose:
            print(f"\nOutput:\n{result}")
        return result

    def _mock_execute(self, task: str, context: str) -> str:
        if "researcher" in self.role.lower():
            tool_result = self.tools[0].run(task) if self.tools else ""
            return (
                f"## Research Findings\n\n{tool_result}\n\n"
                f"**Summary:** Multi-agent architectures distribute complex tasks across specialized agents. "
                f"Combining specialized roles with shared memory yields better results than monolithic models."
            )
        elif "writer" in self.role.lower():
            return (
                f"## Research Report\n\n"
                f"**Introduction**\n"
                f"Based on comprehensive research findings, this report examines the current state of AI agent systems.\n\n"
                f"**Key Findings**\n"
                f"1. Multi-agent architectures distribute tasks across specialized agents, improving quality.\n"
                f"2. Each agent maintains a specific role and goal that guides its reasoning.\n"
                f"3. Tool access enables real-world automation beyond pure text generation.\n\n"
                f"**Context from Researcher**\n{context[:300] if context else 'N/A'}\n\n"
                f"**Recommendations**\n"
                f"- Start with 2–3 specialized agents before scaling to complex pipelines.\n"
                f"- Use structured output for reliable agent-to-agent communication.\n"
                f"- Add human-in-the-loop checkpoints for high-stakes decisions.\n\n"
                f"**Conclusion**\n"
                f"Agentic AI converges language models, retrieval, tool use, and planning into autonomous systems.\n\n"
                f"*{datetime.now().strftime('%B %d, %Y at %H:%M')} | MOCK mode — set MOCK=False + GROQ_API_KEY for real content*"
            )
        return f"[{self.role} completed: {task[:80]}]"

    def _llm_execute(self, task: str, context: str) -> str:
        try:
            from groq import Groq
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            system = (
                f"You are a {self.role}. {self.backstory}\nGoal: {self.goal}\n"
                + (f"Context from previous agents:\n{context}\n" if context else "")
            )
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": task}],
                max_tokens=600,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM error: {e}]"


@dataclass
class Task:
    description: str
    expected_output: str
    agent: Agent


class Crew:
    def __init__(self, agents: List[Agent], tasks: List[Task], verbose: int = 1):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose

    def kickoff(self) -> str:
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"# CREW KICKOFF — {len(self.agents)} agents, {len(self.tasks)} tasks")
            print(f"# Mode: {'MOCK' if MOCK else 'Live LLM'}")
            print(f"{'#'*60}")

        context = ""
        results = []
        for task in self.tasks:
            result = task.agent.execute(task.description, context)
            results.append(result)
            context = result  # each agent's output feeds the next

        if self.verbose:
            print(f"\n{'='*60}\nCREW COMPLETE — {len(self.tasks)} tasks finished.")
        return results[-1]


# ─── Team Setup ──────────────────────────────────────────────────────────────

def create_research_team(topic: str) -> Crew:
    search_tool = MockSearchTool()  # swap: SerperDevTool() when SERPER_API_KEY ready

    researcher = Agent(
        role="Senior AI Researcher",
        goal="Find the latest and most accurate information on the topic",
        backstory="Expert researcher with deep knowledge in AI and technology.",
        tools=[search_tool],
        verbose=True,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Write clear, engaging, and well-structured reports",
        backstory="Excellent writer who turns complex research into readable content.",
        verbose=True,
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[
            Task(
                description=f"Research: {topic}. Find key facts, recent developments, and practical insights.",
                expected_output="Comprehensive research summary with sources",
                agent=researcher,
            ),
            Task(
                description="Write a professional markdown report based on the research findings.",
                expected_output="Complete report with sections, insights, and recommendations",
                agent=writer,
            ),
        ],
        verbose=2,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("🤖 JRC Multi-Agent Research Team")
    print("=" * 60)
    print(f"Mode: {'MOCK (no API needed)' if MOCK else 'Live — Groq LLM'}")
    print("Pipeline: Researcher → Writer")
    print("To enable real LLM: MOCK=False + export GROQ_API_KEY=...\n")

    topic = (
        input("Enter a research topic (or press Enter for default): ").strip()
        or "Best practices for building autonomous AI agents in 2026"
    )

    print(f"\nLaunching agent team → {topic}\n")
    crew = create_research_team(topic)
    result = crew.kickoff()

    report = (
        f"# Multi-Agent Research Report\n\n"
        f"**Topic:** {topic}\n"
        f"**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}\n"
        f"**Mode:** {'MOCK' if MOCK else 'Live LLM'}\n\n---\n\n{result}\n"
    )

    report_path = Path(__file__).parent / "agent_research_report.md"
    report_path.write_text(report)

    print(f"\n✅ Report saved → {report_path.name}")
    print(f"🎉 Month 12 Project 1 Complete! — {datetime.now().strftime('%B %d, %Y')}")
    print("Swap MockSearchTool + set MOCK=False for fully live multi-agent runs.")


if __name__ == "__main__":
    main()
