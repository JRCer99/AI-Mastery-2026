"""
LangGraph autonomous research agent.

Graph: planner → researcher → writer → critic → (revise loop OR finalize) → END

State flows through 5 nodes with conditional edge at critic:
- PASS or iteration >= 2 → finalizer
- FAIL and iteration < 2  → writer (revision loop)
"""

import operator
import anthropic
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from tools import search_knowledge_base, mock_web_search

load_dotenv()
from knowledge_base import seed_knowledge_base

_client = anthropic.Anthropic()


class AgentState(TypedDict):
    query: str
    sub_questions: List[str]
    research_notes: str
    draft_report: str
    critique_passed: bool
    critique_feedback: str
    final_report: str
    iteration: int
    log: Annotated[List[str], operator.add]  # accumulates across all nodes


# --- helper ---

def _llm(prompt: str, max_tokens: int = 1200) -> str:
    resp = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text


# --- nodes ---

def planner(state: AgentState) -> dict:
    raw = _llm(
        f"Break this research query into 3-4 focused sub-questions that together answer it.\n"
        f"Query: {state['query']}\n\n"
        f"Output ONLY a numbered list. No preamble.",
        max_tokens=250
    )
    questions = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]
    return {
        "sub_questions": questions[:4],
        "iteration": 0,
        "log": [f"🗺️  Planner: generated {len(questions)} sub-questions"]
    }


def researcher(state: AgentState) -> dict:
    sections = []
    for q in state["sub_questions"]:
        kb = search_knowledge_base(q)
        web = mock_web_search(q)
        sections.append(
            f"### Sub-question: {q}\n"
            f"**Knowledge Base (RAG):**\n{kb}\n\n"
            f"**Web Search:**\n{web}"
        )
    return {
        "research_notes": "\n\n---\n\n".join(sections),
        "log": [f"🔍 Researcher: gathered context for {len(state['sub_questions'])} sub-questions"]
    }


def writer(state: AgentState) -> dict:
    feedback = state.get("critique_feedback", "")
    revision_block = (
        f"\n\n**Address this critique in the revision:**\n{feedback}"
        if feedback else ""
    )
    draft = _llm(
        f"Write a comprehensive research report on: {state['query']}\n\n"
        f"Research context:\n{state['research_notes']}"
        f"{revision_block}\n\n"
        f"Structure: Executive Summary → Key Findings (3-4 sections) → Conclusion.\n"
        f"Use markdown. Be analytical. Cite knowledge base findings specifically.",
        max_tokens=1800
    )
    action = "revised" if feedback else "written"
    return {
        "draft_report": draft,
        "log": [f"✍️  Writer: draft {action} ({len(draft):,} chars)"]
    }


def critic(state: AgentState) -> dict:
    evaluation = _llm(
        f"Evaluate this research report critically.\n\n"
        f"Original query: {state['query']}\n\n"
        f"Report:\n{state['draft_report']}\n\n"
        f"Check: completeness, accuracy, depth, structure, evidence.\n"
        f"Respond exactly as:\n"
        f"VERDICT: PASS\n"
        f"FEEDBACK: <one sentence>\n\n"
        f"OR\n\n"
        f"VERDICT: FAIL\n"
        f"FEEDBACK: <specific improvements needed>",
        max_tokens=300
    )
    passed = "VERDICT: PASS" in evaluation
    feedback = ""
    for line in evaluation.splitlines():
        if line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()
            break
    new_iter = state["iteration"] + 1
    return {
        "critique_passed": passed,
        "critique_feedback": feedback,
        "iteration": new_iter,
        "log": [f"🧐 Critic: {'PASS' if passed else 'FAIL'} — iteration {new_iter}"]
    }


def finalizer(state: AgentState) -> dict:
    final = _llm(
        f"Polish this report to final publication quality.\n\n"
        f"{state['draft_report']}\n\n"
        f"Requirements:\n"
        f"- Add a clear title as H1\n"
        f"- Ensure smooth section transitions\n"
        f"- Fix any markdown formatting issues\n"
        f"- Add a '---' divider before Conclusion\n"
        f"- End with key takeaways bullet list",
        max_tokens=2000
    )
    return {
        "final_report": final,
        "log": ["✅ Finalizer: report polished and complete"]
    }


# --- routing ---

def _route_after_critic(state: AgentState) -> str:
    if state["critique_passed"] or state["iteration"] >= 2:
        return "finalize"
    return "revise"


# --- graph ---

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalizer", finalizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"finalize": "finalizer", "revise": "writer"}
    )
    graph.add_edge("finalizer", END)

    return graph.compile()


def run_agent(query: str) -> dict:
    """Run the full autonomous agent pipeline. Returns final AgentState."""
    seed_knowledge_base()
    graph = build_graph()
    initial_state: AgentState = {
        "query": query,
        "sub_questions": [],
        "research_notes": "",
        "draft_report": "",
        "critique_passed": False,
        "critique_feedback": "",
        "final_report": "",
        "iteration": 0,
        "log": []
    }
    return graph.invoke(initial_state)
