from langchain_core.tools import tool
from datetime import datetime
import re

@tool
def get_current_time() -> str:
    """Returns current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluates a simple math expression like '15 * 27' or '100 / 4'."""
    try:
        # Only allow safe math characters
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
            return "Error: only basic math expressions allowed"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def simulate_agent(query: str) -> str:
    """Simulated ReAct agent — routes to tools based on query keywords."""
    query_lower = query.lower()

    if any(w in query_lower for w in ["time", "date", "now", "today"]):
        result = get_current_time.invoke({})
        return f"[Tool: get_current_time] → {result}"

    math_match = re.search(r'[\d]+[\s\d\+\-\*\/\.\(\)]+[\d]', query)
    if math_match:
        result = calculate.invoke({"expression": math_match.group()})
        return f"[Tool: calculate] → {result}"

    return "I can help with: time/date questions and math calculations. Try those!"

def main():
    print("🤖 Simple Tool-Use Agent — Month 9 LLM Orchestration")
    print("=" * 55)
    print("Tools available: get_current_time, calculate")
    print("(Simulated agent — swap in real LLM when API key ready)\n")

    examples = [
        "What time is it right now?",
        "What is 15 * 27?",
        "Calculate 100 / 4 + 50",
    ]

    for query in examples:
        print(f"Query:    {query}")
        print(f"Response: {simulate_agent(query)}\n")

    print("-" * 55)
    query = input("Ask the agent something: ")
    print(f"Response: {simulate_agent(query)}")

    print(f"\n🎉 Month 9 Project 1 Complete! - {datetime.now().strftime('%B %d, %Y')}")
    print("Foundation set — ready for multi-tool skeleton and Month 12 capstone.")

if __name__ == "__main__":
    main()
