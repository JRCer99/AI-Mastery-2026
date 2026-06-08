import os
from groq import Groq
from datetime import datetime

# Flip to False once GROQ_API_KEY or ANTHROPIC_API_KEY is available
MOCK = True

SYSTEM_PROMPT = """You are a senior software engineer doing a code review.
Analyze the provided code and return a structured review with:
1. Overall quality score (X/10)
2. Strengths (bullet points)
3. Bugs or issues found (numbered, with line references if possible)
4. Improvement suggestions
5. One refactored snippet showing a key improvement

Be specific and actionable. Format with markdown headers."""

MOCK_REVIEW = """## Overall Quality Score: 6/10

## Strengths
- Simple, readable function structure
- Uses standard library (`json`) appropriately
- Clear function naming

## Bugs & Issues
1. **No error handling** — `get_tasks()` crashes if `tasks.json` doesn't exist (`FileNotFoundError`)
2. **Race condition** — if write fails mid-stream, `tasks.json` gets corrupted
3. **Import inside function** — `import json` inside functions works but is non-standard; move to top of file
4. **No input validation** — empty string or `None` title accepted silently

## Improvement Suggestions
- Move `import json` to top of file
- Wrap file reads in `try/except FileNotFoundError`
- Use atomic write pattern (`write to .tmp`, then `os.replace`) to prevent corruption
- Validate `title` is a non-empty string before appending

## Refactored Snippet
```python
import json
import os

def add_task(tasks: list, title: str, priority: str = "medium") -> None:
    if not title or not isinstance(title, str):
        raise ValueError("Title must be a non-empty string")
    tasks.append({"title": title, "priority": priority, "done": False})
    tmp_path = "tasks.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(tasks, f)
    os.replace(tmp_path, "tasks.json")  # atomic write
```"""

def review_code(code: str, language: str = "Python") -> str:
    if MOCK:
        return MOCK_REVIEW

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please review this {language} code:\n\n```{language.lower()}\n{code}\n```"}
        ],
    )
    return message.choices[0].message.content


def main():
    print("🤖 JRC AI Code Reviewer — Powered by Claude")
    print("=" * 60)

    # Sample code to review
    sample_code = '''
def add_task(tasks, title, priority="medium"):
    task = {"title": title, "priority": priority, "done": False}
    tasks.append(task)
    with open("tasks.json", "w") as f:
        import json
        json.dump(tasks, f)
    print("Task added!")

def get_tasks():
    import json
    with open("tasks.json") as f:
        return json.load(f)
'''

    print("Submitting code for review...\n")
    review = review_code(sample_code, language="Python")

    print(f"# AI Code Review — {datetime.now().strftime('%B %d, %Y %H:%M')}\n")
    print(review)
    print("\n✅ Month 7 Project 2 Complete!")


if __name__ == "__main__":
    main()
