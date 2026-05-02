# Month 07: Generative AI, LLMs & Fine-Tuning

**Focus:** Fine-tune LLMs for domain-specific tasks and build AI-powered tools for real-world use cases.

---

## 🎯 Learning Goals

- Fine-tune a language model on custom data (LoRA / QLoRA)
- Build AI-powered tools using LLM APIs
- Understand prompt engineering and output formatting
- Deploy LLM-based applications

---

## 📅 Projects

- [ ] Project 1: Domain-Specific Fine-tuned LLM
  - Fine-tune a small LLM (Mistral / LLaMA) on custom data
  - Libraries: `transformers`, `peft`, `bitsandbytes`

- [ ] Project 2: 🏆 AI Code Reviewer Tool *(Weekend Project)*
  - LLM-powered code review: bugs, security, quality rating
  - Libraries: `anthropic` or `openai`, `streamlit`

- [ ] Project 3: 🏆 Rate My Resume Web App *(Weekend Project)*
  - AI-powered resume scoring and feedback
  - Libraries: `anthropic` or `openai`, `streamlit`

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| LoRA / QLoRA | Efficient fine-tuning with low memory |
| Prompt engineering | Control LLM output quality |
| System prompts | Set LLM persona and constraints |
| Tokenization limits | Manage context window size |
| Inference optimization | Speed vs quality tradeoffs |

---

## 📚 Resources

- [HuggingFace PEFT — LoRA Fine-tuning](https://huggingface.co/docs/peft)
- [Anthropic — Claude API Docs](https://docs.anthropic.com)
- [Fast.ai — Practical Deep Learning Part 2](https://course.fast.ai/Lessons/part2.html)

---

## 📂 Structure

```
Month-07-GenAI-LLMs/
├── llm_finetuner.py        # Project 1: Fine-tuned LLM
├── ai_code_reviewer.py     # Project 2: 🏆 AI Code Reviewer
└── rate_my_resume_app.py   # Project 3: 🏆 Rate My Resume
```
