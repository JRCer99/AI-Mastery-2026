# Month 07: Generative AI, LLMs & Fine-Tuning

**Focus:** Generative models, GPT-style LLMs, prompt engineering, fine-tuning, and building practical LLM-powered applications.

---

## 🎯 Learning Goals

- Understand how generative models and LLMs work (GPT architecture)
- Apply prompt engineering techniques for real-world tasks
- Fine-tune a pre-trained LLM on domain-specific data
- Build practical apps powered by LLMs via API

---

## 📅 Projects

- [ ] Project 1: Domain-Specific Fine-tuned LLM
  - Fine-tune a GPT-style model on a custom dataset
  - Libraries: `transformers`, `datasets`, `torch`, `peft`

- [ ] Project 2: 🏆 AI Code Reviewer Tool *(Weekend Project)*
  - LLM-powered tool that reviews code and gives feedback
  - Libraries: `anthropic` / `openai`, `streamlit`

- [ ] Project 3: 🏆 Rate My Resume Web App *(Weekend Project)*
  - Upload a resume, get AI-powered feedback and score
  - Libraries: `anthropic` / `openai`, `streamlit`, `pypdf2`

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| Transformer decoder | Core architecture behind GPT-style LLMs |
| Prompt engineering | Control LLM output without retraining |
| Fine-tuning vs RAG | When to retrain vs retrieve |
| LoRA / PEFT | Efficient fine-tuning with fewer parameters |
| Token limits | Context window constraints in production |

---

## 📚 Resources

- [HuggingFace — Fine-tuning LLMs](https://huggingface.co/docs/transformers/training)
- [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

## 📂 Structure

```
Month-07-GenerativeAI-LLMs/
├── domain_llm_finetune.py       # Project 1: Fine-tuned LLM
├── ai_code_reviewer.py          # Project 2: AI Code Reviewer
└── rate_my_resume_app.py        # Project 3: Rate My Resume App
```
