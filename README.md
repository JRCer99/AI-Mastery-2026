# JRC AI Mastery 2026

![Progress](https://img.shields.io/badge/Progress-32%2F32%20Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-LLM%20%2B%20Agents-blueviolet?logo=anthropic)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Apps-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-success)

**12-month self-directed AI/ML roadmap — from Python fundamentals to fully autonomous AI agents.**

Built in parallel with a BS in Computer Science at SNHU. Every month ships at least one runnable project. No tutorials followed passively — every concept implemented from scratch or applied to a real problem.

---

## 🏆 Capstone: Autonomous AI Research Agent

> The final project synthesizing all 12 months of skills into one system.

A fully autonomous research agent that operates without human involvement:

```
Query → [PLANNER] → [RESEARCHER: RAG + Web] → [WRITER] → [CRITIC] → [FINALIZER] → Report
                                                    ↑___________|
                                               (self-critique loop)
```

**Stack:** LangGraph · ChromaDB · Claude Sonnet 4.6 · Streamlit · Python

**What it does:** Takes any research question → decomposes into sub-questions → retrieves context from a ChromaDB vector knowledge base (RAG) → writes a structured report → autonomously critiques and revises → outputs polished markdown.

**→ [View Capstone](Month-12-Agentic-AI/capstone_agent/)**

---

## 🏆 Featured Projects

| Project | Month | Stack | What it does |
|---------|-------|-------|-------------|
| [Autonomous Research Agent](Month-12-Agentic-AI/capstone_agent/) | 12 | LangGraph, RAG, Claude | 5-node autonomous agent with self-critique loop |
| [AI Code Reviewer](Month-07-GenerativeAI-LLMs/) | 7 | Claude API, Streamlit | Reviews code, flags bugs, suggests improvements |
| [Rate My Resume App](Month-07-GenerativeAI-LLMs/) | 7 | Claude API, Streamlit | AI-powered resume scoring and feedback |
| [Knowledge Base Chatbot](Month-11-RAG/) | 11 | RAG, ChromaDB, Claude | Advanced RAG chatbot over custom documents |
| [Semantic Search Engine](Month-10-Vector-DBs/) | 10 | ChromaDB, embeddings | Semantic search over document collections |
| [Study With Me Dashboard](Month-08-MLOps-LLMOps/) | 8 | Streamlit, live data | Real-time productivity and focus tracker |
| [CNN Image Classifier](Month-05-DL-CV-NLP/) | 5 | PyTorch, CNN | Fashion MNIST classifier with training dashboard |
| [Sentiment Analysis App](Month-05-DL-CV-NLP/) | 5 | DistilBERT, Streamlit | Real-time sentiment scoring web app |

---

## 📅 Full Curriculum — 32/32 Complete

### Month 1: Python & Git
- [x] CLI Task Manager (priorities, due dates, JSON persistence)
- [x] Personal Portfolio Generator Script
- [x] Bonus: Study With Me Live Dashboard ✅

### Month 2: Math & Stats
- [x] Exploratory Data Analysis Dashboard
- [x] Gradient Descent from Scratch Visualizer
- [x] Bayesian Probability Simulator

### Month 3: Data Manipulation & Feature Engineering
- [x] Titanic / Kaggle Full Data Pipeline
- [x] Reusable Automated Data Preprocessing Package

### Month 4: Machine Learning & Neural Networks
- [x] End-to-End ML Pipeline (House Price Prediction)
- [x] Model Comparison Dashboard (Scikit-learn)
- [x] Kaggle Competition Submission

### Month 5: Deep Learning, Computer Vision & NLP
- [x] Image Classifier (CNN with PyTorch)
- [x] Sentiment Analysis Web App
- [x] Sequence Prediction Model (LSTM)

### Month 6: Transformers & Transfer Learning
- [x] Fine-tuned BERT for Custom Text Classification
- [x] Transfer Learning Demo (Domain Adaptation)

### Month 7: Generative AI, LLMs & Fine-Tuning
- [x] Domain-Specific Fine-tuned LLM ✅
- [x] 🏆 AI Code Reviewer Tool ✅
- [x] 🏆 Rate My Resume Web App ✅

### Month 8: MLOps & LLMOps
- [x] Dockerized + MLflow Tracked Model ✅
- [x] CI/CD Pipeline with GitHub Actions ✅
- [x] 🏆 Study With Me Live Dashboard ✅

### Month 9: LLM Orchestration & Agent Foundations
- [x] Simple Tool-Use Agent ✅
- [x] Multi-tool Agent Skeleton ✅

### Month 10: Vector Embeddings & Vector Databases
- [x] Semantic Search Engine over Documents ✅
- [x] 🏆 Personal Notes → Searchable Knowledge Base ✅

### Month 11: Retrieval-Augmented Generation (RAG)
- [x] 🏆 Personal Knowledge Base Chatbot — Advanced RAG ✅
- [x] Domain Expert RAG System ✅

### Month 12: Agentic AI & Autonomous Multimodal Agents
- [x] Multi-Agent Research Team ✅
- [x] Autonomous Multimodal Assistant ✅
- [x] 🏆 Chrome Extension — AI Summarizer ✅
- [x] 🏆 Capstone: Fully Autonomous AI Agent System ✅

---

## 🛠 Tech Stack

**Languages:** Python · JavaScript

**ML / Deep Learning:** PyTorch · Scikit-learn · Hugging Face Transformers · LSTM · CNN · BERT

**LLMs & Agents:** Anthropic Claude API · LangGraph · LangChain · Fine-tuning · Prompt Engineering

**RAG & Vector DBs:** ChromaDB · Sentence Transformers · ONNX embeddings · Semantic search

**MLOps:** Docker · MLflow · GitHub Actions CI/CD · Experiment tracking

**Web & UI:** Streamlit · Chrome Extensions (Manifest V3)

---

## 📂 Repository Structure

```
AI-Mastery-2026/
├── Month-01-Python-Git/
├── Month-02-Math-Stats/
├── Month-03-Data-Prep/
├── Month-04-ML-Neural-Nets/
├── Month-05-DL-CV-NLP/
├── Month-06-Transformers/
├── Month-07-GenerativeAI-LLMs/      # 🏆 AI Code Reviewer + Rate My Resume
├── Month-08-MLOps-LLMOps/           # 🏆 Study With Me Dashboard + CI/CD
├── Month-09-LLM-Orchestration/
├── Month-10-Vector-DBs/             # 🏆 Searchable Knowledge Base
├── Month-11-RAG/                    # 🏆 Knowledge Base Chatbot
└── Month-12-Agentic-AI/
    ├── multi_agent_research.py
    ├── multimodal_assistant.py
    ├── chrome_extension/            # 🏆 AI Summarizer Chrome Extension
    └── capstone_agent/              # 🏆 Autonomous Research Agent
```

---

## 🎯 Goal

Build production-ready AI engineering skills — from fundamentals to autonomous agents — while completing a BS in Computer Science.
