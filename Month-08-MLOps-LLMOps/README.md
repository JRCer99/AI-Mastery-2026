# Month 08: MLOps & LLMOps

**Focus:** Production-grade ML pipelines — containerization, experiment tracking, CI/CD, and live deployment.

---

## 🎯 Learning Goals

- Containerize ML models with Docker
- Track experiments with MLflow
- Build CI/CD pipelines with GitHub Actions
- Deploy a production-ready dashboard or ML service

---

## 📅 Projects

- [x] Project 1: Dockerized + MLflow Tracked Model ✅
  - Containerize a trained model, track runs with MLflow
  - Libraries: `mlflow`, `docker`, `fastapi` or `flask`

- [x] Project 2: CI/CD Pipeline with GitHub Actions ✅
  - Automate test → lint → deploy on push
  - Tools: GitHub Actions, pytest, flake8

- [ ] Project 3: 🏆 Study With Me Live Dashboard *(Weekend Project)*
  - Real-time study session tracker with live stats
  - Libraries: `streamlit`, `sqlite3` or `firebase`

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| Docker | Reproducible environments for ML |
| MLflow | Experiment tracking, model registry |
| CI/CD | Automate quality checks on every push |
| Model serving | REST API for real-time inference |
| Monitoring | Track model drift in production |

---

## 📚 Resources

- [MLflow — Official Docs](https://mlflow.org/docs/latest/index.html)
- [Docker — Getting Started](https://docs.docker.com/get-started/)
- [GitHub Actions — ML Workflow](https://docs.github.com/en/actions)

---

## 📂 Structure

```
Month-08-MLOps-LLMOps/
├── dockerized_model/           # Project 1: Docker + MLflow
├── .github/workflows/          # Project 2: CI/CD Pipeline
└── study_with_me_dashboard.py  # Project 3: 🏆 Live Dashboard
```
