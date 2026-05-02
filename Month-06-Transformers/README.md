# Month 06: Transformers & Transfer Learning

**Focus:** Fine-tune pre-trained transformer models for custom NLP tasks and apply transfer learning for domain adaptation.

---

## 🎯 Learning Goals

- Understand attention mechanisms and transformer architecture
- Fine-tune BERT/RoBERTa on custom classification datasets
- Apply transfer learning to adapt models to new domains
- Evaluate NLP models: accuracy, F1, confusion matrix

---

## 📅 Projects

- [x] Project 1: Fine-tuned BERT for Custom Text Classification ✅
  - Fine-tune BERT on a domain-specific dataset
  - Libraries: `transformers`, `torch`, `datasets`

- [ ] Project 2: Transfer Learning Demo (Domain Adaptation)
  - Adapt a pre-trained model to a new domain/task
  - Libraries: `transformers`, `torch`, `sklearn`

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| Self-attention | Core mechanism behind transformers |
| Fine-tuning | Adapt pre-trained weights to new tasks |
| Tokenization | WordPiece / BPE encoding for BERT |
| Transfer learning | Leverage large pre-trained models |
| Freezing layers | Control which weights update during training |

---

## 📚 Resources

- [HuggingFace — Fine-tuning a Pretrained Model](https://huggingface.co/docs/transformers/training)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## 📂 Structure

```
Month-06-Transformers/
├── bert_classifier.py          # Project 1: Fine-tuned BERT
└── transfer_learning_demo.py   # Project 2: Domain Adaptation
```
