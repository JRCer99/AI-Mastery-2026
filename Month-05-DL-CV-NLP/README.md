# Month 05: Deep Learning, Computer Vision & NLP

**Focus:** Build neural networks from scratch using PyTorch — CNNs for image classification, sentiment analysis with NLP, and sequence prediction.

---

## 🎯 Learning Goals

- Understand forward pass, backpropagation, and gradient descent in neural networks
- Build and train CNNs for image classification with PyTorch
- Apply NLP preprocessing: tokenization, embeddings, sentiment analysis
- Build a sequence prediction model (RNN/LSTM)
- Evaluate deep learning models: accuracy, loss curves, confusion matrix

---

## 📅 Projects

- [x] Project 1: CNN Image Classifier (Fashion MNIST) ✅
  - 10-class image classification with a custom CNN
  - Libraries: `torch`, `torchvision`, `matplotlib`, `numpy`

- [ ] Project 2: Sentiment Analysis Web App
  - Train NLP model on movie/product reviews, simple web UI
  - Libraries: `torch`, `transformers` or `sklearn`, `flask`

- [ ] Project 3: Sequence Prediction Model
  - LSTM-based sequence predictor (time series or text)
  - Libraries: `torch`, `numpy`, `matplotlib`

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| Convolutional layers | Extract spatial features from images |
| Backpropagation | How neural networks learn from error |
| Dropout / Batch Norm | Prevent overfitting in deep networks |
| Embeddings | Dense numeric representation of text |
| LSTM / RNN | Model sequential / time-dependent data |
| Loss curves | Diagnose underfitting vs overfitting |

---

## 📚 Resources

- [PyTorch — Official 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Fast.ai — Practical Deep Learning](https://course.fast.ai/)
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

---

## 📂 Structure

```
Month-05-DL-CV-NLP/
├── cnn_image_classifier.py     # Project 1: CNN Fashion MNIST
├── sentiment_analysis.py       # Project 2: Sentiment Web App
├── sequence_predictor.py       # Project 3: LSTM Sequence Model
├── fashion_mnist_cnn.pth       # Saved model (gitignored)
└── data/                       # datasets (gitignored)
```
