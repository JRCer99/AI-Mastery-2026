import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_data(seq_length=1000, window=50):
    x = np.linspace(0, 50, seq_length)
    data = np.sin(x) + np.random.normal(0, 0.1, seq_length)

    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])

    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y).reshape(-1, 1)

    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).to(device)

    return X, y, data

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step only
        return out

def train_model(model, X, y, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Training LSTM...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    print("✅ Training Complete!")

def visualize_predictions(model, X, data):
    model.eval()
    with torch.no_grad():
        predictions = model(X).cpu().numpy().flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(data[50:], label='Actual Data', alpha=0.7)
    plt.plot(range(50, len(predictions)+50), predictions, label='LSTM Predictions', alpha=0.9)
    plt.title('LSTM Sequence Prediction vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lstm_predictions.png', dpi=300)
    print("✅ Prediction visualization saved as 'lstm_predictions.png'")
    plt.show()

def main():
    X, y, data = generate_data()
    model = LSTMModel().to(device)

    train_model(model, X, y, epochs=100)
    visualize_predictions(model, X, data)

    torch.save(model.state_dict(), 'lstm_sequence_model.pth')
    print("✅ Model saved as lstm_sequence_model.pth")

    print(f"\n🎉 Month 5 Project 3 Complete! - {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
