"""
Month 02 — Project 2: Gradient Descent from Scratch Visualizer
Demonstrates how gradient descent navigates a loss surface to find the minimum.
Key insight: learning rate controls speed vs stability of convergence.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def objective_function(x):
    """Loss surface: f(x) = x²  (minimum at x=0)"""
    return x ** 2


def gradient(x):
    """df/dx = 2x"""
    return 2 * x


def gradient_descent(start_x, learning_rate=0.15, iterations=30):
    x = start_x
    history = [x]

    for i in range(iterations):
        x = x - learning_rate * gradient(x)
        history.append(x)
        if i % 10 == 0:
            print(f"  Iter {i:2d} | x = {x:.4f} | f(x) = {objective_function(x):.4f}")

    return np.array(history)


def visualize_gradient_descent():
    x = np.linspace(-10, 10, 400)
    y = objective_function(x)

    starts = [-8.0, -3.0, 6.0]
    colors = ["red", "green", "orange"]

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, "b-", label="f(x) = x²  (loss surface)", linewidth=2.5)

    for start, color in zip(starts, colors):
        print(f"\n📍 Starting at x = {start}")
        history = gradient_descent(start, learning_rate=0.15, iterations=30)
        fy = objective_function(history)

        plt.plot(history, fy, color=color, marker="o", markersize=4,
                 linestyle="--", alpha=0.8, label=f"Start x={start}")
        plt.plot(history[-1], fy[-1], "x", color=color,
                 markersize=12, markeredgewidth=3)

    plt.title("Gradient Descent from Scratch — Visualizer", fontsize=16, pad=20)
    plt.xlabel("x  (parameter)")
    plt.ylabel("f(x)  (loss)")
    plt.legend()
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/gradient_descent_visualization.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Visualization saved → {path}")
    return path


def compare_learning_rates():
    """Show how learning rate affects convergence speed and stability."""
    rates = [0.01, 0.1, 0.5, 0.95]
    colors = ["blue", "green", "orange", "red"]
    start = -8.0

    plt.figure(figsize=(12, 5))

    # Left: loss over iterations per learning rate
    plt.subplot(1, 2, 1)
    for lr, color in zip(rates, colors):
        history = gradient_descent(start, learning_rate=lr, iterations=50)
        plt.plot(objective_function(history), color=color, label=f"lr={lr}")
    plt.title("Loss vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("f(x) loss")
    plt.legend()
    plt.yscale("log")

    # Right: descent paths on loss surface
    plt.subplot(1, 2, 2)
    x_range = np.linspace(-10, 10, 400)
    plt.plot(x_range, objective_function(x_range), "k-", linewidth=2, label="f(x) = x²")
    for lr, color in zip(rates, colors):
        history = gradient_descent(start, learning_rate=lr, iterations=50)
        plt.plot(history, objective_function(history), color=color,
                 marker="o", markersize=3, linestyle="--", alpha=0.7, label=f"lr={lr}")
    plt.title("Descent Paths")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()

    plt.suptitle("Learning Rate Comparison", fontsize=14)
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/learning_rate_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Learning rate comparison saved → {path}")
    return path


def main():
    print("🚀 Gradient Descent from Scratch Visualizer")
    print("="*60)

    p1 = visualize_gradient_descent()

    print("\n📊 Comparing learning rates...")
    p2 = compare_learning_rates()

    print("\n🎉 Month 02 Project 2 Complete!")
    print("Key insight: small lr = slow but stable | large lr = fast but can overshoot")
    print(f"\nOutputs:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
