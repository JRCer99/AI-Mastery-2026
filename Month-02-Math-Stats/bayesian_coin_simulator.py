"""
Month 02 — Project 3: Bayesian Probability Simulator (Coin Flipping)
Demonstrates Bayesian updating: how beliefs change as evidence accumulates.
Key concept: prior + likelihood → posterior (Bayes' Theorem in action).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta as beta_dist  # renamed to avoid clash with self.beta

OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


class BayesianCoinSimulator:
    def __init__(self, true_bias=0.7, prior_alpha=2, prior_beta=2):
        self.true_bias = true_bias          # hidden true P(Heads)
        self.alpha = prior_alpha            # Beta shape param (Heads pseudocount)
        self.beta = prior_beta              # Beta shape param (Tails pseudocount)
        self.initial_alpha = prior_alpha    # saved for prior plotting
        self.initial_beta = prior_beta
        self.snapshots = []                 # (alpha, beta, n_flips) after each update

    def flip_coin(self):
        return np.random.rand() < self.true_bias

    def update_posterior(self, new_heads, new_tails):
        """Conjugate update: Beta(α,β) + (H,T) → Beta(α+H, β+T)"""
        self.alpha += new_heads
        self.beta += new_tails

    def plot_snapshot(self, n_flips, total_heads):
        x = np.linspace(0, 1, 300)
        prior = beta_dist.pdf(x, self.initial_alpha, self.initial_beta)
        posterior = beta_dist.pdf(x, self.alpha, self.beta)
        posterior_mean = self.alpha / (self.alpha + self.beta)

        plt.figure(figsize=(10, 6))
        plt.plot(x, prior, "b--", alpha=0.6,
                 label=f"Prior  Beta({self.initial_alpha}, {self.initial_beta})")
        plt.plot(x, posterior, "r-", linewidth=2.5,
                 label=f"Posterior  Beta({self.alpha}, {self.beta})\nMean = {posterior_mean:.3f}")
        plt.axvline(self.true_bias, color="green", linestyle=":", linewidth=2,
                    label=f"True bias = {self.true_bias:.1f}")
        plt.title(f"Bayesian Update — {n_flips} flips  "
                  f"({total_heads} Heads, {n_flips - total_heads} Tails)\n"
                  f"True bias = {self.true_bias:.1%}", fontsize=13)
        plt.xlabel("Probability of Heads (coin bias)")
        plt.ylabel("Density")
        plt.xlim(0, 1)
        plt.legend()
        plt.tight_layout()

        path = f"{OUTPUT_DIR}/bayesian_update_{n_flips:04d}_flips.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_summary(self):
        """Overlay all posterior snapshots to show belief evolution."""
        x = np.linspace(0, 1, 300)
        _, ax = plt.subplots(figsize=(12, 7))

        cmap = plt.cm.Reds
        n = len(self.snapshots)
        for i, (a, b, flips) in enumerate(self.snapshots):
            color = cmap(0.3 + 0.7 * i / max(n - 1, 1))
            ax.plot(x, beta_dist.pdf(x, a, b), color=color,
                    alpha=0.8, linewidth=1.5, label=f"n={flips}")

        ax.plot(x, beta_dist.pdf(x, self.initial_alpha, self.initial_beta),
                "b--", linewidth=2, label="Prior (initial)", alpha=0.7)
        ax.axvline(self.true_bias, color="green", linestyle=":", linewidth=2,
                   label=f"True bias = {self.true_bias:.1f}")

        ax.set_title("Belief Evolution — All Posteriors", fontsize=14)
        ax.set_xlabel("Probability of Heads")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        plt.tight_layout()

        path = f"{OUTPUT_DIR}/bayesian_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Summary plot saved → {path}")
        return path

    def run_simulation(self, num_flips=200, update_every=20):
        print("🚀 Bayesian Coin Flip Simulator")
        print(f"True coin bias (hidden): {self.true_bias:.1%} Heads")
        print(f"Starting prior: Beta({self.alpha}, {self.beta}) — weakly assumes fair coin\n")

        total_heads = 0
        batch_heads = 0

        for i in range(1, num_flips + 1):
            if self.flip_coin():
                total_heads += 1
                batch_heads += 1

            if i % update_every == 0:
                batch_tails = update_every - batch_heads
                self.update_posterior(batch_heads, batch_tails)
                self.snapshots.append((self.alpha, self.beta, i))
                self.plot_snapshot(i, total_heads)

                posterior_mean = self.alpha / (self.alpha + self.beta)
                print(f"After {i:3d} flips → {total_heads} Heads ({total_heads/i:.1%}) | "
                      f"Posterior Beta({self.alpha}, {self.beta}) | "
                      f"Estimated bias = {posterior_mean:.3f}")
                batch_heads = 0

        self.plot_summary()
        print(f"\n🎉 Simulation complete!")
        print(f"Final posterior: Beta({self.alpha}, {self.beta})")
        final_mean = self.alpha / (self.alpha + self.beta)
        print(f"Estimated bias: {final_mean:.3f}  |  True bias: {self.true_bias:.3f}")


def main():
    simulator = BayesianCoinSimulator(true_bias=0.7, prior_alpha=2, prior_beta=2)
    simulator.run_simulation(num_flips=200, update_every=20)

    print("\n✅ Month 02 Project 3 Complete!")
    print("Key insight: more data → posterior narrows and converges to true value.")


if __name__ == "__main__":
    main()
