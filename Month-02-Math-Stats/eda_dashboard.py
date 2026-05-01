"""
Month 02 — Project 1: Exploratory Data Analysis Dashboard
Loads Titanic dataset, computes stats, generates visualizations, outputs Markdown report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


def load_dataset():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    print("\n" + "="*50)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("="*50)
    print("\n1. Basic Information:")
    print(df.info())
    print("\n2. Summary Statistics:")
    print(df.describe())
    print("\n3. Missing Values:")
    print(df.isnull().sum())


def visualize_distributions(df):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='Age', kde=True, bins=30)
    plt.title('Age Distribution')

    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='Fare', kde=True, bins=30)
    plt.title('Fare Distribution')

    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='Pclass', hue='Survived')
    plt.title('Survival by Passenger Class')

    plt.subplot(2, 2, 4)
    sns.countplot(data=df, x='Sex', hue='Survived')
    plt.title('Survival by Gender')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/distributions.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Distributions saved → {path}")
    return path


def plot_correlation_heatmap(df):
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return None
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Correlation heatmap saved → {path}")
    return path


def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("ℹ️  No missing values to plot.")
        return None
    _, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind="bar", ax=ax, color="tomato", edgecolor="black")
    ax.set_title("Missing Values per Column")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/missing_values.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Missing values chart saved → {path}")
    return path


def generate_report(df, plot_paths):
    numeric = df.select_dtypes(include=np.number)
    rows, cols = df.shape
    missing_total = int(df.isnull().sum().sum())
    survival_rate = f"{df['Survived'].mean():.1%}" if 'Survived' in df.columns else "N/A"

    lines = [
        "# EDA Report — Titanic Dataset",
        f"*Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}*",
        "",
        "## Dataset Overview",
        f"- **Rows:** {rows}",
        f"- **Columns:** {cols}",
        f"- **Missing values:** {missing_total} total",
        f"- **Numeric columns:** {len(numeric.columns)}",
        f"- **Survival rate:** {survival_rate}",
        "",
        "## Summary Statistics",
        numeric.describe().round(3).to_markdown(),
        "",
        "## Missing Values",
    ]

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("- No missing values.")
    else:
        for col, cnt in missing.items():
            lines.append(f"- **{col}**: {cnt} ({cnt/rows*100:.1f}%)")

    lines += ["", "## Visualizations"]
    for path in filter(None, plot_paths):
        label = os.path.basename(path).replace("_", " ").replace(".png", "").title()
        lines.append(f"\n### {label}\n![{label}]({path})")

    lines += [
        "",
        "## Next Steps",
        "- Handle missing values (imputation strategies)",
        "- Feature engineering for ML pipeline (Month 04)",
        "- Build predictive model (survival classification)",
        "",
        "---",
        "*AI Mastery 2026 — Month 02, Project 1*",
    ]

    report_path = f"{OUTPUT_DIR}/eda_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Report saved → {report_path}")
    return report_path


def main():
    df = load_dataset()
    explore_data(df)

    print("\n📊 Generating visualizations...")
    plot_paths = [
        visualize_distributions(df),
        plot_correlation_heatmap(df),
        plot_missing_values(df),
    ]

    print("\n📝 Generating report...")
    generate_report(df, plot_paths)

    print("\n🎉 Month 02 Project 1 Complete! Outputs in 'eda_output/'")


if __name__ == "__main__":
    main()
