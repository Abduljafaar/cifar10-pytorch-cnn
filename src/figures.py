import os

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

RESULTS_FILE = "results/experiment_results.csv"
FIGURES_DIR = "results/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)


# --------------------------------------------------
# Load experiment results
# --------------------------------------------------

results = pd.read_csv(RESULTS_FILE)

print("Experiment results loaded successfully!")
print(results)


# --------------------------------------------------
# Figure 1: Accuracy vs Epochs
# --------------------------------------------------

print()
print("Creating Figure 1: Accuracy vs Epochs...")

baseline_results = results[
    results["Model"].str.contains(
        "Baseline",
        case=False,
        na=False
    )
]

plt.figure(figsize=(8, 5))

plt.plot(
    baseline_results["Epochs"],
    baseline_results["Test Accuracy"],
    marker="o",
    label="Baseline CNN"
)

improved_results = results[
    results["Model"].str.contains(
        "Improved",
        case=False,
        na=False
    )
]

plt.plot(
    improved_results["Epochs"],
    improved_results["Test Accuracy"],
    marker="o",
    label="Improved CNN"
)

plt.xlabel("Number of Epochs")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Number of Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "accuracy_vs_epochs.png"
    )
)

plt.close()

print(
    "Saved: "
    "results/figures/accuracy_vs_epochs.png"
)


# --------------------------------------------------
# Figure 2: Model Comparison
# --------------------------------------------------

print()
print("Creating Figure 2: Model Comparison...")

model_names = [
    "Baseline CNN\n20 Epochs",
    "Improved CNN\n10 Epochs"
]

model_accuracies = [
    61.89,
    62.52
]

plt.figure(figsize=(7, 5))

plt.bar(
    model_names,
    model_accuracies
)

plt.ylabel("Test Accuracy (%)")
plt.title("Final Model Comparison")
plt.ylim(0, 100)

for i, accuracy in enumerate(
    model_accuracies
):
    plt.text(
        i,
        accuracy + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "model_comparison.png"
    )
)

plt.close()

print(
    "Saved: "
    "results/figures/model_comparison.png"
)


# --------------------------------------------------
# Figure 3: Final Model Class-wise Accuracy
# --------------------------------------------------

print()
print("Creating Figure 3: Final Model Class-wise Accuracy...")


# These values come from the final
# Improved CNN trained for 10 epochs.

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

class_accuracies = [
    69.60,
    70.60,
    45.60,
    35.00,
    51.20,
    53.10,
    82.50,
    71.40,
    72.60,
    73.60
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    classes,
    class_accuracies
)

plt.xlabel("CIFAR-10 Class")
plt.ylabel("Accuracy (%)")
plt.title(
    "Class-wise Accuracy - Improved CNN (10 Epochs)"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(0, 100)

# Display accuracy value above each bar
for bar, accuracy in zip(
    bars,
    class_accuracies
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "class_accuracy.png"
    )
)

plt.close()

print(
    "Saved: "
    "results/figures/class_accuracy.png"
)


# --------------------------------------------------
# Completion message
# --------------------------------------------------

print()
print("=" * 60)
print("FIGURE GENERATION COMPLETED")
print("=" * 60)

print()
print("Generated files:")

print(
    "1. results/figures/"
    "accuracy_vs_epochs.png"
)

print(
    "2. results/figures/"
    "model_comparison.png"
)

print(
    "3. results/figures/"
    "class_accuracy.png"
)

print()
print(
    "All figures are ready for the assignment."
)