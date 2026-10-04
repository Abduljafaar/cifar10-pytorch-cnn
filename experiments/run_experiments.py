import sys
from pathlib import Path
import csv
import torch


# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Import project modules
# ============================================================

from src.dataset import get_dataloaders
from src.models import BaselineCNN, ImprovedCNN
from src.train import train_model


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64

LEARNING_RATE = 0.001

MOMENTUM = 0.9

SEED = 42


# ============================================================
# Experiment list
# ============================================================

EXPERIMENTS = [
    {
        "model": "baseline",
        "epochs": 2,
        "name": "baseline_2epochs"
    },
    {
        "model": "baseline",
        "epochs": 5,
        "name": "baseline_5epochs"
    },
    {
        "model": "baseline",
        "epochs": 10,
        "name": "baseline_10epochs"
    },
    {
        "model": "baseline",
        "epochs": 20,
        "name": "baseline_20epochs"
    },
    {
        "model": "improved",
        "epochs": 10,
        "name": "improved_10epochs"
    }
]


# ============================================================
# Create results directory
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Set random seed
# ============================================================

def set_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Create model
# ============================================================

def create_model(model_type):

    if model_type == "baseline":

        return BaselineCNN()

    elif model_type == "improved":

        return ImprovedCNN()

    else:

        raise ValueError(
            f"Unknown model type: {model_type}"
        )


# ============================================================
# Run experiments
# ============================================================

def run_experiments():

    print()
    print("=" * 70)
    print("CIFAR-10 CNN EXPERIMENTS")
    print("=" * 70)

    print()
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print(f"Momentum       : {MOMENTUM}")
    print(f"Random seed    : {SEED}")

    print()
    print("Experiments to run:")

    for experiment in EXPERIMENTS:

        print(
            f"  - {experiment['name']}"
        )

    print()
    print("=" * 70)


    # Store final results
    results = []


    # ========================================================
    # Run each experiment
    # ========================================================

    for experiment_number, experiment in enumerate(
        EXPERIMENTS,
        start=1
    ):

        model_type = experiment["model"]

        epochs = experiment["epochs"]

        experiment_name = experiment["name"]


        print()
        print()
        print("#" * 70)

        print(
            f"EXPERIMENT {experiment_number}"
        )

        print(
            f"Model  : {model_type.upper()} CNN"
        )

        print(
            f"Epochs : {epochs}"
        )

        print(
            f"Name   : {experiment_name}"
        )

        print("#" * 70)

        # Set seed
        set_seed(SEED)

        # ----------------------------------------------------
        # Load CIFAR-10
        # ----------------------------------------------------

        print()
        print("Loading CIFAR-10 dataset...")

        trainloader, testloader = get_dataloaders(
            batch_size=BATCH_SIZE
        )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = create_model(
            model_type
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        (
            trained_model,
            train_history,
            test_history
        ) = train_model(

            model= model,

            trainloader=trainloader,

            testloader=testloader,

            epochs=epochs,

            learning_rate=LEARNING_RATE,

            momentum=MOMENTUM,

            model_name=experiment_name
        )


        # ----------------------------------------------------
        # Get final accuracy
        # ----------------------------------------------------

        final_train_accuracy = (
            train_history[-1]
        )

        final_test_accuracy = (
            test_history[-1]
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "Model": (
                    "Baseline CNN"
                    if model_type == "baseline"
                    else "Improved CNN"
                ),

                "Epochs": epochs,

                "Train Accuracy": (
                    final_train_accuracy
                ),

                "Test Accuracy": (
                    final_test_accuracy
                ),

                "Model File": (
                    f"models/{experiment_name}.pth"
                )
            }
        )


        # ----------------------------------------------------
        # Print experiment result
        # ----------------------------------------------------

        print()
        print("-" * 70)

        print(
            f"Experiment completed: "
            f"{experiment_name}"
        )

        print(
            f"Final Training Accuracy: "
            f"{final_train_accuracy:.2f}%"
        )

        print(
            f"Final Test Accuracy: "
            f"{final_test_accuracy:.2f}%"
        )

        print("-" * 70)


    # ========================================================
    # Save results to CSV
    # ========================================================

    results_file = (
        RESULTS_DIR /
        "experiment_results.csv"
    )


    with open(
        results_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Model",
                "Epochs",
                "Train Accuracy",
                "Test Accuracy",
                "Model File"
            ]
        )

        writer.writeheader()

        writer.writerows(results)


    # ========================================================
    # Final comparison table
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL EXPERIMENT RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Model':<18}"
        f"{'Epochs':<10}"
        f"{'Train Acc':<15}"
        f"{'Test Acc':<15}"
    )

    print("-" * 58)


    for result in results:

        print(
            f"{result['Model']:<18}"
            f"{result['Epochs']:<10}"
            f"{result['Train Accuracy']:<15.2f}"
            f"{result['Test Accuracy']:<15.2f}"
        )


    print()
    print("=" * 70)

    print(
        f"Results saved to:"
    )

    print(
        results_file
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_experiments()