import torch

from dataset import get_dataloaders
from models import BaselineCNN, ImprovedCNN


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# CIFAR-10 classes
# --------------------------------------------------

CLASSES = (
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
)


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate_model(
    model,
    testloader
):
    """
    Evaluate a trained model on the CIFAR-10
    test dataset.

    Returns:
        test_accuracy
        class_accuracy
    """

    model = model.to(DEVICE)

    # Evaluation mode
    model.eval()

    correct = 0
    total = 0

    # Per-class statistics
    class_correct = [
        0 for _ in range(10)
    ]

    class_total = [
        0 for _ in range(10)
    ]

    # No gradients required during evaluation
    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Get predicted class
            _, predicted = torch.max(
                outputs,
                1
            )

            # Total accuracy
            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            # Class-wise accuracy
            for label, prediction in zip(
                labels,
                predicted
            ):

                label = label.item()
                prediction = prediction.item()

                class_total[label] += 1

                if label == prediction:
                    class_correct[label] += 1

    # Overall test accuracy
    test_accuracy = (
        100.0 * correct / total
    )

    # Class-wise accuracy
    class_accuracy = []

    for i in range(10):

        accuracy = (
            100.0
            * class_correct[i]
            / class_total[i]
        )

        class_accuracy.append(
            accuracy
        )

    return (
        test_accuracy,
        class_accuracy
    )


# --------------------------------------------------
# Load saved model
# --------------------------------------------------

def load_model(
    model,
    model_path
):
    """
    Load trained model weights.
    """

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model = model.to(DEVICE)

    return model


# --------------------------------------------------
# Print evaluation results
# --------------------------------------------------

def print_results(
    model_name,
    test_accuracy,
    class_accuracy
):

    print()
    print("=" * 50)
    print(f"{model_name} Evaluation Results")
    print("=" * 50)

    print(
        f"Overall Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print()
    print("Class-wise Accuracy:")
    print("-" * 40)

    for class_name, accuracy in zip(
        CLASSES,
        class_accuracy
    ):

        print(
            f"{class_name:<12}: "
            f"{accuracy:.2f}%"
        )

    print("=" * 50)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting model evaluation...")
    print()

    # Load CIFAR-10
    trainloader, testloader = get_dataloaders(
        batch_size=64
    )

    # --------------------------------------------------
    # Evaluate final improved model
    # --------------------------------------------------

    improved_model = ImprovedCNN()

    improved_model = load_model(
        improved_model,
        "models/improved_10epochs.pth"
    )

    (
        improved_accuracy,
        improved_class_accuracy
    ) = evaluate_model(
        improved_model,
        testloader
    )

    print_results(
        "Improved CNN - 10 Epochs",
        improved_accuracy,
        improved_class_accuracy
    )

    print()
    print("Evaluation completed successfully!")