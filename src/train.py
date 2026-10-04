import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_model(
    model,
    trainloader,
    testloader,
    epochs=10,
    learning_rate=0.001,
    momentum=0.9,
    model_name="model"
):
    """
    Train a CNN model and evaluate it after every epoch.

    Returns:
        model
        train_accuracy_history
        test_accuracy_history
    """

    # Move model to CPU or GPU
    model = model.to(DEVICE)

    print(f"Training device: {DEVICE}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    # Store accuracy after every epoch
    train_accuracy_history = []
    test_accuracy_history = []

    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------

    for epoch in range(epochs):

        # ==============================================
        # TRAINING
        # ==============================================

        model.train()

        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels in trainloader:

            # Move data to device
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Calculate predictions
            _, predicted = torch.max(outputs, 1)

            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        # Training accuracy
        train_accuracy = 100.0 * correct / total

        # Average training loss
        average_loss = running_loss / len(trainloader)

        # ==============================================
        # TESTING
        # ==============================================

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in testloader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # Forward pass
                outputs = model(images)

                # Prediction
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        # Test accuracy
        test_accuracy = 100.0 * correct / total

        # Store history
        train_accuracy_history.append(
            train_accuracy
        )

        test_accuracy_history.append(
            test_accuracy
        )

        # Print results
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.4f} "
            f"Train Accuracy: {train_accuracy:.2f}% "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    models_directory = Path("models")

    models_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        models_directory /
        f"{model_name}.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print()
    print(
        f"Model saved to: {model_path}"
    )

    return (
        model,
        train_accuracy_history,
        test_accuracy_history
    )


# --------------------------------------------------
# Test training function
# --------------------------------------------------

if __name__ == "__main__":

    from dataset import get_dataloaders
    from models import BaselineCNN

    print("Testing training pipeline...")
    print()

    # Load CIFAR-10
    trainloader, testloader = get_dataloaders(
        batch_size=64
    )

    # Create baseline model
    model = BaselineCNN()

    # Test with only 1 epoch
    #
    # We use 1 epoch here only to verify
    # that the complete training pipeline works.
    trained_model, train_history, test_history = (
        train_model(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            epochs=1,
            model_name="training_test"
        )
    )

    print()
    print("Training pipeline test completed!")
    print(
        f"Final Train Accuracy: "
        f"{train_history[-1]:.2f}%"
    )

    print(
        f"Final Test Accuracy: "
        f"{test_history[-1]:.2f}%"
    )