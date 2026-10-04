import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Baseline CNN based on the PyTorch CIFAR-10 tutorial.

    Architecture:
        Conv2D: 3  -> 6
        ReLU
        MaxPool

        Conv2D: 6  -> 16
        ReLU
        MaxPool

        Fully Connected: 400 -> 120
        Fully Connected: 120 -> 84
        Fully Connected: 84 -> 10
    """

    def __init__(self):
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            16 * 5 * 5,
            120
        )

        self.fc2 = nn.Linear(
            120,
            84
        )

        self.fc3 = nn.Linear(
            84,
            10
        )

    def forward(self, x):

        # Conv1 -> ReLU -> MaxPool
        x = self.pool(
            F.relu(
                self.conv1(x)
            )
        )

        # Conv2 -> ReLU -> MaxPool
        x = self.pool(
            F.relu(
                self.conv2(x)
            )
        )

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        # Final output
        x = self.fc3(x)

        return x


class ImprovedCNN(nn.Module):
    """
    Improved CNN with additional convolutional layers
    and dropout.

    Architecture:
        Conv2D: 3   -> 32
        ReLU

        Conv2D: 32  -> 64
        ReLU
        MaxPool

        Conv2D: 64  -> 128
        ReLU
        MaxPool

        Fully Connected: 8192 -> 512
        Dropout
        Fully Connected: 512 -> 10
    """

    def __init__(self):
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Third convolutional layer
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Fully connected layers
        #
        # Input image:
        # 32 x 32
        #
        # After first pooling:
        # 16 x 16
        #
        # After second pooling:
        # 8 x 8
        #
        # Therefore:
        # 128 x 8 x 8 = 8192
        self.fc1 = nn.Linear(
            128 * 8 * 8,
            512
        )

        self.fc2 = nn.Linear(
            512,
            10
        )

        # Dropout for reducing overfitting
        self.dropout = nn.Dropout(
            p=0.5
        )

    def forward(self, x):

        # Conv1 -> ReLU
        x = F.relu(
            self.conv1(x)
        )

        # Conv2 -> ReLU -> MaxPool
        x = F.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        # Conv3 -> ReLU -> MaxPool
        x = F.relu(
            self.conv3(x)
        )

        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = F.relu(
            self.fc1(x)
        )

        # Dropout
        x = self.dropout(x)

        # Final output
        x = self.fc2(x)

        return x


# --------------------------------------------------
# Test both models
# --------------------------------------------------

if __name__ == "__main__":

    print("Testing CNN models...")
    print()

    # Create models
    baseline_model = BaselineCNN()
    improved_model = ImprovedCNN()

    # Create dummy CIFAR-10 input
    #
    # 4 images
    # 3 channels (RGB)
    # 32 x 32 pixels
    test_input = torch.randn(
        4,
        3,
        32,
        32
    )

    # Forward pass
    baseline_output = baseline_model(
        test_input
    )

    improved_output = improved_model(
        test_input
    )

    # Display output shapes
    print(
        "Baseline CNN output shape:",
        baseline_output.shape
    )

    print(
        "Improved CNN output shape:",
        improved_output.shape
    )

    print()

    print(
        "Expected output shape:",
        torch.Size([4, 10])
    )

    print()

    # Check whether shapes are correct
    if baseline_output.shape == torch.Size([4, 10]):
        print("Baseline CNN test: PASSED")
    else:
        print("Baseline CNN test: FAILED")

    if improved_output.shape == torch.Size([4, 10]):
        print("Improved CNN test: PASSED")
    else:
        print("Improved CNN test: FAILED")