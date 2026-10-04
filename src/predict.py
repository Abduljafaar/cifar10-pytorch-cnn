import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from models import ImprovedCNN


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/improved_10epochs.pth"
OUTPUT_DIR = "results/predictions"
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "sample_predictions.png")

NUM_IMAGES = 16

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CIFAR-10 PREDICTION")
print("=" * 60)
print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print()


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# Load CIFAR-10 Test Dataset
# ============================================================

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

print(f"Test images available: {len(test_dataset)}")
print()


# ============================================================
# Load Model
# ============================================================

model = ImprovedCNN()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

# Handle both direct state_dict and checkpoint dictionaries
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# Prediction
# ============================================================

images = []
actual_labels = []
predicted_labels = []

correct = 0


with torch.no_grad():

    for i in range(NUM_IMAGES):

        image, label = test_dataset[i]

        # Add batch dimension
        input_image = image.unsqueeze(0).to(device)

        # Model prediction
        output = model(input_image)

        prediction = torch.argmax(output, dim=1).item()

        # Store results
        images.append(image)
        actual_labels.append(label)
        predicted_labels.append(prediction)

        if prediction == label:
            correct += 1


# ============================================================
# Print Predictions
# ============================================================

print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(NUM_IMAGES):

    actual = CLASS_NAMES[actual_labels[i]]
    predicted = CLASS_NAMES[predicted_labels[i]]

    result = "CORRECT" if actual == predicted else "WRONG"

    print(
        f"{i + 1:02d}. "
        f"Actual: {actual:<10} "
        f"Predicted: {predicted:<10} "
        f"[{result}]"
    )


sample_accuracy = (correct / NUM_IMAGES) * 100

print()
print("=" * 60)
print(f"Sample Accuracy: {sample_accuracy:.2f}%")
print("=" * 60)


# ============================================================
# Create Output Directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for i, ax in enumerate(axes.flat):

    # Convert image back from normalized format
    image = images[i].permute(1, 2, 0).numpy()

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    image = image * std + mean

    # Keep pixel values valid
    image = image.clip(0, 1)

    ax.imshow(image)

    actual = CLASS_NAMES[actual_labels[i]]
    predicted = CLASS_NAMES[predicted_labels[i]]

    ax.set_title(
        f"Actual: {actual}\nPredicted: {predicted}",
        fontsize=9
    )

    ax.axis("off")


plt.suptitle(
    "CIFAR-10 Improved CNN Predictions",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_IMAGE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print()
print(f"Prediction image saved to:")
print(OUTPUT_IMAGE)
print()
print("Prediction completed successfully!")