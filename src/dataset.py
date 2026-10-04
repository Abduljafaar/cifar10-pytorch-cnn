import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(batch_size=64):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )

    return trainloader, testloader


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


if __name__ == "__main__":

    trainloader, testloader = get_dataloaders()

    images, labels = next(iter(trainloader))

    print("CIFAR-10 loaded successfully!")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Classes:", CLASSES)