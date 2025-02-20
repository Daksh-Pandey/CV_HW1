from torchvision.models import resnet18, ResNet18_Weights
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, input_mlp_dim, num_classes):
        super(ConvNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, stride=4),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(input_mlp_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(in_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
