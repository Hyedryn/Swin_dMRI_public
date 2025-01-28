import torch.nn as nn
from torchvision.models import resnet18, resnet34


class ResNet(nn.Module):
    def __init__(self, num_classes, size, freeze_mode="allButStage4"):
        super(ResNet, self).__init__()

        assert freeze_mode in ["allButStage4", "all", "None"], "Unknown freeze mode"

        # Load the ResNet model
        if size == "resnet18":
            self.resnet = resnet18(weights='IMAGENET1K_V1')
        elif size == "resnet34":
            self.resnet = resnet34(weights='IMAGENET1K_V1')
        else:
            raise Exception("Unknown size")

        # Freeze the backbone if required
        if freeze_mode in ["allButStage4", "all"]:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Unfreeze the last layer if required
        if freeze_mode == "allButStage4":
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

        # Modify the classification head
        num_features = self.resnet.fc.in_features
        print("ResNet num_features", num_features)
        self.resnet.fc = nn.Linear(num_features, num_classes)

        # Ensure the classification head's parameters are trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
